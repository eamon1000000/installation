import os
import io
import math
import base64
import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TVF
from diffusers import StableDiffusionInpaintPipeline
from ultralytics import YOLO
from openai import OpenAI


def load_models(hf_token, device="cuda"):
    print("Loading Stable Diffusion inpainting pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    print(f"  SD pipeline loaded on {device}.")

    print("Loading YOLO segmentation model...")
    yolo = YOLO("yolov8m-seg.pt")
    print("  YOLO loaded.")

    return pipe, yolo


# ── Object detection ───────────────────────────────────────────────────────────

def generate_object_masks(image, model, sort_by="confidence",
                           feather_radius=11, min_confidence=0.25, min_area=500):
    img_w, img_h = image.size
    results = model(image, verbose=False)[0]

    if results.masks is None:
        return []

    detections = []
    for mask_xy, conf, cls_id in zip(results.masks.xy, results.boxes.conf, results.boxes.cls):
        conf_val = float(conf)
        label = model.names[int(cls_id)]

        if conf_val < min_confidence:
            continue

        mask_array = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask_array, [mask_xy.astype(np.int32)], 255)

        area = int((mask_array > 0).sum())
        if area < min_area:
            continue

        mask_pil = Image.fromarray(mask_array).convert("L")
        if feather_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))

        detections.append({
            "mask":       mask_pil,
            "label":      label,
            "confidence": conf_val,
            "area":       area,
        })

    if sort_by == "confidence":
        detections.sort(key=lambda d: d["confidence"], reverse=True)
    elif sort_by == "area":
        detections.sort(key=lambda d: d["area"], reverse=True)

    print(f"YOLO: {len(detections)} object(s) detected.")
    for i, d in enumerate(detections):
        print(f"  [{i}] {d['label']:20s} conf={d['confidence']:.2f}  "
              f"area={d['area']:,}px ({100*d['area']/(img_w*img_h):.1f}%)")

    return detections


# ── Prompt generation ──────────────────────────────────────────────────────────

def image_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_inpaint_prompt(image, client, iteration=0, detected_label=None):
    instructions = (
        "You are a prompt generator for an AI inpainting model "
        "that is part of a recursive art installation. A photograph is being "
        "viewed by a vision model, with objects classified and then segmented. "
        "The segmented masks will then be iteratively inpainted based on the prompts you provide. "
        "At each step, a region is erased and you will repaint it based on the object's classification "
        "as well as your own interpretation of the image.\n\n"
        "Your job: look at the current state of the image and write a concise "
        f"inpainting prompt (1-2 sentences max) that describes the scene and infill the object {detected_label}. "
        f"Include key visual details  about the {detected_label} and the scene like colors, subjects, and composition, but allow "
        "for drift and reinterpretation. The prompt should feel like a memory "
        "of the image, not a perfect description.\n\n"
        "Return ONLY the prompt text, nothing else."
    )

    temperature = min(0.8 + (iteration * 0.02) + np.random.random(), 1.1)

    response = client.responses.create(
        model="gpt-4.1",
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Iteration {iteration}. "
                            f"Object to replace: {detected_label}. "
                            "Write a prompt for what should replace the masked region."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_to_base64(image)}",
                    },
                ],
            }
        ],
        max_output_tokens=150,
        temperature=temperature,
    )

    return response.output_text.strip()


# ── Inpainting ─────────────────────────────────────────────────────────────────

def custom_inpaint_variants(pipe, image, mask, prompt,
                             n_variants=2,
                             num_inference_steps=50,
                             guidance_min=5.0,
                             guidance_max=10.0,
                             mask_scale=0.9,
                             img_filter_sigma=0.2,
                             enable_repaint=True,
                             device="cuda"):
    image_512 = image.resize((512, 512), Image.LANCZOS)
    mask_512  = mask.resize((512, 512), Image.NEAREST)

    img_np  = np.array(image_512).astype(np.float32) / 255.0
    mask_np = np.array(mask_512).astype(np.float32) / 255.0

    image_tensor = (torch.from_numpy(img_np)
                    .permute(2, 0, 1).unsqueeze(0)
                    .to(device, dtype=torch.float16)) * 2.0 - 1.0

    mask_tensor = (torch.from_numpy(mask_np)
                   .unsqueeze(0).unsqueeze(0)
                   .to(device, dtype=torch.float16))
    mask_binary = (mask_tensor > 0.5).to(dtype=torch.float16)

    with torch.no_grad():
        tok_c    = pipe.tokenizer(prompt, padding="max_length",
                                  max_length=pipe.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        cond_emb = pipe.text_encoder(tok_c.input_ids.to(device))[0]

        tok_u      = pipe.tokenizer([""], padding="max_length",
                                    max_length=pipe.tokenizer.model_max_length,
                                    return_tensors="pt")
        uncond_emb = pipe.text_encoder(tok_u.input_ids.to(device))[0]
        text_emb   = torch.cat([uncond_emb, cond_emb])

        image_latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        image_latents = image_latents * pipe.vae.config.scaling_factor

    mask_latent = torch.nn.functional.interpolate(mask_binary, size=(64, 64), mode="nearest")

    masked_image_tensor = image_tensor * (1.0 - mask_binary)
    with torch.no_grad():
        masked_img_latents = pipe.vae.encode(masked_image_tensor).latent_dist.sample()
        masked_img_latents = masked_img_latents * pipe.vae.config.scaling_factor

    if img_filter_sigma > 0:
        masked_img_latents = TVF.gaussian_blur(
            masked_img_latents, kernel_size=[5, 5], sigma=img_filter_sigma
        )

    scaled_mask = mask_latent * mask_scale
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps

    variants = []
    for idx in range(n_variants):
        generator = torch.Generator(device).manual_seed(idx * 137)
        latents = (torch.randn((1, 4, 64, 64), generator=generator,
                               device=device, dtype=torch.float16)
                   * pipe.scheduler.init_noise_sigma)

        for step_idx, t in enumerate(timesteps):
            progress = step_idx / len(timesteps)
            guidance = (guidance_min
                        + 0.5 * (guidance_max - guidance_min)
                        * (1.0 + math.cos(math.pi * progress)))

            if enable_repaint:
                rp_noise    = torch.randn_like(image_latents)
                t_idx       = torch.tensor([t.item()], dtype=torch.long, device=device)
                noised_orig = pipe.scheduler.add_noise(image_latents, rp_noise, t_idx)
                latents     = latents * mask_latent + noised_orig * (1.0 - mask_latent)

            inp = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
            inp = torch.cat([inp,
                             scaled_mask.repeat(2, 1, 1, 1),
                             masked_img_latents.repeat(2, 1, 1, 1)], dim=1)

            with torch.no_grad():
                noise_pred = pipe.unet(inp, t, encoder_hidden_states=text_emb).sample

            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
            latents    = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

        decoded = (decoded / 2.0 + 0.5).clamp(0, 1)
        decoded = (decoded.squeeze(0).permute(1, 2, 0)
                   .cpu().float().numpy() * 255).astype(np.uint8)
        variants.append(Image.fromarray(decoded))
        print(f"    variant {idx + 1}/{n_variants} done.")

    return variants


# ── GIF assembly ───────────────────────────────────────────────────────────────

def assemble_gif(frames, output_path, frame_duration=300):
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=frame_duration,
    )


# ── Top-level pipeline call ────────────────────────────────────────────────────

def run_pipeline(image_path, output_dir, pipe, yolo_model, openai_client,
                 sort_by="confidence", n_variants=2, num_inference_steps=50,
                 progress_callback=None):
    """
    Full pipeline: image → YOLO detection → iterative inpainting → GIF.

    progress_callback(message: str) is called at each major step so the
    Flask server can push status updates to the browser.

    Returns (detections_info, gif_path) where detections_info is a list of
    dicts safe to serialise as JSON {label, confidence, area}.
    """
    os.makedirs(output_dir, exist_ok=True)

    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    image = Image.open(image_path).convert("RGB")
    log("Image loaded. Running object detection...")

    detections = generate_object_masks(image, yolo_model, sort_by=sort_by)

    if not detections:
        log("No objects detected — try rearranging the collage.")
        return [], None

    labels = [d["label"] for d in detections]
    log(f"Detected: {labels}. Starting inpainting...")

    all_frames  = [image]
    current_img = image.copy()

    for i, d in enumerate(detections):
        log(f"[{i+1}/{len(detections)}] Replacing '{d['label']}' "
            f"(conf={d['confidence']:.2f})...")

        prompt = generate_inpaint_prompt(
            current_img, openai_client, iteration=i, detected_label=d["label"]
        )
        log(f"  Prompt: {prompt} {d['label']}")

        variants = custom_inpaint_variants(
            pipe, current_img, d["mask"], prompt,
            n_variants=n_variants,
            num_inference_steps=num_inference_steps,
        )

        current_img = variants[0]
        all_frames.extend(variants)

        for j, frame in enumerate(variants):
            frame.save(os.path.join(output_dir, f"iter{i:02d}_{d['label']}_v{j:02d}.png"))

    # Resize all frames to a consistent size before assembling
    gif_size = (768, 768)
    all_frames = [f.resize(gif_size, Image.LANCZOS) for f in all_frames]

    gif_path = os.path.join(output_dir, "output.gif")
    assemble_gif(all_frames, gif_path)
    log("Done.")

    detections_info = [
        {"label": d["label"], "confidence": round(d["confidence"], 2), "area": d["area"]}
        for d in detections
    ]
    return detections_info, gif_path