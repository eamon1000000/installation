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
from ultralytics import YOLO, SAM
from transformers import AutoProcessor, AutoModelForCausalLM
from openai import OpenAI


def load_models(hf_token, detection_mode="yolo", device="cuda"):
    print("Loading Stable Diffusion inpainting pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    print(f"  SD pipeline loaded on {device}.")

    models = {}

    if detection_mode == "yolo":
        print("Loading YOLO segmentation model...")
        models["yolo"] = YOLO("yolov8m-seg.pt")
        print("  YOLO loaded.")

    elif detection_mode == "sam":
        print("Loading SAM2 model...")
        models["sam"] = SAM("sam2_b.pt")
        print("  SAM2 loaded.")

    elif detection_mode == "florence":
        print("Loading Florence-2 model...")
        models["florence_model"] = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
        models["florence_processor"] = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True,
        )
        print("  Florence-2 loaded.")
        print("Loading SAM2 model...")
        models["sam"] = SAM("sam2_b.pt")
        print("  SAM2 loaded.")

    return pipe, models


# ── Object detection ───────────────────────────────────────────────────────────

def generate_object_masks(image, model, sort_by="confidence",
                           feather_radius=11, min_confidence=0.1, min_area=200,
                           yolo_conf=0.1, yolo_iou=0.3):
    img_w, img_h = image.size
    results = model(image, verbose=False, conf=yolo_conf, iou=yolo_iou)[0]

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


# ── SAM detection ─────────────────────────────────────────────────────────────

def _describe_region(cropped_image, client):
    """Ask GPT-4o to name whatever is in a cropped mask region (2-4 words)."""
    response = client.responses.create(
        model="gpt-4.1",
        instructions=(
            "You are looking at a cropped region of a larger image. "
            "Identify the single most prominent object or material visible. "
            "Respond with 2-4 words only, no punctuation, no explanation."
        ),
        input=[{
            "role": "user",
            "content": [{
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_to_base64(cropped_image)}",
            }],
        }],
        max_output_tokens=20,
        temperature=0.3,
    )
    return response.output_text.strip().lower()


def generate_sam_masks(image, sam_model, openai_client,
                       max_segments=6, min_area=2000, feather_radius=11):
    """
    Segment everything in the image with SAM2, then ask GPT-4o to label
    each region. Returns the same dict format as generate_object_masks.
    Capped at max_segments largest regions to keep runtime predictable.
    """
    img_w, img_h = image.size
    results = sam_model(image, verbose=False)

    if not results or results[0].masks is None:
        print("SAM: no masks found.")
        return []

    raw_masks = results[0].masks.data.cpu().numpy()  # (N, H, W) binary

    candidates = []
    for mask_arr in raw_masks:
        # Resize mask to original image dimensions
        mask_resized = cv2.resize(
            mask_arr.astype(np.uint8) * 255,
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )
        area = int((mask_resized > 0).sum())
        if area >= min_area:
            candidates.append((area, mask_resized))

    # Keep only the N largest to avoid too many GPT-4o calls
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:max_segments]

    print(f"SAM: {len(candidates)} region(s) after filtering (asking GPT-4o to label each)...")

    detections = []
    img_array = np.array(image.convert("RGB"))

    for i, (area, mask_arr) in enumerate(candidates):
        # Crop the masked region for GPT-4o labelling
        ys, xs = np.where(mask_arr > 0)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        cropped = Image.fromarray(img_array[y1:y2, x1:x2])

        label = _describe_region(cropped, openai_client)
        print(f"  [{i}] GPT-4o says: '{label}'  area={area:,}px")

        mask_pil = Image.fromarray(mask_arr).convert("L")
        if feather_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))

        detections.append({
            "mask":       mask_pil,
            "label":      label,
            "confidence": 1.0,   # SAM doesn't produce confidence scores
            "area":       area,
        })

    return detections


# ── Florence-2 + SAM detection ─────────────────────────────────────────────────

def generate_florence_masks(image, florence_model, florence_processor, sam_model,
                             feather_radius=11, min_area=500, device="cuda"):
    """
    Florence-2 open-vocabulary object detection → bounding boxes + labels,
    then SAM2 converts each bounding box to a pixel-accurate mask.
    """
    img_w, img_h = image.size

    # Florence-2 open-vocabulary detection
    inputs = florence_processor(
        text="<OD>", images=image, return_tensors="pt"
    ).to(device, torch.float16)

    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

    generated_text = florence_processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed = florence_processor.post_process_generation(
        generated_text, task="<OD>", image_size=(img_w, img_h)
    )

    bboxes = parsed["<OD>"]["bboxes"]    # [[x1,y1,x2,y2], ...]
    labels = parsed["<OD>"]["labels"]

    print(f"Florence-2: {len(bboxes)} object(s) detected.")
    for lbl, bb in zip(labels, bboxes):
        print(f"  {lbl:30s}  bbox={[round(v) for v in bb]}")

    if not bboxes:
        return []

    detections = []
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # SAM2 with bounding box prompt
        sam_results = sam_model(image, bboxes=[[x1, y1, x2, y2]], verbose=False)

        if not sam_results or sam_results[0].masks is None:
            continue

        mask_arr = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
        mask_arr = cv2.resize(mask_arr, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        area = int((mask_arr > 0).sum())
        if area < min_area:
            continue

        mask_pil = Image.fromarray(mask_arr).convert("L")
        if feather_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))

        detections.append({
            "mask":       mask_pil,
            "label":      label,
            "confidence": 1.0,
            "area":       area,
        })

    # Largest first
    detections.sort(key=lambda d: d["area"], reverse=True)
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

def run_pipeline(image_path, output_dir, pipe, models, openai_client,
                 detection_mode="yolo", sort_by="confidence",
                 n_variants=2, num_inference_steps=50,
                 progress_callback=None):
    """
    Full pipeline: image → detection → iterative inpainting → GIF.

    detection_mode : 'yolo' | 'sam' | 'florence'
    models         : dict returned by load_models()
    progress_callback(message: str) is called at each major step.

    Returns (detections_info, gif_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    image = Image.open(image_path).convert("RGB")
    log(f"Image loaded. Running object detection [{detection_mode}]...")

    if detection_mode == "yolo":
        detections = generate_object_masks(image, models["yolo"], sort_by=sort_by)
    elif detection_mode == "sam":
        detections = generate_sam_masks(image, models["sam"], openai_client)
    elif detection_mode == "florence":
        detections = generate_florence_masks(
            image,
            models["florence_model"],
            models["florence_processor"],
            models["sam"],
        )
    else:
        raise ValueError(f"Unknown detection_mode '{detection_mode}'. Choose yolo | sam | florence.")

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