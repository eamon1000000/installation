import os
import io
import json
import math
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import cv2
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline
from ultralytics import SAM
from openai import OpenAI


def load_models(hf_token, backend="sdxl", device="cuda"):
    if backend == "sdxl":
        print("Loading SDXL inpainting pipeline...")
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        print(f"  SDXL pipeline loaded on {device}.")
    else:
        print("Loading SD 1.5 inpainting pipeline...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            token=hf_token,
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        print(f"  SD 1.5 pipeline loaded on {device}.")

    print("Loading SAM2 model...")
    sam = SAM("sam2_b.pt")
    print("  SAM2 loaded.")

    return pipe, sam


# ── SAM detection ─────────────────────────────────────────────────────────────

def _draw_numbered_segments(image, candidates):
    """Overlay numbered outlines on the image for all candidates."""
    vis = np.array(image.convert("RGB")).copy()
    colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80),
        (255, 80, 255), (80, 255, 255), (255, 160, 40), (160, 40, 255),
        (40, 200, 255), (255, 40, 160), (160, 255, 40), (40, 255, 160),
        (220, 120, 40), (40, 120, 220), (220, 40, 120), (120, 220, 40),
        (40, 220, 120), (120, 40, 220), (200, 200, 40), (40, 200, 200),
    ]
    for i, (_, mask_arr) in enumerate(candidates):
        color = colors[i % len(colors)]
        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 3)
        ys, xs = np.where(mask_arr > 0)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(vis, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6)
        cv2.putText(vis, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
    return Image.fromarray(vis)


_GPT_INSTRUCTIONS = (
    "You are analysing a segmented image for an AI art installation. "
    "The image shows numbered outlined regions detected by a segmentation model.\n\n"
    "Your task: identify ONLY green paper clover shapes.\n"
    "Look for four-leaf clover or flower shapes cut or folded from green paper. "
    "They will appear as distinct green shapes — look for clover silhouettes, "
    "petal shapes, or flower-like outlines in green. Include every one you can find.\n\n"
    "Ignore people, furniture, walls, floors, screens, cables, and all other objects entirely.\n\n"
    "Return a JSON array containing only the regions you identified as clovers. "
    "Each entry must have: 'id' (integer segment number), "
    "'label' (use 'green paper clover'). "
    "If you find none, return an empty array []. "
    "Return ONLY the JSON array, no other text."
)


def _sam_candidates(image, sam_model, min_area=2000, max_segments=10,
                    intermediate_dir=None, frame_idx=0):
    """Run SAM only. Returns (vis_image, pool) — no GPT call."""
    img_w, img_h = image.size
    results = sam_model(image, verbose=False)

    if not results or results[0].masks is None:
        print(f"SAM frame {frame_idx}: no masks.")
        return None, []

    raw_masks = results[0].masks.data.cpu().numpy()
    candidates = []
    for mask_arr in raw_masks:
        mask_resized = cv2.resize(
            mask_arr.astype(np.uint8) * 255, (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )
        area = int((mask_resized > 0).sum())
        if area >= min_area:
            candidates.append((area, mask_resized))

    candidates.sort(key=lambda x: x[0], reverse=True)
    pool = candidates[:max(max_segments * 3, 30)]

    vis_image = _draw_numbered_segments(image, pool)
    if intermediate_dir:
        vis_image.save(os.path.join(intermediate_dir, f"sam_f{frame_idx:02d}.png"))

    return vis_image, pool


def _gpt_select(vis_image, pool, openai_client, feather_radius=3):
    """Run GPT selection on a pre-computed vis_image/pool. Returns detections."""
    if not pool:
        return []

    response = openai_client.responses.create(
        model="gpt-4.1",
        instructions=_GPT_INSTRUCTIONS,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": "Find all green paper clovers in this segmented image."},
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{image_to_base64(vis_image)}"},
            ],
        }],
        max_output_tokens=600,
        temperature=0.2,
    )

    try:
        selections = json.loads(response.output_text.strip())
    except json.JSONDecodeError:
        print("  GPT-4.1 JSON parse failed.")
        selections = []

    detections = []
    for sel in selections:
        idx = sel.get("id")
        if not isinstance(idx, int) or idx >= len(pool):
            continue
        area, mask_arr = pool[idx]
        mask_pil = Image.fromarray(mask_arr).convert("L")
        if feather_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        detections.append({
            "mask":       mask_pil,
            "label":      sel.get("label", "green paper clover"),
            "confidence": 1.0,
            "area":       area,
        })

    return detections


def generate_sam_masks(image, sam_model, openai_client,
                       max_segments=10, min_area=2000, feather_radius=3,
                       intermediate_dir=None):
    """Convenience wrapper: SAM + GPT in one call (used by single-image mode)."""
    vis_image, pool = _sam_candidates(image, sam_model, min_area, max_segments,
                                      intermediate_dir, frame_idx=0)
    if not pool:
        return []
    print(f"SAM: {len(pool)} candidates — asking GPT-4.1...")
    detections = _gpt_select(vis_image, pool, openai_client, feather_radius)
    print(f"  Found: {[d['label'] for d in detections]}")
    return detections

# ── Prompt generation ──────────────────────────────────────────────────────────

def image_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


PERSON_KEYWORDS = [
    "person", "human", "man", "woman", "people", "figure", "standing",
    "seated person", "sitting person", "individual", "someone", "body",
    "portrait", "face", "visitor", "student", "child", "adult", "girl", "boy",
]


def _is_person(label):
    label_lower = label.lower()
    return any(kw in label_lower for kw in PERSON_KEYWORDS)


def generate_person_inpaint_prompt(image, mask, client):
    """
    Crop the masked person region, send it to GPT, and ask it to describe
    their specific features before generating an uncannifying inpaint prompt.
    """
    img_array = np.array(image.convert("RGB"))
    mask_array = np.array(mask.resize(image.size, Image.NEAREST))
    ys, xs = np.where(mask_array > 127)
    if len(ys) == 0:
        cropped = image
    else:
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        cropped = Image.fromarray(img_array[y1:y2, x1:x2])

    response = client.responses.create(
        model="gpt-4.1",
        instructions=(
            "You are a prompt generator for an AI inpainting model in a surreal art installation. "
            "You will be shown a cropped image of a person. "
            "Carefully observe their specific features: hair colour and texture, face shape, "
            "skin tone, eye colour, clothing colours and style, build, expression, posture. "
            "Write a single 2-sentence inpainting prompt that takes this specific person and "
            "transforms them into a hyper-real, uncanny, golden android version of themselves — "
            "their recognisable features are preserved but pushed to an extreme: "
            "skin becomes luminous and glassy like polished amber resin, "
            "their specific hair colour is retained but lacquered into perfect sculptural geometry, "
            "eyes are wide, chromatic and slightly too large, "
            "cheekbones and jawline become architectural and idealised. "
            "Clothing becomes glossy and saturated — amplify whatever they are wearing. "
            "The overall effect: Ryan Trecartin video strangeness, high-fashion editorial, "
            "sci-fi glamour, beautiful and deeply unsettling. "
            "Return ONLY the prompt text, nothing else."
        ),
        input=[{
            "role": "user",
            "content": [{
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_to_base64(cropped)}",
            }],
        }],
        max_output_tokens=200,
        temperature=0.9,
    )
    return response.output_text.strip()


def _match_conditional_prompt(label, conditional_prompts):
    """Return a fixed prompt if the label matches any keyword, otherwise None."""
    label_lower = label.lower()
    for keywords, prompt in conditional_prompts:
        if any(kw.lower() in label_lower for kw in keywords):
            return prompt
    return None


def generate_inpaint_prompt(image, client, iteration=0, detected_label=None):
    instructions = (
        "You are a prompt generator for an AI inpainting model in a surreal art installation. "
        "The installation exists in a single aesthetic world: mid-century modern meets science fiction — "
        "think 1962 World's Fair, Stanley Kubrick, Eames furniture, NASA optimism, Ryan Trecartin strangeness. "
        "Everything is uncanny, nostalgic, futuristic, and delightful all at once.\n\n"
        "Your job is to transform the detected object into this world. Rules:\n"
        "- Pull the object into the mid-century modern sci-fi universe: warm wood tones, brushed aluminium, "
        "bakelite, amber cathode glow, hairpin legs, moulded fibreglass, smoked glass, brass details.\n"
        "- Make it strange and alive — glowing, hovering, too perfect, slightly wrong in a beautiful way.\n"
        "- Keep a thread back to the original object's shape or material so the transformation feels grounded.\n"
        "- Be specific and visual: describe textures, light sources, colours, and surface quality.\n"
        "- Nostalgic but uncanny. Optimistic but eerie. Beautiful and deeply weird.\n"
        "- 1-2 sentences maximum. Return ONLY the prompt text, nothing else."
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

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, ugly, flat, dull, washed out, "
    "overexposed, noisy, pixelated, out of focus, boring, mundane, "
    "modern, contemporary, digital, plastic, generic, stock photo, "
    "photorealistic, hyperrealistic, 3d render, cgi, white, blank, empty"
)


def inpaint_sdxl(pipe, image, mask, prompt,
                 n_variants=1,
                 num_inference_steps=50,
                 guidance_scale=13.0,
                 strength=1.0,
                 grey_fill=True,
                 device="cuda"):
    """
    SDXL inpainting. grey_fill=True replaces the masked region with mid-grey
    before passing to the pipeline, breaking white-on-white context anchoring.
    """
    mask_l = mask.convert("L")

    if grey_fill:
        # Fill masked region with mid-grey so the model isn't anchored to white
        grey = Image.new("RGB", image.size, (128, 128, 128))
        image_in = Image.composite(grey, image, mask_l)
    else:
        image_in = image

    image_1024 = image_in.resize((1024, 1024), Image.LANCZOS)
    mask_1024  = mask_l.resize((1024, 1024), Image.NEAREST)

    results = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image_1024,
        mask_image=mask_1024,
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        num_images_per_prompt=n_variants,
    ).images

    for i in range(len(results)):
        print(f"    variant {i + 1}/{n_variants} done.")

    return results


def inpaint_sd15(pipe, image, mask, prompt,
                 n_variants=1,
                 num_inference_steps=50,
                 guidance_scale=12.0,
                 device="cuda"):
    """
    SD 1.5 inpainting via the standard pipeline call.
    """
    image_512 = image.resize((512, 512), Image.LANCZOS)
    mask_512  = mask.resize((512, 512), Image.NEAREST)

    results = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image_512,
        mask_image=mask_512,
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=n_variants,
    ).images

    for i in range(len(results)):
        print(f"    variant {i + 1}/{n_variants} done.")

    return results


# ── GIF assembly ───────────────────────────────────────────────────────────────

def _save_gif_now(frames, orig_size, output_path):
    """Resize, ping-pong, and save whatever frames exist so far."""
    max_dim = 768
    w, h    = orig_size
    scale   = min(max_dim / w, max_dim / h)
    gif_size = (int(w * scale), int(h * scale))
    sized    = [f.resize(gif_size, Image.LANCZOS) for f in frames]
    pingpong = sized + sized[-2:0:-1]
    assemble_gif(pingpong, output_path)


def assemble_gif(frames, output_path, frame_duration=150):
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=frame_duration,
    )


# ── Burst pipeline ────────────────────────────────────────────────────────────

def run_pipeline_burst(image_paths, output_dir, pipe, sam_model, openai_client,
                       backend="sdxl",
                       num_inference_steps=50,
                       guidance_scale=13.0, strength=1.0, grey_fill=True,
                       sam_min_area=500,
                       conditional_prompts=None,
                       progress_callback=None,
                       file_callback=None,
                       gif_callback=None):
    """
    Burst pipeline: N images → parallel SAM+GPT detection → parallel prompt
    generation → sequential inpainting → GIF of (orig, inpainted) pairs.
    """
    os.makedirs(output_dir, exist_ok=True)
    intermediate_dir = os.path.join(os.path.dirname(output_dir), "intermediate")
    if os.path.exists(intermediate_dir):
        import shutil
        shutil.rmtree(intermediate_dir)
    os.makedirs(intermediate_dir)

    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    images = [Image.open(p).convert("RGB") for p in image_paths]
    orig_size = images[0].size
    log(f"Loaded {len(images)} image(s). Running SAM on all frames...")

    # ── Phase 1: SAM (sequential — GPU-bound) ─────────────────────────────────
    sam_results = []   # [(frame_idx, image, vis_image, pool)]
    for i, img in enumerate(images):
        log(f"  SAM frame {i + 1}/{len(images)}...")
        vis_image, pool = _sam_candidates(img, sam_model,
                                          min_area=sam_min_area,
                                          intermediate_dir=intermediate_dir,
                                          frame_idx=i)
        if pool:
            sam_results.append((i, img, vis_image, pool))
        else:
            log(f"    Frame {i + 1}: no candidates.")

    if not sam_results:
        log("No regions found in any frame.")
        return [], None

    # ── Phase 2: GPT selection (parallel — API-bound) ─────────────────────────
    log("Running GPT selection on all frames in parallel...")
    frame_detections = {}   # frame_idx → (image, detections)

    def _select(args):
        fi, img, vis_image, pool = args
        dets = _gpt_select(vis_image, pool, openai_client, feather_radius=3)
        return fi, img, dets

    with ThreadPoolExecutor(max_workers=min(len(sam_results), 8)) as ex:
        futures = {ex.submit(_select, r): r[0] for r in sam_results}
        for future in as_completed(futures):
            fi, img, dets = future.result()
            if dets:
                frame_detections[fi] = (img, dets)
                log(f"  Frame {fi + 1}: {len(dets)} clover(s).")
            else:
                log(f"  Frame {fi + 1}: no clovers — skipping.")

    if not frame_detections:
        log("No clovers detected in any frame.")
        return [], None

    # ── Phase 3: Prompt generation (parallel across all detections) ────────────
    log("Generating prompts...")
    all_tasks = [
        (fi, di, d)
        for fi, (img, dets) in frame_detections.items()
        for di, d in enumerate(dets)
    ]
    prompt_map = {}   # (fi, di) → (prompt, type)

    def _build_prompt(args):
        fi, di, d = args
        conditional = _match_conditional_prompt(d["label"], conditional_prompts or [])
        if conditional:
            return (fi, di), conditional, "conditional"
        img = frame_detections[fi][0]
        p = generate_inpaint_prompt(img, openai_client, iteration=di,
                                    detected_label=d["label"])
        return (fi, di), p, "gpt"

    with ThreadPoolExecutor(max_workers=min(len(all_tasks), 8)) as ex:
        futures = {ex.submit(_build_prompt, t): t for t in all_tasks}
        for future in as_completed(futures):
            key, p, pt = future.result()
            prompt_map[key] = (p, pt)
            fi, di = key
            log(f"  Frame {fi + 1} clover {di + 1} ({pt}): {p[:70]}...")
            if file_callback:
                label = frame_detections[fi][1][di]["label"]
                file_callback({"label": label, "prompt": p, "type": pt})

    # ── Phase 4: Inpainting + GIF assembly (sequential — GPU-bound) ───────────
    log("Inpainting...")
    gif_path   = os.path.join(output_dir, "output.gif")
    gif_frames = []

    for fi in sorted(frame_detections.keys()):
        orig_img, dets = frame_detections[fi]
        current = orig_img.copy()

        for di, d in enumerate(dets):
            prompt, _ = prompt_map.get((fi, di), ("", "gpt"))
            log(f"  Frame {fi + 1}, clover {di + 1}/{len(dets)}...")

            if backend == "sdxl":
                result = inpaint_sdxl(pipe, current, d["mask"], prompt,
                                      n_variants=1,
                                      num_inference_steps=num_inference_steps,
                                      guidance_scale=guidance_scale,
                                      strength=strength,
                                      grey_fill=grey_fill)
            else:
                result = inpaint_sd15(pipe, current, d["mask"], prompt,
                                      n_variants=1,
                                      num_inference_steps=num_inference_steps,
                                      guidance_scale=guidance_scale)

            frame_result = result[0].resize(orig_size, Image.LANCZOS)
            current = Image.composite(frame_result, current, d["mask"].convert("L"))

        # First frame: original + inpainted. All subsequent: inpainted only.
        if not gif_frames:
            gif_frames.append(orig_img.resize(orig_size, Image.LANCZOS))
        gif_frames.append(current)

        _save_gif_now(gif_frames, orig_size, gif_path)
        if gif_callback:
            gif_callback(gif_path)
        log(f"  Frame {fi + 1} done ({len(gif_frames)} GIF frames so far).")

    log("Done.")

    detections_info = [
        {"label": d["label"], "confidence": round(d["confidence"], 2), "area": d["area"]}
        for _, dets in (frame_detections[fi] for fi in sorted(frame_detections))
        for d in dets
    ]
    return detections_info, gif_path


# ── Single-image pipeline (kept for reference / debug) ────────────────────────

def run_pipeline(image_path, output_dir, pipe, sam_model, openai_client,
                 backend="sdxl",
                 n_variants=1, num_inference_steps=50,
                 guidance_scale=13.0, strength=1.0, grey_fill=True,
                 sam_max_segments=10, sam_min_area=500,
                 conditional_prompts=None,
                 progress_callback=None,
                 file_callback=None,
                 gif_callback=None):
    """
    Full pipeline: image → SAM2 detection → iterative inpainting → GIF.

    Returns (detections_info, gif_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    intermediate_dir = os.path.join(os.path.dirname(output_dir), "intermediate")
    if os.path.exists(intermediate_dir):
        import shutil
        shutil.rmtree(intermediate_dir)
    os.makedirs(intermediate_dir)

    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    image    = Image.open(image_path).convert("RGB")
    orig_size = image.size   # (w, h) — preserved throughout for aspect ratio
    log("Image loaded. Running SAM2 segmentation...")

    detections = generate_sam_masks(image, sam_model, openai_client,
                                    max_segments=sam_max_segments,
                                    min_area=sam_min_area,
                                    intermediate_dir=intermediate_dir)

    if not detections:
        log("No objects detected — try rearranging the collage.")
        return [], None

    labels = [d["label"] for d in detections]
    log(f"Detected: {labels}. Generating prompts in parallel...")

    # ── Generate all prompts concurrently before inpainting begins ────────────
    def _build_prompt(args):
        i, d = args
        if _is_person(d["label"]):
            p = generate_person_inpaint_prompt(image, d["mask"], openai_client)
            return i, p, "person"
        conditional = _match_conditional_prompt(d["label"], conditional_prompts or [])
        if conditional:
            return i, conditional, "conditional"
        p = generate_inpaint_prompt(image, openai_client, iteration=i, detected_label=d["label"])
        return i, p, "gpt"

    prompts     = [None] * len(detections)
    prompt_types = [None] * len(detections)
    with ThreadPoolExecutor(max_workers=min(len(detections), 8)) as ex:
        futures = {ex.submit(_build_prompt, (i, d)): i for i, d in enumerate(detections)}
        for future in as_completed(futures):
            i, p, pt = future.result()
            prompts[i]      = p
            prompt_types[i] = pt
            log(f"  [{i+1}] prompt ready ({pt}): {p[:80]}...")

    log("All prompts ready. Starting inpainting...")

    # Save masks/previews once upfront before inpainting begins
    for i, d in enumerate(detections):
        slug = d["label"].replace(" ", "_")[:20]
        d["mask"].save(os.path.join(intermediate_dir, f"mask_{i:02d}_{slug}.png"))

    gif_path    = os.path.join(output_dir, "output.gif")
    all_frames  = [image]
    current_img = image.copy()

    # Outer loop: rounds. Inner loop: segments.
    # Each round does one inpainting pass across all segments,
    # so the GIF builds as: seg1r1, seg2r1, seg3r1, seg1r2, seg2r2 ...
    for round_idx in range(n_variants):
        log(f"Round {round_idx + 1}/{n_variants}...")
        for i, d in enumerate(detections):
            log(f"  [{i+1}/{len(detections)}] '{d['label']}'...")

            prompt      = prompts[i]
            prompt_type = prompt_types[i]
            if round_idx == 0 and file_callback:
                file_callback({"label": d["label"], "prompt": prompt, "type": prompt_type})

            if backend == "sdxl":
                result = inpaint_sdxl(
                    pipe, current_img, d["mask"], prompt,
                    n_variants=1,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    grey_fill=grey_fill,
                )
            else:
                result = inpaint_sd15(
                    pipe, current_img, d["mask"], prompt,
                    n_variants=1,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

            frame_orig  = result[0].resize(orig_size, Image.LANCZOS)
            current_img = Image.composite(frame_orig, current_img, d["mask"].convert("L"))
            all_frames.append(frame_orig)

            slug = d["label"].replace(" ", "_")[:20]
            frame_orig.save(os.path.join(output_dir, f"r{round_idx:02d}_{i:02d}_{slug}.png"))

            # Push updated GIF to browser after every single frame
            _save_gif_now(all_frames, orig_size, gif_path)
            if gif_callback:
                gif_callback(gif_path)
            log(f"    GIF updated ({len(all_frames)} frames).")

    log("Done.")

    detections_info = [
        {"label": d["label"], "confidence": round(d["confidence"], 2), "area": d["area"]}
        for d in detections
    ]
    return detections_info, gif_path