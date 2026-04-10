import os
import sys
import shutil
import threading
from datetime import datetime

import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from server import run_server, update_state
from pipeline import load_models, run_pipeline_burst

# ── Config ─────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0              # change if the wrong camera opens
OUTPUT_DIR     = "static/output"
INPUT_DIR      = "static/input"

# ── Backend selection ──────────────────────────────────────────────────────────
SD_BACKEND = "sdxl"             # "sdxl" | "sd15"

# ── Burst capture ──────────────────────────────────────────────────────────────
BURST_MAX = 10                  # max shots per burst (SPACE × N, then ENTER to process)

# ── SAM parameters ─────────────────────────────────────────────────────────────
SAM_MIN_AREA  = 500             # ignore regions smaller than this (px²)

# ── Shared SD parameters ───────────────────────────────────────────────────────
SD_STEPS = 50                   # denoising steps

# ── SDXL parameters (used when SD_BACKEND = "sdxl") ──────────────────────────
SDXL_GUIDANCE  = 13.0           # prompt adherence — higher = more literal (7-16)
SDXL_STRENGTH  = 0.9            # 1.0 = complete disregard for original region
SDXL_GREY_FILL = True           # replace masked region with grey before inpainting
SDXL_SEED      = 42           # set to an integer for reproducible results, None = random

# ── SD 1.5 parameters (used when SD_BACKEND = "sd15") ─────────────────────────
SD15_GUIDANCE  = 12.0           # prompt adherence for SD 1.5 (7-15)

# ── Conditional prompts ────────────────────────────────────────────────────────
CONDITIONAL_PROMPTS = [

    (
        ["clover", "paper clover", "green clover", "green paper clover",
         "paper flower", "green flower", "flower token", "green paper",
         "flower piece", "paper bloom", "folded flower", "flower decoration",
         "four leaf", "four-leaf", "cloverleaf"],
        "A four-leaf clover bursting into vivid botanical life — dense overlapping petals "
        "unfurling in deep emerald and jade, threaded with luminous white veins, "
        "dewy and hyper-real, as if painted by a Renaissance botanist who had seen the future. "
        "Lush, intricate, gloriously alive.",
    ),

    (
        ["bird", "paper bird", "bird silhouette", "bird cutout"],
        "A vivid tropical bird in blazing sunshine yellow — sleek and glossy like lacquered enamel, "
        "wings tucked, side view of bird, every feather crisp and jewel-bright, beak a sharp stroke of tangerine, "
        "eye a single gleaming black bead. Luminous, electric, impossibly saturated.",
    ),

    (
        ["horse", "paper horse", "horse silhouette", "horse cutout"],
        "A small toy horse, chunky and joyful — painted in bold primary colours with a glossy finish, "
        "thick blocky legs, a bright flowing mane in candy pink or electric blue, "
        "surface smooth as a fairground carousel, cheerful and slightly surreal. "
        "Charming, vivid, and unmistakably a toy. side view of the horse.",
    ),

]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

# ── Load models once at startup ────────────────────────────────────────────────
print(f"Loading models (backend={SD_BACKEND})...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pipe, sam_model = load_models(os.getenv("HF_TOKEN"), backend=SD_BACKEND)
print("All models ready.\n")

# ── Pipeline state ─────────────────────────────────────────────────────────────
_pipeline_running = False
_pipeline_lock    = threading.Lock()
_burst_buffer     = []          # list of (image_path, capture_filename)


def _run_pipeline_thread(image_paths, capture_filenames):
    global _pipeline_running

    def progress(msg):
        update_state(message=msg)

    def on_file(f):
        from server import _state, _lock
        with _lock:
            _state["files"].append(f)

    def on_gif(gif_path):
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d%H%M%S%f")
        update_state(gif_url=f"/static/output/output.gif?t={ts}")

    try:
        update_state(files=[])
        detections_info, gif_path = run_pipeline_burst(
            image_paths         = image_paths,
            output_dir          = OUTPUT_DIR,
            pipe                = pipe,
            sam_model           = sam_model,
            openai_client       = client,
            backend             = SD_BACKEND,
            num_inference_steps = SD_STEPS,
            guidance_scale      = SDXL_GUIDANCE if SD_BACKEND == "sdxl" else SD15_GUIDANCE,
            strength            = SDXL_STRENGTH,
            grey_fill           = SDXL_GREY_FILL,
            seed                = SDXL_SEED,
            sam_min_area        = SAM_MIN_AREA,
            conditional_prompts = CONDITIONAL_PROMPTS,
            progress_callback   = progress,
            file_callback       = on_file,
            gif_callback        = on_gif,
        )

        if gif_path:
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            update_state(
                status      = "done",
                message     = f"Done! {len(image_paths)} shot(s) processed. Press SPACE to capture.",
                detections  = detections_info,
                gif_url     = f"/static/output/output.gif?t={ts}",
                capture_url = f"/static/input/{capture_filenames[-1]}",
            )
        else:
            update_state(
                status  = "idle",
                message = "No clovers detected — try again.",
            )

    except Exception as e:
        print(f"Pipeline error: {e}")
        update_state(status="error", message=f"Error: {e}")

    finally:
        with _pipeline_lock:
            _pipeline_running = False


def _trigger_burst_pipeline():
    global _pipeline_running, _burst_buffer

    with _pipeline_lock:
        if _pipeline_running or not _burst_buffer:
            return
        _pipeline_running = True
        burst = list(_burst_buffer)
        _burst_buffer.clear()

    image_paths      = [p for p, _ in burst]
    capture_filenames = [f for _, f in burst]

    update_state(
        status      = "processing",
        message     = f"Processing {len(burst)} shot(s)...",
        gif_url     = None,
        detections  = [],
        files       = [],
        capture_url = f"/static/input/{capture_filenames[-1]}",
    )

    t = threading.Thread(
        target = _run_pipeline_thread,
        args   = (image_paths, capture_filenames),
        daemon = True,
    )
    t.start()


# ── Camera loop (runs on main thread — required by OpenCV on Windows) ──────────

def camera_loop():
    global _burst_buffer

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX at the top of main.py.")
        return

    print("Camera ready.")
    print("Open  http://localhost:5000  in a browser for the display.\n")
    print(f"SPACE = add shot to burst (up to {BURST_MAX})")
    print("ENTER = process burst")
    print("Q     = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        with _pipeline_lock:
            running = _pipeline_running

        burst_count = len(_burst_buffer)

        if running:
            label = "Processing — please wait..."
            color = (0, 165, 255)
        elif burst_count > 0:
            label = f"SPACE = capture ({burst_count}/{BURST_MAX})  |  ENTER = process  |  Q = quit"
            color = (0, 165, 255)   # orange — burst collecting
        else:
            label = f"SPACE = capture  |  Q = quit"
            color = (0, 220, 0)     # green — ready

        cv2.putText(frame, label, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, label, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Installation — Operator View", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            with _pipeline_lock:
                if _pipeline_running:
                    print("Still processing — please wait.")
                    continue

            timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            capture_filename = f"capture_{timestamp}.jpg"
            image_path       = os.path.join(INPUT_DIR, capture_filename)
            cv2.imwrite(image_path, frame)
            _burst_buffer.append((image_path, capture_filename))
            print(f"  Shot {len(_burst_buffer)}/{BURST_MAX}: {image_path}")

            update_state(
                status      = "collecting",
                message     = f"Shot {len(_burst_buffer)}/{BURST_MAX} — SPACE for more, ENTER to process.",
                capture_url = f"/static/input/{capture_filename}",
            )

            if len(_burst_buffer) >= BURST_MAX:
                print(f"\nBurst full ({BURST_MAX} shots). Processing...")
                _trigger_burst_pipeline()

        if key == 13:   # ENTER
            if not running and _burst_buffer:
                print(f"\nProcessing {len(_burst_buffer)} shot(s)...")
                _trigger_burst_pipeline()

    cap.release()
    cv2.destroyAllWindows()


# ── Test mode (no camera) ──────────────────────────────────────────────────────

def test_mode(image_path):
    """Run the pipeline on a static image (treated as a burst of one)."""
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    capture_filename = "test_" + os.path.basename(image_path)
    dest = os.path.join(INPUT_DIR, capture_filename)
    shutil.copy(image_path, dest)

    update_state(
        status      = "processing",
        message     = "Test image loaded. Running pipeline...",
        capture_url = f"/static/input/{capture_filename}",
        gif_url     = None,
        detections  = [],
    )

    print(f"Test mode: running pipeline on {image_path}")
    print("Open  http://localhost:5000  to watch progress.\n")
    _run_pipeline_thread([dest], [capture_filename])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_server, daemon=True)
    flask_thread.start()

    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        test_image = sys.argv[idx + 1]
        test_mode(test_image)
        print("\nPipeline done. Browse to http://localhost:5000 to see the result.")
        print("Press Ctrl+C to quit.")
        while True:
            threading.Event().wait(1)
    else:
        camera_loop()
