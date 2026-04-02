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
from pipeline import load_models, run_pipeline

# ── Config ─────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0          # change if the wrong camera opens
OUTPUT_DIR   = "static/output"
INPUT_DIR    = "static/input"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

# ── Load models once at startup ────────────────────────────────────────────────
print("Loading models (this takes ~30s the first time)...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pipe, yolo_model = load_models(os.getenv("HF_TOKEN"))
print("All models ready.\n")

# ── Pipeline state ─────────────────────────────────────────────────────────────
_pipeline_running = False
_pipeline_lock    = threading.Lock()


def _run_pipeline_thread(image_path, capture_filename):
    global _pipeline_running

    def progress(msg):
        update_state(message=msg)

    try:
        detections_info, gif_path = run_pipeline(
            image_path       = image_path,
            output_dir       = OUTPUT_DIR,
            pipe             = pipe,
            yolo_model       = yolo_model,
            openai_client    = client,
            progress_callback= progress,
        )

        if gif_path:
            # Use cache-busting timestamp so the browser reloads the GIF
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            gif_url     = f"/static/output/output.gif?t={ts}"
            capture_url = f"/static/input/{capture_filename}"
            update_state(
                status      = "done",
                message     = "Done! Press SPACE to capture again.",
                detections  = detections_info,
                gif_url     = gif_url,
                capture_url = capture_url,
            )
        else:
            update_state(
                status  = "idle",
                message = "No objects detected — try rearranging the collage.",
            )

    except Exception as e:
        print(f"Pipeline error: {e}")
        update_state(status="error", message=f"Error: {e}")

    finally:
        with _pipeline_lock:
            _pipeline_running = False


# ── Camera loop (runs on main thread — required by OpenCV on Windows) ──────────

def camera_loop():
    global _pipeline_running

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)   # CAP_DSHOW = faster init on Windows
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX at the top of main.py.")
        return

    print("Camera ready.")
    print("Open  http://localhost:5000  in a browser for the display.\n")
    print("SPACE = capture and process")
    print("Q     = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        with _pipeline_lock:
            running = _pipeline_running

        # Overlay status text on the operator preview
        if running:
            label = "Processing — please wait..."
            color = (0, 165, 255)   # orange
        else:
            label = "SPACE = capture  |  Q = quit"
            color = (0, 220, 0)     # green

        cv2.putText(frame, label, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)   # shadow
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
                _pipeline_running = True

            # Save the captured frame to disk
            timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
            capture_filename = f"capture_{timestamp}.jpg"
            image_path      = os.path.join(INPUT_DIR, capture_filename)
            cv2.imwrite(image_path, frame)
            print(f"\nCaptured: {image_path}")

            update_state(
                status      = "processing",
                message     = "Image captured. Running object detection...",
                gif_url     = None,
                detections  = [],
                capture_url = f"/static/input/{capture_filename}",
            )

            t = threading.Thread(
                target  = _run_pipeline_thread,
                args    = (image_path, capture_filename),
                daemon  = True,
            )
            t.start()

    cap.release()
    cv2.destroyAllWindows()


# ── Test mode (no camera) ──────────────────────────────────────────────────────

def test_mode(image_path):
    """Run the pipeline on a static image, no camera needed."""
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    # Copy image into input dir so the browser can show it
    capture_filename = "test_" + os.path.basename(image_path)
    dest = os.path.join(INPUT_DIR, capture_filename)
    shutil.copy(image_path, dest)

    update_state(
        status      = "processing",
        message     = "Test image loaded. Running object detection...",
        capture_url = f"/static/input/{capture_filename}",
        gif_url     = None,
        detections  = [],
    )

    print(f"Test mode: running pipeline on {image_path}")
    print("Open  http://localhost:5000  to watch progress.\n")
    _run_pipeline_thread(dest, capture_filename)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_server, daemon=True)
    flask_thread.start()

    # Usage: python main.py --image path/to/test.jpg
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        test_image = sys.argv[idx + 1]
        test_mode(test_image)
        print("\nPipeline done. Browse to http://localhost:5000 to see the result.")
        print("Press Ctrl+C to quit.")
        while True:
            threading.Event().wait(1)
    else:
        camera_loop()           # normal camera mode