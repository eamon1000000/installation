# S.M.I.L.E.
### Stochastically Mutative Image Limbo Engine

A live generative inpainting installation. Participants hold paper cutouts of clovers, birds, or horses before a camera. The system detects and segments the shapes using SAM2, classifies them via GPT-4.1 vision, and inpaints each region using Stable Diffusion XL. Results are compiled into an animated GIF displayed on a locally served web interface.

---

## How it works

1. Operator opens a camera window and presses **SPACE** to capture frames (up to 10)
2. **ENTER** triggers the pipeline on the captured burst
3. SAM2 segments all distinct regions in each frame
4. GPT-4.1 views a numbered overlay and classifies each region as clover, bird, horse, or background
5. Each detected shape is inpainted using a handcrafted prompt matched to its class
6. A ping-pong GIF is assembled and pushed live to the browser display at `http://localhost:5000`

---

## Requirements

### Hardware
- NVIDIA GPU with at least 8 GB VRAM (tested on RTX A4000 16 GB)
- CUDA 12.4+ recommended

### API Keys
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_key_here
HF_TOKEN=your_huggingface_token_here
```

---

## Installation

**1. Install PyTorch first** (before anything else):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**2. Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

**3. Download the SAM2 model checkpoint:**

Place `sam2_b.pt` in the project root. It will be downloaded automatically by `ultralytics` on first run if not present.

---

## Running the installation

### Camera mode (default)
```bash
python main.py
```
- Opens a camera preview window
- **SPACE** — capture a frame (accumulates up to 10)
- **ENTER** — process the captured burst
- **Q** — quit
- Open `http://localhost:5000` in a browser for the display

### Test mode (no camera)
Run the pipeline on a static image:
```bash
python main.py --image path/to/your/image.jpg
```
Then open `http://localhost:5000` to see the result.

---

## Configuration

All tunable parameters are at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Camera device index — change if the wrong camera opens |
| `SD_BACKEND` | `"sdxl"` | Inpainting model: `"sdxl"`  for stable diffusion XL or `"sd15"` for stable diffusion 1.5|
| `BURST_MAX` | `10` | Max frames per burst |
| `SAM_MIN_AREA` | `500` | Minimum segment area in px² — filters out noise |
| `SD_STEPS` | `50` | Denoising steps |
| `SDXL_GUIDANCE` | `13.0` | Prompt adherence (7–16) |
| `SDXL_STRENGTH` | `0.9` | How aggressively the model overwrites the masked region |
| `SDXL_GREY_FILL` | `True` | Replace masked region with mid-grey before inpainting |

### Adding or changing shape classes

Shape classes and their inpainting prompts are defined in `CONDITIONAL_PROMPTS` in `main.py`. Each entry is a tuple of `(keywords_list, prompt_string)`. The keywords are matched against GPT-4.1's classification output.

The GPT-4.1 classification instructions (what shapes it looks for) are defined in `_GPT_INSTRUCTIONS` in `pipeline.py`.

---

## File structure

```
installation/
├── main.py                  # Entry point — camera loop, burst capture, pipeline orchestration
├── pipeline.py              # All ML logic: SAM2, GPT-4.1, SDXL inpainting, GIF assembly
├── server.py                # Flask server — serves display page and /status endpoint
├── requirements.txt         # Python dependencies
├── colab_pipeline.ipynb     # Standalone Colab notebook (see below)
├── .env                     # API keys (not committed)
├── sam2_b.pt                # SAM2 model checkpoint
├── static/
│   ├── input/               # Captured frames saved here
│   └── output/              # output.gif and inpainted frames saved here
└── templates/
    └── index.html           # Browser display page
```

---

## Testing on your own machine (Colab notebook)

If you don't have a local NVIDIA GPU, `colab_pipeline.ipynb` lets you run the same SAM2 + SDXL pipeline in Google Colab on a free T4 GPU — no camera or Flask server needed.

**Setup:**
1. Upload a folder of images to your Google Drive
2. Open the notebook in Colab and run the setup cells to install dependencies and mount Drive
3. Set the `IMAGE_DIR` variable to point to your Drive folder
4. Run the pipeline cell — it processes all images in the folder and saves a GIF

The Colab notebook uses the same `pipeline.py` logic and the same conditional prompts, so output should be consistent with the installation version.

---

## Pipeline architecture

The pipeline is optimised around two bottlenecks — GPU compute and API latency:

- **SAM2 segmentation** — sequential on GPU (one frame at a time)
- **GPT-4.1 classification + prompt matching** — parallel across all frames via `ThreadPoolExecutor` (API-bound, not GPU-bound)
- **SDXL inpainting** — sequential on GPU (composites each result back onto the current frame before the next)
- **GIF assembly** — runs after every completed frame, pushes a live update to the browser

---

## Models used

| Model | Source | Purpose |
|---|---|---|
| `stable-diffusion-xl-1.0-inpainting-0.1` | HuggingFace / diffusers | Inpainting at 1024×1024 |
| `runwayml/stable-diffusion-inpainting` | HuggingFace / diffusers | Fallback SD 1.5 backend at 512×512 |
| `sam2_b.pt` | Ultralytics / Meta | Class-agnostic instance segmentation |
| `gpt-4.1` | OpenAI Responses API | Shape classification + prompt generation |
