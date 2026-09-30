# WasteWise AI — Autonomous Municipal Waste Analytics Agent

> **Current phase:** computer-vision backend only:
> YOLOv8 detection, training/evaluation pipeline, FastAPI inference API, PostgreSQL persistence and tests.
> The frontend, LangGraph agent, LLM, reports and alerts come in later phases.

```
Image → YOLOv8 → detections (class, confidence, bbox)
      → counts & composition → recyclable % / contamination % → JSON API response
```

**Scientific boundary.** The system measures **object counts** and **count-based
composition** from 2D images. It does **not** estimate weight in kilograms.
"Contamination %" always means *object-count contamination*.

Waste classes (fixed order, used everywhere):

| id | class | recycling/dry stream |
|----|-------|----------------------|
| 0 | plastic | recyclable |
| 1 | paper | recyclable |
| 2 | glass | recyclable |
| 3 | metal | recyclable |
| 4 | cardboard | recyclable |
| 5 | organic | contaminant |

---

## Project layout

```
backend/            FastAPI application (app/), requirements, .env.example
ml/                 training / evaluation / prediction scripts + shared utils
datasets/waste/     YOLO-format dataset (images/ + labels/, not committed)
models/             promoted model weights (not committed) - see models/README.md
runs/               Ultralytics training/evaluation outputs (not committed)
notebooks/          Colab/Kaggle notebooks
```

## 1. Environment setup

Python **3.10+** is required (tested locally with 3.14; Colab/Kaggle use 3.12).
All commands below are run from the project root.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

# 1) PyTorch first. With an NVIDIA GPU (CUDA 12.6 build; needs a recent driver):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#    CPU only instead:
# pip install torch torchvision

# 2) Everything else (backend + ML + test tools)
pip install -r backend/requirements-dev.txt

# 3) Verify
python ml/check_env.py
```

If `Activate.ps1` is blocked, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision          # see pytorch.org for the right CUDA variant
pip install -r backend/requirements-dev.txt
python ml/check_env.py
```

### Google Colab / Kaggle (training only)

PyTorch with CUDA is preinstalled. Enable a GPU runtime, then:

```bash
!git clone <your-repo-url> wastewise && cd wastewise
!pip install -q -r ml/requirements.txt
!python ml/check_env.py
```

### Expected `check_env.py` output (GPU machine)

```
torch         : 2.x.x+cu126
CUDA available: True
ultralytics   : 8.x.x
Auto device   : CUDA:0 (NVIDIA GeForce RTX ..., 6.0 GB)
...
Smoke test: yolov8n.pt on bus.jpg (device=0)
  N objects in ...s: bus, person, person, ...
Environment OK
```

On a CPU-only machine `Auto device` shows `CPU`; that is supported, only slower.

---

## 2. Dataset

Source: **GARBAGE CLASSIFICATION 3 (v2)**, Roboflow Universe, CC BY 4.0: 10,464 images
with six classes. Its `biodegradable` class is mapped explicitly to `organic`.
Details, attribution and known limitations: [datasets/README.md](datasets/README.md).

```bash
python ml/download_dataset.py     # ~196 MB into datasets/raw/, SHA-256 verified
python ml/prepare_dataset.py      # COCO -> YOLO, stratified 70/20/10 split (seed 42)
python ml/verify_dataset.py       # label/class/leakage checks; --check-images decodes all images
```

Resulting split (images): train 7,325 · val 2,092 · test 1,047. The split is
frozen in the committed `datasets/waste/manifest.csv`.

---

## 3. Demo (current prototype model)

Full explanation, 10-minute demo script and viva Q&A: [docs/WasteWise_Project_Guide.md](docs/WasteWise_Project_Guide.md).

The current model `yolov8s_prototype-b0c0f155` is a quick prototype: 10 epochs on
50% of the training images. Test-set mAP@50 is 0.272. Results are reliable only for
glass, organic and cardboard.

```bat
.venv\Scripts\activate.bat
python ml/predict.py --image demo_images --show          :: one window per image, press any key
python ml/predict.py --image path\to\photo.jpg --show    :: any photo of your own
python ml/evaluate.py                                    :: test-set metrics (prints the summary table)
```

Annotated images and a JSON file per image are saved in `runs/predict/<timestamp>/`.
Test-set plots and metrics are in `runs/evaluate/yolov8s_prototype-b0c0f155_test/`.

---

*The training, evaluation, prediction, API, testing and database
sections are added as each stage is implemented.*
