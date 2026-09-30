# WasteWise AI: Complete Project Guide, Demo Script and Viva Preparation

> Every number in this guide comes from real outputs in this repository:
> `datasets/waste/dataset_report.json`, `runs/detect/yolov8s_prototype/results.csv`,
> `runs/evaluate/yolov8s_prototype-b0c0f155_test/metrics.json` and `runs/predict/demo/`.
> If you retrain, these numbers change. Re-check them before the viva.

---

## Part A: The project in one minute

**What it is:** WasteWise AI is the computer-vision core of an *Autonomous Municipal
Waste Analytics Agent*. You give it a photo of waste. A **YOLOv8** object detector
finds each waste object and labels it as one of **six classes**: plastic, paper,
glass, metal, cardboard or organic. Deterministic Python code then turns those
detections into **counts, composition percentages, recyclable percentage and
contamination percentage**.

**The pipeline:**

```
Image → YOLOv8s → detections (class, confidence, bounding box)
      → counts per class → composition % → recyclable % / contamination % → JSON + annotated image
```

**Key design rule:** *YOLO perceives, Python calculates.* The neural network only
detects objects. Every number shown to a user (counts, percentages) is computed by
plain, testable Python code, never "guessed" by a model.

**Scientific boundary:** we measure **object counts**, not **weight**. A normal 2D
photo carries no information about mass or density, so we never claim kilograms.
"Contamination %" always means *object-count contamination*.

**Current status (honest):** a working **prototype**. The model was trained quickly
(10 epochs on half the training data, 33 minutes). It reaches **mAP@50 = 0.272** on
a held-out test set of 1,198 images. It works well for glass, organic and
cardboard, and poorly for plastic, paper and metal.

---

## Part B: Scope

### Built in this phase
| Part | File(s) |
|---|---|
| Dataset download, conversion, splitting, verification | `ml/download_dataset.py`, `ml/prepare_dataset.py`, `ml/verify_dataset.py` |
| Reproducible YOLOv8 training | `ml/train.py`, `ml/configs/train.yaml` |
| Test-set evaluation with plots and exports | `ml/evaluate.py` |
| Single-image / folder prediction with statistics | `ml/predict.py` |
| Statistics and contamination logic (shared with the future API) | `backend/app/services/statistics_service.py`, `contamination_service.py` |
| Canonical class list (single source of truth) | `backend/app/ml/label_map.py` |
| Cloud training kit (Kaggle / Colab) | `ml/package_for_cloud.py`, `notebooks/train_cloud.ipynb` |

### Not built yet (later phases; say so honestly if asked)
FastAPI backend (next stage), PostgreSQL storage, LangGraph agent, LLM chatbot,
React dashboard, PDF reports, alerts, IoT and route optimisation.

---

## Part C: Architecture

```
                ┌──────────────────────── ML pipeline (offline) ────────────────────────┐
 Raw datasets → │ download → prepare (map classes, COCO→YOLO, split) → verify → train → │ → models/best.pt
 (GC3, TACO)    │ evaluate (test set) → metrics.json / plots                              │   models/best.json
                └──────────────────────────────────────────────────────────────────────────┘

                ┌──────────────────────── Inference (online) ───────────────────────────┐
 Photo ───────→ │ YOLOv8s (loaded once) → detections → statistics_service               │ → JSON report
                │                                     → contamination_service            │ → annotated image
                └──────────────────────────────────────────────────────────────────────────┘
```

**Layer separation** (a common viva question: *why separate?*):
- `ml/` scripts handle training and evaluation. They are heavy and run offline.
- `backend/app/services/` holds pure Python maths with no ML and no I/O. It is easy
  to unit-test and is reused by both `predict.py` and the future FastAPI app.
- `backend/app/ml/label_map.py` is the only place the six class names are defined.
  `data.yaml` and the trained model's class names are checked against it
  automatically, so the class order can never silently drift.

---

## Part D: Technology choices and why

| Technology | Why |
|---|---|
| **YOLOv8 (Ultralytics)** | Required by the curriculum (YOLOv8 or ResNet). It is an **object detector**, not an image classifier, so it finds *multiple* objects per image with boxes. That is essential for counting. |
| **YOLOv8s** ("small") | 11.1 M parameters and 28.4 GFLOPs: noticeably more accurate than YOLOv8n but still fits a 6 GB laptop GPU at 640 px with batch 16 (about 4.3 GB GPU memory used). |
| **PyTorch 2.14 + CUDA 12.6** | Framework under Ultralytics; CUDA for the RTX 4050 GPU. |
| **Python 3.14** (code compatible with 3.10+) | Local machine version; Colab and Kaggle use 3.12. |
| **FastAPI + Pydantic** (next stage) | Required backend; typed request/response validation. |
| **PostgreSQL + SQLAlchemy + Alembic** (later) | History of analyses and individual detections, with schema migrations. |

---

## Part E: Dataset (explain this confidently; examiners love data questions)

### E1. The six classes (fixed order = YOLO class ids)
| id | class | recycling stream |
|---|---|---|
| 0 | plastic | recyclable |
| 1 | paper | recyclable |
| 2 | glass | recyclable |
| 3 | metal | recyclable |
| 4 | cardboard | recyclable |
| 5 | organic | **contaminant** |

### E2. Sources
1. **GARBAGE CLASSIFICATION 3 (v2)**, Roboflow Universe, **CC BY 4.0**. 10,464 images with
   the same six classes (its `biodegradable` is renamed to `organic`). Downloaded from a
   Hugging Face mirror at a pinned revision, with SHA-256 checksums verified.
2. **TACO (Trash Annotations in Context)**, Proença & Simões 2020. 1,500 real-world
   litter photos, 4,784 annotations in 60 categories, mapped to our six classes **by material**.

### E3. Why two sources?
- GC3 has all six classes, including organic, but mostly **clean product-style photos**.
- TACO adds **real, cluttered outdoor scenes**, but has only **8 food-waste annotations**,
  so it cannot teach "organic" on its own.
- Combining them gives all six classes plus some real-world realism.

### E4. Class mapping (never silent)
- Each source has a YAML config (`ml/configs/sources/*.yaml`) with an explicit `class_map`
  and `drop` list. Any label not listed makes preparation **fail with an error**.
- TACO examples: clear plastic bottle → plastic; drink can → metal; glass bottle → glass;
  corrugated carton → cardboard; food waste → organic.
- Deliberately dropped: cigarettes (667), unlabelled litter (517), Tetra Pak drink cartons
  (45, composite material), tissues (42, soiled paper), rope, shoes, batteries and blister packs.
  1,323 TACO boxes were dropped in total. Those objects stay in the images as unlabelled background.

### E5. Preprocessing
- **YOLO label format:** one `.txt` per image; each line is
  `class_id x_center y_center width height`, all normalised to 0–1.
- COCO boxes `[x, y, w, h]` in pixels are converted to that format and clipped to the image.
  17 boxes smaller than 1 px after clipping were removed.
- TACO photos (up to 6,000 px) were **EXIF-rotated**, because the annotations use the
  rotated frame, and resized to at most 1,280 px on the long side. Normalised labels do
  not change with a uniform resize.
- GC3 images were already 416×416. Roboflow stretched them and added flips and
  90° rotations to the training images.

### E6. Split: 70 / 20 / 10, stratified, per source, frozen
- The **published GC3 split was unusable**: its test split had **zero glass** and only
  20 cardboard boxes. We pooled everything and re-split.
- **Stratification:** each image is assigned to a group by the *rarest* class it contains,
  and each group is split 70/20/10 with a fixed seed (42). Rare classes are therefore
  represented in every split.
- **Per source:** each source is split independently, so adding TACO did not change GC3's split.
- **Leakage check:** images are compared by SHA-256 hash. 1 exact duplicate was removed,
  and the verifier confirms no identical image appears in two splits.
- **Frozen:** `datasets/waste/manifest.csv` records the split for every file and is committed
  to git. Rebuilding reproduces it byte for byte (verified).

### E7. Final numbers
| | train | val | test |
|---|---|---|---|
| Images | 8,374 | 2,391 | 1,198 |
| Boxes | 54,321 | 14,765 | 8,447 |

| Class (images / boxes) | train | val | test |
|---|---|---|---|
| plastic | 1,815 / 5,395 | 522 / 1,724 | 259 / 1,070 |
| paper | 1,350 / 3,238 | 384 / 857 | 189 / 496 |
| glass | 1,952 / 5,549 | 558 / 1,837 | 269 / 677 |
| metal | 1,529 / 4,522 | 439 / 1,306 | 216 / 557 |
| cardboard | 1,160 / 3,407 | 333 / 990 | 164 / 505 |
| organic | 1,611 / **32,210** | 454 / 8,051 | 229 / 5,142 |

- **Imbalance:** organic accounts for 32,210 of 54,321 training boxes (**59%**),
  because food images often contain many small items (e.g. dozens of cherries).
- **TACO share:** only 1,049 of 8,374 training images (**12.5%**).

### E8. Verification (`ml/verify_dataset.py`)
It checks that:
- class names match the canonical list;
- every image has a label file;
- each line has 5 values, class ids are 0–5, and coordinates are in [0, 1];
- no image is duplicated across splits;
- optionally (`--check-images`), every image decodes.

To prove the checker works, we injected four faults into a copy of the dataset; it
caught all four. One bug found this way, duplicates without labels going unnoticed, was fixed.

---

## Part F: How YOLOv8 works (theory questions)

### F1. Detection vs classification
- **Classification** gives one label per image ("this image is plastic").
- **Detection** gives *every* object with a **class, a confidence and a bounding box**.
  Counting and composition require detection.

### F2. Architecture (YOLOv8s)
- **Backbone:** a CSPDarknet-style network built from **C2f** blocks, ending in **SPPF**
  (spatial pyramid pooling, fast). It extracts visual features at several resolutions.
- **Neck:** a **PAN-FPN** that merges fine detail (small objects) with high-level meaning
  (large objects) across three scales.
- **Head:** **decoupled** (separate branches for class and box) and **anchor-free**.
  At 640 px it predicts on grids of 80×80, 40×40 and 20×20 cells (strides 8, 16 and 32),
  giving 8,400 candidate boxes per image.
- **Model size:** 129 layers (72 after fusing layers for inference),
  **11,137,922 parameters**, **28.4 GFLOPs**.

### F3. Training losses (the curves you will show)
| Loss | Meaning | Weight used |
|---|---|---|
| **box loss** (CIoU) | How well predicted boxes overlap the true boxes | 7.5 |
| **cls loss** (BCE) | How correct the class probabilities are | 0.5 |
| **DFL** (Distribution Focal Loss) | Sharpens the box edges: each edge distance is predicted as a small probability distribution | 1.5 |

A **task-aligned assigner** decides which of the 8,400 candidates are responsible for
each real object during training.

### F4. After prediction
- **Confidence:** the model's certainty in a detection (0–1).
- **Confidence threshold:** detections below it are discarded. We use **0.35** for demos.
- **NMS (non-maximum suppression):** when several boxes overlap the same object by more
  than the IoU threshold (**0.7**), only the most confident one is kept.

### F5. Transfer learning
- Training started from **COCO-pretrained** YOLOv8s weights (80 everyday classes).
- 349 of 355 weight tensors were transferred. Only the final class-prediction layers were
  re-initialised, because we have 6 classes instead of 80.
- This is why even 10 short epochs produced a usable model.

---

## Part G: Training

### G1. Settings of the current model (`yolov8s_prototype`)
| Setting | Value | Why |
|---|---|---|
| Starting weights | `yolov8s.pt` (COCO) | Transfer learning |
| Epochs | **10** | Quick prototype (the time limit) |
| Fraction of training data | **0.5** (4,187 of 8,374 images) | Halves the time per epoch |
| Image size | 640 | Ultralytics standard; keeps small TACO objects visible |
| Batch size | 16 | Fits the 6 GB GPU |
| Optimizer | `auto` → **AdamW, lr 0.001, momentum 0.9** | Ultralytics picks AdamW for short runs (< 10,000 iterations; ours was 10 × 262 = 2,620) |
| LR schedule | 3 warm-up epochs, then linear decay to 1% (`lrf=0.01`) | Stable start |
| Weight decay | 0.0005 | Regularisation |
| Gradient accumulation | nominal batch 64 ÷ batch 16 → optimizer step every 4 batches | Ultralytics default (`nbs=64`) |
| Mixed precision (AMP) | on | Faster, less GPU memory |
| Seed / deterministic | 42 / true | Reproducibility |
| Augmentation | HSV colour jitter, translate 0.1, scale 0.5, horizontal flip 0.5 | Robustness |
| Mosaic | **effectively OFF** | `close_mosaic=10` disables mosaic for the last 10 epochs, which is the whole run. This is a known cause of lower accuracy. |
| Data-loader workers | 0 | Windows memory limit (see G3) |
| Hardware | RTX 4050 Laptop GPU (6 GB), 16 GB RAM | |
| **Training time** | **33.2 minutes** | |

### G2. Training curves (`runs/detect/yolov8s_prototype/results.csv`)
| Epoch | train box / cls / dfl | val box / cls / dfl | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| 1 | 1.656 / 2.170 / 1.649 | 1.793 / 4.148 / 1.843 | 0.437 | 0.150 | 0.128 | 0.066 |
| 5 | 1.606 / 1.678 / 1.626 | 1.386 / 3.904 / 1.539 | 0.467 | 0.223 | 0.199 | 0.125 |
| 10 | 1.378 / 1.234 / 1.469 | 1.238 / 3.622 / 1.418 | 0.615 | 0.262 | 0.264 | 0.173 |

**How to read this:**
- All training losses go down, so the model is learning.
- Validation box and DFL losses also go down, so it generalises to unseen images.
- mAP roughly doubled (0.128 → 0.264), and the **best epoch was the last one**. The model
  was still improving, which is classic **underfitting**: it needs more training.
- Validation classification loss is noisy (between 3.21 and 4.15) and much higher than the
  training value (1.23). Class identity is the hardest part, which matches the weak classes.
  It did not trend upward, so there is no clear overfitting.

### G3. Engineering challenge you can talk about (examiners like this)
The first full 100-epoch run **crashed during epoch 2**. The investigation, done by
measurement rather than guessing:
- Windows' **memory commit limit** was the problem, not the GPU. Other applications
  already committed about 19 GB of about 25 GB.
- The training process committed about 9 GB, and **each data-loader worker about 1.5 GB**.
- Ultralytics keeps **2× as many validation workers** alive after the first validation.
  That is why it crashed at epoch 2.
- A separate Windows CUDA pinned-memory error was also found and fixed.

**Fixes:**
- `workers=0` (measured stable);
- pinned memory disabled on Windows;
- an **auto-resume loop** that continues from `last.pt` after a crash;
- a **Kaggle/Colab notebook** for full training on a cloud GPU.

### G4. Outputs of every training run
Saved in `runs/detect/<name>/`:
- `weights/best.pt` and `weights/last.pt`;
- `results.csv` and `results.png` (training curves);
- `args.yaml` (all Ultralytics settings);
- `run_config.json` (our config, the environment, a dataset fingerprint and the final metrics);
- the confusion matrix and PR/F1 curves on the validation set.

**Promotion:** `--promote` copies the best weights to `models/best.pt` and writes
`models/best.json`. The model version (`yolov8s_prototype-b0c0f155`) combines the run
name with the first 8 characters of the weights' SHA-256 hash, so two different models
can never share a version name.

---

## Part H: Evaluation (the most important part to understand)

### H1. The metrics in plain words
- **IoU (Intersection over Union)** = overlap area ÷ combined area of the predicted and true boxes.
  A detection counts as correct (a **TP**) if the class is right *and* IoU ≥ threshold.
- **Precision** = TP / (TP + FP): *when it says "glass", how often is it glass?*
- **Recall** = TP / (TP + FN): *of all real glass objects, how many did it find?*
- **F1** = 2·P·R / (P + R): the balance of the two.
- **AP (Average Precision)** = the area under the precision-recall curve for one class.
  The curve is traced by sweeping the confidence threshold from high to low.
- **mAP@50** = AP averaged over all 6 classes, with IoU threshold 0.50.
- **mAP@50-95** = mAP averaged over IoU thresholds 0.50, 0.55, …, 0.95. It is stricter
  and rewards precise boxes. It is the main COCO benchmark metric.

### H2. Evaluation protocol
- **Held-out test set:** 1,198 images the model never saw in training, and never used
  to choose the best epoch (the validation set did that).
- mAP is computed at **confidence 0.001**, so the full precision-recall curve is used.
  This is the standard; evaluating at 0.35 or 0.50 would cut the curve short and under-report mAP.
- The **confusion matrix** is computed at **confidence 0.25** (Ultralytics standard).
  Box matching for NMS uses IoU **0.7**.
- The **per-class precision and recall** in the table are taken at the single confidence
  where mean F1 is highest: **0.363**.

### H3. Test-set results (`runs/evaluate/yolov8s_prototype-b0c0f155_test/summary.txt`)
| | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| **Overall** | **0.654** | **0.270** | **0.382** | **0.272** | **0.179** |
| Plastic | 0.773 | 0.038 | 0.073 | 0.088 | 0.049 |
| Paper | 1.000* | 0.000 | 0.000 | 0.007 | 0.004 |
| Glass | 0.421 | 0.706 | 0.528 | **0.555** | **0.430** |
| Metal | 0.601 | 0.050 | 0.093 | 0.128 | 0.059 |
| Cardboard | 0.413 | 0.400 | 0.407 | 0.333 | 0.254 |
| Organic | 0.715 | 0.424 | 0.532 | 0.521 | 0.275 |

\* **Paper's precision of 1.000 is a reporting quirk, not a strength.** At the chosen
confidence (0.363), the model makes *no* paper predictions. Precision is then 0/0, and
Ultralytics shows 1.0 by default. The honest measures for paper are recall 0.000 and
mAP 0.007. The high precision of plastic and metal comes from the same effect: very few,
high-confidence predictions.

### H4. By source: clean photos vs real litter
| Subset | Images | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| GC3 (clean photos) | 1,047 | 0.617 | 0.299 | 0.294 | 0.192 |
| TACO (real outdoor litter) | 151 | 0.530 | 0.032 | **0.023** | 0.018 |

The model almost completely fails on real-world litter scenes. This is **domain shift**:
TACO objects are small, dirty, partly hidden and on busy backgrounds, and TACO was only
12.5% of the training images. **It is the most important limitation for a municipal use case.**

### H5. Confusion matrix at confidence 0.25 (`confusion_matrix_normalized.png`)
How to read it: **columns = true class, rows = predicted class**. Each column sums to 1.
- The **bottom row "background"** = real objects the model **missed** (false negatives).
- The **right column "background"** = detections where **no object exists** (false positives).

| True class | Correct | Missed (background) | Main confusion |
|---|---|---|---|
| glass | **0.74** | 0.24 | – |
| organic | 0.53 | 0.46 | – |
| cardboard | 0.46 | 0.52 | – |
| metal | 0.07 | 0.46 | **→ glass 0.35** |
| plastic | 0.04 | **0.67** | → glass 0.15, → organic 0.09 |
| paper | 0.00 | **0.62** | **→ cardboard 0.22**, → glass 0.10 |

Of the false positives (right column): 57% are labelled organic, 31% glass and 10% cardboard.

**Main message:** the biggest problem is **missed objects** (low recall), not wrong labels.
The real confusions are **metal ↔ glass** (bottle caps and cans next to bottles) and
**paper ↔ cardboard** (similar material).

### H6. Choosing the confidence threshold (`BoxF1_curve.png`)
- Mean F1 across all classes peaks at **0.27 at confidence 0.363**.
- The demo uses **0.35**; the original spec suggested 0.50. At 0.50 more correct
  detections would be discarded, lowering recall even further.
- Per class, F1 peaks at very different confidences: glass around 0.55–0.6, organic
  around 0.3, and metal and plastic below 0.1. So one global threshold is a compromise.

### H7. Speed
Inference takes about **7.7 ms per image** (about 130 images per second) on the RTX 4050
at 640 px, plus about 1 ms of pre- and post-processing.

---

## Part I: Statistics and contamination logic (you must be able to do this by hand)

Code: `backend/app/services/statistics_service.py` and `contamination_service.py`.

### I1. Formulas
```
counts[c]              = number of detections of class c   (all six classes always present, 0 if none)
total                  = sum of counts
composition_percent[c] = counts[c] / total × 100             (rounded to 2 decimals)
recyclable_percent     = (plastic+paper+glass+metal+cardboard) / total × 100
contamination_percent  = organic / total × 100               (for the recycling/dry stream)
if total == 0          → every percentage = 0, severity = "none"   (no division by zero)
```

### I2. Severity (spec thresholds, configurable)
| Contamination | Severity |
|---|---|
| no objects detected | none |
| < 10% | low |
| 10% – 20% (inclusive) | moderate |
| > 20% | high |

### I3. Worked example (spec example)
Plastic 10, paper 4, metal 2, organic 4 → total 20.
- Contamination = 4 / 20 × 100 = **20%** → **moderate** (exactly 20 is not "> 20").
- Recyclable = 16 / 20 × 100 = **80%**.
- Composition: plastic 50%, paper 20%, metal 10%, organic 20%, glass 0%, cardboard 0%.

### I4. Why object count and not weight?
A photo does not show mass. A single watermelon rind and a bottle cap both count as
one object, even though their weights differ enormously. True mass needs a weighing
sensor or calibrated depth data, which are listed as future extensions.

### I5. Why is the rounding sometimes off by 0.01?
Each percentage is rounded separately. For example, 1/3 + 1/3 + 1/3 → 33.33 × 3 = 99.99.
This is documented, not a bug.

---

## Part J: Prediction demo results (`runs/predict/demo/`)

These are held-out **test images**, never seen during training. Confidence threshold 0.35.

| Image | Ground truth | Model found | Result |
|---|---|---|---|
| `gc3_biodegradable1489…` (fruit) | 9 organic | **9 organic** (conf 0.69–0.88) | Contamination 100%, severity high |
| `gc3_cardboard52…` | 5 cardboard | **5 cardboard** (0.56–0.65) | Recyclable 100% |
| `gc3_glass2841…` (bottles) | 6 glass | **5 glass** (0.43–0.90) | Missed one bottle |
| `gc3_glass507…` (beer bottle) | 1 glass + 1 metal cap | **glass 0.97 + metal 0.37** | Both found |
| `taco_batch_12_000059` (street litter) | 6 plastic | **nothing** | Zero-detection handling shown |

Each image produces:
- a printed table of detections, counts and percentages;
- `<name>_annotated.jpg` (boxes plus a summary strip above the image);
- `<name>.json`, in the same structure the FastAPI endpoint will return.

---

## Part K: Limitations (say these before the examiner does)

1. **A short prototype training run:** 10 epochs, half the data, mosaic off. The model is still underfitting.
2. **Plastic, paper and metal recall is below 5%,** so counts and contamination are *undercounts* for these classes.
3. **Domain shift:** mAP@50 is 0.023 on real litter photos (TACO) versus 0.294 on clean photos (GC3).
4. **Class imbalance:** 59% of the training boxes are organic.
5. **Dataset realism:** GC3 is mostly clean web photos, and its "organic" class is mostly *fresh food*, not bin waste.
6. **Labelling convention:** bottle caps are separate metal objects, so a capped bottle counts as 2 objects.
7. **Count-based metrics only:** no weight or volume.
8. **One global confidence threshold** for all classes.

---

## Part L: Improvements and future work

**For the model, in order of expected impact:**
1. Full training: all 8,374 images, 50–100 epochs, mosaic on. The Kaggle notebook is ready.
2. More real-world bin and street images (more TACO-like data), plus more distinct paper examples.
3. Reduce organic's dominance, e.g. cap the boxes per image or sample organic-heavy images less often.
4. Try YOLOv8m, or a larger image size for small objects, if GPU memory allows.
5. Per-class confidence thresholds.

**For the system (later phases in the spec):**
1. FastAPI `POST /api/v1/analyses` (next stage).
2. PostgreSQL history of analyses and detections.
3. Threshold alerts (the "autonomous" behaviour).
4. LangGraph agent with safe, typed tools.
5. React dashboard.
6. PDF reports.

---

## Part M: Commands reference

```bat
:: one-time setup
python -m venv .venv
.venv\Scripts\activate.bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r backend/requirements-dev.txt
python ml/check_env.py

:: dataset
python ml/download_dataset.py
python ml/prepare_dataset.py --overwrite
python ml/verify_dataset.py

:: training (the prototype that was actually used)
python ml/train.py --epochs 10 --fraction 0.5 --name yolov8s_prototype --promote

:: evaluation and prediction
python ml/evaluate.py
python ml/predict.py --image demo_images --show
python ml/predict.py --image path\to\photo.jpg --show

:: cloud training bundle
python ml/package_for_cloud.py
```

---

## Part N: The 10-minute demo script

### Before the demo (15 minutes earlier)
- [ ] Laptop plugged in. Close Chrome and Docker Desktop (frees RAM).
- [ ] Open a terminal: `cd /d "D:\Minor Project"` then `.venv\Scripts\activate.bat`.
- [ ] **Warm-up run:** `python ml/predict.py --image demo_images`. The first run loads CUDA
      and takes a few seconds longer.
- [ ] **Test `--show` once.** The pop-up windows have not been tried on your machine yet.
      If a window doesn't appear, run without `--show` and open the saved images instead.
- [ ] Open these files in VS Code tabs, in this order:
  1. `README.md`
  2. `datasets/README.md`
  3. `runs/detect/yolov8s_prototype/results.png`
  4. `runs/evaluate/yolov8s_prototype-b0c0f155_test/summary.txt`
  5. `…/confusion_matrix_normalized.png`
  6. `…/BoxF1_curve.png`
  7. `backend/app/services/contamination_service.py`
- [ ] Backup: `runs/predict/demo/*_annotated.jpg` already exist, in case the live run fails.

### 0:00 – 1:00 · Problem and idea
> "Cities need to know what is in their waste to improve recycling, but manual inspection
> is slow, unhygienic and inconsistent. WasteWise AI analyses a photo of waste: it detects
> every object, classifies it into six categories and calculates composition and recycling
> contamination automatically."

Also mention: this phase is the computer-vision core. The agent, dashboard and reports come later.

### 1:00 – 2:30 · Architecture (show the README)
- Image → YOLOv8s → detections → Python statistics → JSON and annotated image.
- **"YOLO perceives, Python calculates."** The model never produces the percentages.
- "We measure object counts, not weight, because a 2D photo has no mass information."

### 2:30 – 3:45 · Dataset (show `datasets/README.md`)
- Two sources: GC3 (10,464 images, six classes) and TACO (1,500 real litter photos,
  60 categories mapped by material).
- "The published split had zero glass in the test set, so we re-split 70/20/10,
  stratified by the rarest class, with duplicate checks. The split is frozen in a manifest."
- Final: 11,963 images; 8,374 / 2,391 / 1,198.

### 3:45 – 6:00 · Live demo (the highlight)
```bat
python ml/predict.py --image demo_images --show
```
Talk through each window:
- **Fruit:** "9 organic objects. For a dry recycling bin that is 100% contamination, severity high."
- **Cardboard:** "5 cardboard, 100% recyclable, 0% contamination."
- **Bottles:** "5 of 6 bottles found; one missed, which is honest prototype behaviour."
- **Beer bottle:** "It even separates the metal cap from the glass bottle."
- **Street litter:** "Nothing detected. The system reports that no composition can be
  calculated instead of dividing by zero. It also shows our main limitation on real-world scenes."

Point at the terminal: the class, confidence and box of every detection, counts, percentages
and severity. Mention that a JSON file is saved per image, the same format the API will return.

### 6:00 – 7:30 · Evaluation (show `summary.txt`, then the confusion matrix)
- "Evaluated on 1,198 held-out test images: precision 0.654, recall 0.270, mAP@50 0.272, mAP@50-95 0.179."
- Glass is best (mAP@50 0.555), then organic (0.521). Paper is weakest (0.007).
- Confusion matrix: "The biggest issue is missed objects, the bottom row. The real
  confusions are metal with glass, and paper with cardboard."
- F1 curve: "This is why we use confidence 0.35 rather than 0.50."

### 7:30 – 8:30 · Training (show `results.png`)
- "Transfer learning from COCO; 10 epochs on half the data in 33 minutes on an RTX 4050."
- "Losses fall and mAP rises until the last epoch. It is underfitting, so more training will help."
- Optional: the Windows memory crash story and how it was diagnosed.

### 8:30 – 9:30 · Limitations (say them yourself)
Prototype training; low recall for plastic, paper and metal; domain shift on real litter
(TACO mAP@50 0.023); class imbalance (59% organic boxes).

### 9:30 – 10:00 · Next steps and close
Full training on Kaggle, the FastAPI backend, PostgreSQL history, alerts, the LangGraph
agent and the dashboard.
> "The core pipeline, from image to detection to reliable statistics, is working and
> measured honestly. The next phases turn it into the full autonomous analytics agent."

### If something goes wrong
| Problem | What to do |
|---|---|
| `--show` window doesn't appear | Run without `--show`; open the images in `runs\predict\<timestamp>\` |
| CUDA / GPU error | Add `--device cpu` (slower but works) |
| Anything else | Show the pre-saved `runs\predict\demo\*_annotated.jpg` and `summary.txt` |

---

## Part O: Viva questions and answers

### Project and scope
**Q: What problem does your project solve?**
Manual waste inspection is slow, unhygienic and inconsistent. We automatically detect and
count waste objects in images and compute composition and recycling contamination, as the
data layer for municipal decisions.

**Q: Why is it called an "autonomous agent" if there's no agent yet?**
This phase is the perception and analytics core. The autonomous parts (threshold alerts,
recommendations, the LangGraph tool-using agent) are later phases that build on these
trusted numbers.

**Q: Why can't you give the weight of the waste?**
A 2D image contains no mass or density information. Weight would need a weighing sensor or
calibrated depth data. Claiming kilograms would be scientifically invalid, so we report counts.

**Q: Why not let an LLM or the model calculate the percentages?**
Percentages must be exact and reproducible. Deterministic Python is testable and cannot
"hallucinate". The model only perceives.

### Dataset
**Q: Where did your data come from? Is it legal to use?**
GARBAGE CLASSIFICATION 3 (Roboflow, CC BY 4.0, attributed in `datasets/README.md`) and TACO
(an academic dataset; cited). Both are public datasets for research use.

**Q: How many images and boxes?**
11,963 images: train 8,374, val 2,391, test 1,198. Training boxes: 54,321.

**Q: Why 70/20/10?**
Enough training data, a large validation set for choosing the best epoch, and an untouched
test set for the final honest score. It is a common starting point, recommended in our spec.

**Q: What is the difference between the validation and test sets?**
Validation is used *during* training to track progress and pick `best.pt`. Test is used
only once at the end, so its score isn't biased by any choices we made.

**Q: What is stratified splitting? Why "rarest class"?**
Stratification keeps class proportions similar across splits. An image with several classes
is grouped by its rarest one, so minority classes are guaranteed a share of every split.

**Q: What is data leakage and how did you prevent it?**
Leakage means the same (or nearly the same) image appears in both train and test, which
inflates scores. We hash every image with SHA-256, remove exact duplicates, and verify that
no hash appears in two splits. Limitation: near-duplicates (e.g. slightly different crops)
are not detected.

**Q: How did you handle TACO's 60 categories?**
An explicit material-based mapping in a YAML file. Categories that don't fit (cigarettes,
composites, textiles) are explicitly dropped, never silently renamed. Unknown labels stop
the script with an error.

**Q: What does a YOLO label line mean?**
`class_id x_center y_center width height`. All values except the class are fractions of the
image width and height, so they are resolution-independent. For example,
`2 0.5 0.5 0.2 0.4` is a glass object centred in the image, 20% as wide and 40% as tall as the image.

**Q: Why did you rotate TACO images?**
Phone photos store their orientation in EXIF metadata, and TACO's annotations match the
rotated view. We apply the rotation before saving, so boxes and pixels agree. We verified
this visually.

**Q: Is your dataset imbalanced?**
Yes, at box level: 59% of training boxes are organic, because food images contain many
small items. At image level the classes are closer (1,160–1,952 training images each).

### Model and theory
**Q: Why YOLOv8 and not ResNet?**
ResNet is a classifier: one label per image. We need *multiple objects with locations* to
count them, so we need a detector. YOLOv8 is fast, accurate and well supported.

**Q: Why the "s" (small) variant?**
It's the best accuracy that fits a 6 GB laptop GPU comfortably: 11.1 M parameters and
28.4 GFLOPs. "n" is less accurate; "m" and up need more memory and time.

**Q: What does "anchor-free" mean?**
Older YOLOs predicted offsets from predefined anchor box shapes. YOLOv8 predicts the object
centre and distances to the box edges directly, which is simpler and generalises better.

**Q: What are the three losses?**
Box loss (CIoU: overlap, centre distance and aspect ratio), classification loss (binary
cross-entropy) and Distribution Focal Loss (precise box edges). The weights are 7.5, 0.5 and 1.5.

**Q: What is NMS?**
Non-maximum suppression. The model proposes many overlapping boxes per object; NMS keeps the
most confident one and removes others that overlap it by more than IoU 0.7.

**Q: What is IoU?**
Intersection over Union: overlap area ÷ union area of two boxes. 1 = perfect match, 0 = no overlap.

**Q: What is transfer learning here?**
We started from YOLOv8s weights trained on COCO (80 everyday classes). The feature extractor
already knows edges, shapes and textures, so it learns waste classes much faster. 349 of 355
tensors were reused; only the class head was re-initialised for 6 classes.

**Q: What is the confidence score?**
The model's certainty (0–1) that a box contains an object of that class. Detections below
the threshold (0.35 in our demo) are discarded.

### Training
**Q: Which optimizer and learning rate?**
Ultralytics' `auto` chose **AdamW, lr 0.001, momentum 0.9**, because the run was short
(2,620 iterations; it uses AdamW up to 10,000). The learning rate comes from Ultralytics'
formula `0.002 × 5 / (4 + number of classes)` = 0.01 / 10 = **0.001**. This came with
3 warm-up epochs, linear decay and weight decay 0.0005.

**Q: What is an epoch? A batch?**
An epoch is one full pass over the training data. A batch is 16 images processed together;
one epoch was 262 batches (4,187 images ÷ 16).

**Q: What is mixed precision (AMP)?**
Some calculations run in 16-bit instead of 32-bit floating point. It's faster and uses less
GPU memory, with negligible accuracy loss.

**Q: What is mosaic augmentation, and why was it off?**
Mosaic stitches 4 training images into one, which helps with small objects and context.
Ultralytics turns it off for the final `close_mosaic=10` epochs to fine-tune on normal images.
Because we trained only 10 epochs, it was off the whole time. That's a known reason for our
lower accuracy, and it's fixed by training longer.

**Q: How do you ensure reproducibility?**
- Fixed seed (42) and deterministic mode.
- A frozen, committed data split (`manifest.csv`).
- Pinned dataset revision with checksums.
- `run_config.json` records the config, library versions, GPU and a dataset fingerprint.
- The model version includes the weights hash.

**Q: Why only 10 epochs on half the data?**
Local training was limited by RAM (Windows memory commit), not the GPU. For the prototype
demo we prioritised a working, measured model in 33 minutes. Full training is prepared on Kaggle.

**Q: Is the model overfitting or underfitting?**
Underfitting. Training and validation losses were both still decreasing, and the best mAP
was at the final epoch. Overfitting would show validation loss rising while training loss
falls, which we did not see.

**Q: What problems did you face?**
The Windows memory crash, diagnosed by measuring commit charge per process: each data-loader
worker needed about 1.5 GB, and Ultralytics keeps twice as many validation workers alive.
Also a CUDA pinned-memory error. Fixes: workers=0, pinned memory off, auto-resume, and a cloud notebook.

### Evaluation
**Q: Explain precision and recall with your numbers.**
Overall precision 0.654: when the model detects something, it's right about 65% of the time.
Recall 0.270: it finds only about 27% of all objects. It's cautious, and misses more than it mislabels.

**Q: What is mAP@50 vs mAP@50-95?**
mAP@50 counts a box as correct at ≥ 50% overlap. mAP@50-95 averages over stricter overlaps up
to 95%, so it also rewards precise box placement. Ours: 0.272 and 0.179.

**Q: Why is paper's precision 1.000 when the model is bad at paper?**
At the reporting confidence (0.363) the model makes zero paper predictions. Precision becomes
0/0, which Ultralytics reports as 1. Recall 0.000 and mAP 0.007 show the truth: paper is not detected.

**Q: Which class is best and worst, and why?**
Glass is best (mAP@50 0.555, 74% correctly detected in the confusion matrix): bottles have
distinctive shapes. Paper is worst (0.007): it looks like cardboard, and flat, crumpled
sheets have no consistent shape.

**Q: Which classes are confused?**
At confidence 0.25: metal → glass 35% (caps and cans next to bottles), paper → cardboard 22%,
plastic → glass 15%. But the largest error is *missing* objects entirely: 67% of plastic and
62% of paper get no detection.

**Q: Why evaluate at confidence 0.001 but demo at 0.35?**
mAP needs the *whole* precision-recall curve, which requires keeping even low-confidence
predictions; 0.001 is the standard. For the demo we pick the threshold where F1 is highest
(about 0.36), which is the best balance of false alarms and misses.

**Q: Why is performance so low on TACO?**
Domain shift. TACO objects are small, dirty and partly hidden on grass and sand, while most
training images are clean product photos. TACO is only 12.5% of the training data.

**Q: Is mAP@50 0.272 good?**
For a 10-epoch, half-data prototype it demonstrates that the pipeline works, and glass and
organic are usable. It is not good enough for deployment. Published YOLOv8 results on similar
waste datasets are typically much higher after full training, which is our next step.
Don't quote a specific external number unless you have its source.

**Q: How would you improve accuracy?**
In order: full training (all data, more epochs, mosaic on), more real-world images, fewer
organic-dominated images, per-class thresholds, and possibly YOLOv8m.

### Statistics logic
**Q: Calculate contamination for plastic 5, paper 2, glass 1, metal 1, cardboard 2, organic 1.**
Total 12. Contamination = 1/12 × 100 = **8.33%** → low. Recyclable = 11/12 = **91.67%**.
Composition: plastic 41.67%, paper 16.67%, glass 8.33%, metal 8.33%, cardboard 16.67%, organic 8.33%.

**Q: What happens when nothing is detected?**
All counts are 0, all percentages are 0, severity is "none", and a message says no reliable
composition could be calculated. There is no division by zero.

**Q: Why is organic "contamination"?**
In a dry or recycling waste stream, food waste contaminates recyclables: it soils paper and
cardboard and makes them unrecyclable. The allowed set is configurable per stream in `contamination_service.py`.

**Q: What if the model detects a class you don't know?**
`canonical_name()` raises `UnknownWasteClassError`. At startup, the model's class names are
also checked against the canonical list, and loading fails if they differ.

**Q: What are the limits of count-based contamination?**
One big item and one tiny item weigh very differently but count equally. Missed detections
also bias the counts; for example, low plastic recall would *overstate* organic's share.

### Engineering
**Q: How is your code organised, and why?**
- `ml/`: offline training and evaluation.
- `backend/app/services`: pure calculation.
- `backend/app/ml`: model and class definitions.

Separation lets us unit-test the maths without a GPU and reuse it in the API.

**Q: Why store a model version?**
So every analysis result can be traced to the exact model that produced it. The version
includes a hash of the weights file.

**Q: How will the FastAPI backend use the model?**
It loads it **once at startup** (not per request), validates its class names, then for each
uploaded image runs detection → statistics → contamination and returns the JSON structure
you saw in the demo.

**Q: What tests do you have?**
The dataset verifier was tested with injected faults (all 4 caught). Unit and API tests with
pytest (counts, percentages, zero detections, invalid classes, uploads) come in the testing
stage. Be honest that they aren't written yet.

### Tricky questions
**Q: Did you train on the test set?**
No. The split is frozen and hash-checked. The test set was used only in `evaluate.py`, after training.

**Q: Your demo images: were they in training?**
No. They were copied from the held-out test split (see `datasets/waste/manifest.csv`, split = test).

**Q: Why not just use a bigger model?**
The bottleneck is training length and data realism, not model capacity. A bigger model on
the same 10 epochs would still underfit, and it needs more GPU memory.

**Q: What does 0.37 confidence on the bottle cap mean?**
The model is fairly unsure. It barely passed our 0.35 threshold. At 0.50 it would have been dropped.

**Q: If a bin is full of plastic the model misses, what happens to contamination?**
Missed plastic lowers the total count, so organic's share, and therefore the contamination %,
is overstated. This is why recall matters for this application.

---

## Part P: Glossary

| Term | Meaning |
|---|---|
| Epoch | One pass over all training images |
| Batch | Images processed together in one step (16) |
| Bounding box | Rectangle around an object `[x1, y1, x2, y2]` in pixels |
| Confidence | Model certainty for a detection (0–1) |
| IoU | Overlap ÷ union of two boxes |
| NMS | Removes duplicate overlapping boxes |
| TP / FP / FN | Correct detection / false alarm / missed object |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| AP / mAP | Area under the PR curve / its mean over classes |
| Transfer learning | Starting from weights trained on another dataset (COCO) |
| Augmentation | Random changes to training images (flip, colour, scale) for robustness |
| Mosaic | Augmentation combining 4 images into one |
| Underfitting | Model still improving; hasn't learned enough yet |
| Overfitting | Model memorises training data; validation gets worse |
| Domain shift | Test images look different from training images |
| Stratified split | Split that preserves class proportions |
| Data leakage | Test data (or copies of it) seen during training |
| AMP | Automatic mixed precision (16-bit maths) |
| COCO | Large public dataset (80 classes) used for pretraining |

---

## Part Q: Numbers cheat sheet (memorise these)

| Item | Value |
|---|---|
| Classes | 6: plastic, paper, glass, metal, cardboard, organic |
| Images (train / val / test) | 8,374 / 2,391 / 1,198 (total 11,963) |
| Sources | GC3 10,464 + TACO 1,500 (1 duplicate removed) |
| Model | YOLOv8s, 11.1 M params, 28.4 GFLOPs |
| Training | 10 epochs, 50% data (4,187 images), batch 16, 640 px, AdamW lr 0.001, 33.2 min |
| Test precision / recall | 0.654 / 0.270 |
| Test mAP@50 / mAP@50-95 | **0.272 / 0.179** |
| Best class | Glass: mAP@50 0.555 |
| Worst class | Paper: mAP@50 0.007 |
| GC3 vs TACO mAP@50 | 0.294 vs 0.023 |
| Demo confidence threshold | 0.35 (F1 peak 0.27 at 0.363) |
| Inference speed | about 7.7 ms per image (RTX 4050) |
| Severity thresholds | < 10% low · 10–20% moderate · > 20% high |
| Model version | `yolov8s_prototype-b0c0f155` |
