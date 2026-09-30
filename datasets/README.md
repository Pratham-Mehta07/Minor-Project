# datasets/

Images and labels are **not committed**. They are rebuilt from the source data by
scripts. Only `waste/data.yaml`, `waste/manifest.csv` (the frozen split) and
`waste/dataset_report.json` are tracked in git.

```
data/                         TACO as downloaded (batch_*/ + annotations.json), project root
datasets/
├── raw/<source-name>/        downloaded source archives (untouched)
└── waste/                    prepared YOLO dataset used for training
    ├── data.yaml
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/      one .txt per image (empty = no objects)
    ├── manifest.csv                  file -> split, source, source file, sha256, per-class box counts
    └── dataset_report.json           sources, mappings, seed, per-split and per-source statistics
```

## Rebuild

```bash
python ml/download_dataset.py            # GARBAGE CLASSIFICATION 3, ~196 MB, SHA-256 verified
# TACO: place batch_1..batch_15 and annotations.json in data/ (https://github.com/pedropro/TACO)
python ml/prepare_dataset.py             # add --overwrite to rebuild
python ml/verify_dataset.py              # add --check-images to decode every image
```

`prepare_dataset.py` is deterministic. Each source is split on its own with the same
`--seed`, so adding a source never changes another source's split. You can compare
against the committed `manifest.csv`.

## YOLO label format

One line per object: `class_id x_center y_center width height`, where the coordinates
are fractions of the image width and height (0-1). Class ids:

| id | class |
|----|-------|
| 0 | plastic |
| 1 | paper |
| 2 | glass |
| 3 | metal |
| 4 | cardboard |
| 5 | organic |

## Sources and attribution

### GARBAGE CLASSIFICATION 3 (v2): `gc3_*` files
"GARBAGE CLASSIFICATION 3" (v2) by Material Identification, Roboflow Universe,
<https://universe.roboflow.com/material-identification/garbage-classification-3>,
licensed under **CC BY 4.0**. Downloaded from the Hugging Face mirror
<https://huggingface.co/datasets/keremberke/garbage-object-detection> at a pinned revision.

- Class mapping: `biodegradable → organic`. The other five names already match.
- The published splits are skewed by class (e.g. the original test split has no
  glass), so all 10,464 images are pooled and re-split 70/20/10, stratified by the
  rarest class in each image.
- Known characteristics (limitations to report):
  - Images were resized by Roboflow to 416×416 with stretching. Training images
    carry random flips and 90° rotations.
  - Mostly clean web/product photos. "Organic" is largely fresh food, not
    decomposing bin waste.
  - Bottle caps are often labelled as separate `metal` objects.

### TACO: `taco_*` files
Pedro F. Proença and Pedro Simões, "TACO: Trash Annotations in Context for Litter
Detection", arXiv:2003.06975, 2020. <https://github.com/pedropro/TACO>.
Images originate from Flickr under their original licences.

- 1,500 real-world litter photos in 60 categories, mapped **by material** in
  [ml/configs/sources/taco.yaml](../ml/configs/sources/taco.yaml).
- Explicitly dropped: cigarettes, unlabelled litter, composites (Tetra Pak drink
  cartons, blister packs, squeezable tubes), tissues, textiles, shoes and batteries.
  Those objects remain in the images as unlabelled background.
- Photos are EXIF-rotated (the annotations use the rotated frame) and downscaled to
  ≤1280 px on the long side.
- Only 8 `Food waste` annotations exist, so TACO contributes almost nothing to `organic`.

To add another source, create `ml/configs/sources/<name>.yaml` with an explicit
`class_map`/`drop`, add it to `DEFAULT_SOURCES` in `prepare_dataset.py` or pass it
with `--source`.
