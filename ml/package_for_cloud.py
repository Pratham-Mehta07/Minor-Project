"""Bundle the prepared dataset + ML code into one zip for training on Kaggle/Colab.

Usage:
    python ml/package_for_cloud.py            # -> dist/wastewise_cloud.zip

The archive contains a single top-level folder `wastewise/` that mirrors this
project's layout, so every script works unchanged after extraction.
Upload it as a Kaggle Dataset (or to Google Drive for Colab) and run
notebooks/train_cloud.ipynb.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from utils.paths import PROJECT_ROOT, resolve_path

TOP = "wastewise"
CODE_GLOBS = [
    "ml/*.py",
    "ml/utils/*.py",
    "ml/configs/**/*.yaml",
    "ml/requirements.txt",
    "backend/app/__init__.py",
    "backend/app/ml/__init__.py",
    "backend/app/ml/label_map.py",  # canonical classes, imported by ml/utils/classes.py
    "notebooks/train_cloud.ipynb",
    "README.md",
]
DATASET_FILES = ["data.yaml", "manifest.csv", "dataset_report.json"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # already compressed -> stored as-is


def collect(dataset_dir: Path) -> list[Path]:
    files = {p for pattern in CODE_GLOBS for p in PROJECT_ROOT.glob(pattern) if p.is_file()}
    files.update(dataset_dir / name for name in DATASET_FILES if (dataset_dir / name).exists())
    for sub in ("images", "labels"):
        files.update(p for p in (dataset_dir / sub).rglob("*") if p.is_file() and p.suffix != ".cache")
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="datasets/waste", help="prepared dataset directory")
    parser.add_argument("--out", default="dist/wastewise_cloud.zip")
    args = parser.parse_args()

    dataset_dir = resolve_path(args.dataset)
    if not (dataset_dir / "data.yaml").exists():
        raise SystemExit(f"No prepared dataset in {dataset_dir}. Run ml/prepare_dataset.py first.")

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = collect(dataset_dir)
    images = sum(1 for f in files if f.suffix.lower() in IMAGE_SUFFIXES)

    with zipfile.ZipFile(out, "w") as zf:
        for path in files:
            arcname = f"{TOP}/{path.relative_to(PROJECT_ROOT).as_posix()}"
            compress = zipfile.ZIP_STORED if path.suffix.lower() in IMAGE_SUFFIXES else zipfile.ZIP_DEFLATED
            zf.write(path, arcname, compress_type=compress)

    print(f"Wrote {out} ({out.stat().st_size / 1e9:.2f} GB): {len(files)} files, {images} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
