"""Independently validate a YOLO dataset before training.

Checks: class names match the canonical list, every image has a label file,
label syntax/class ids/coordinates are valid, and no image appears in two splits.

Usage:
    python ml/verify_dataset.py
    python ml/verify_dataset.py --data datasets/waste/data.yaml --check-images
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from utils.classes import WASTE_CLASSES
from utils.paths import resolve_path
from utils.splitting import SPLITS
from utils.yolo_labels import IMAGE_SUFFIXES, label_dir_for, parse_label_file

DEFAULT_DATA = "datasets/waste/data.yaml"
MAX_ERRORS_SHOWN = 20


def load_names(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names or [])


def dataset_root(data_yaml: Path, data: dict) -> Path:
    root = Path(data["path"]) if data.get("path") else data_yaml.parent
    return root if root.is_absolute() else (data_yaml.parent / root).resolve()


def check_image(path: Path) -> str | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as exc:  # PIL raises many exception types for corrupt files
        return f"{path.name}: unreadable image ({exc})"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA, help="dataset YAML")
    parser.add_argument("--check-images", action="store_true", help="also decode every image (slower)")
    args = parser.parse_args()

    data_yaml = resolve_path(args.data)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    names = load_names(data)
    if names != list(WASTE_CLASSES):
        errors.append(f"data.yaml names {names} != canonical {list(WASTE_CLASSES)}")
    if "nc" in data and data["nc"] != len(WASTE_CLASSES):
        errors.append(f"data.yaml nc={data['nc']} but there are {len(WASTE_CLASSES)} classes")

    root = dataset_root(data_yaml, data)
    hashes: dict[str, set[str]] = defaultdict(set)
    summary: dict[str, dict] = {}

    for split in SPLITS:
        if split not in data:
            errors.append(f"data.yaml has no '{split}' entry")
            continue
        image_dir = root / data[split]
        label_dir = label_dir_for(image_dir)
        images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if image_dir.exists() else []
        if not images:
            errors.append(f"{split}: no images in {image_dir}")
            continue

        boxes, img_per_class = Counter(), Counter()
        background = 0
        for img in images:
            hashes[hashlib.sha256(img.read_bytes()).hexdigest()].add(split)
            label = label_dir / f"{img.stem}.txt"
            if not label.exists():
                errors.append(f"{split}/{img.name}: missing label file")
                continue
            parsed, label_errors = parse_label_file(label, len(WASTE_CLASSES))
            errors.extend(f"{split}/{e}" for e in label_errors)
            if not parsed and not label_errors:
                background += 1
            boxes.update(WASTE_CLASSES[b.class_id] for b in parsed)
            img_per_class.update({WASTE_CLASSES[b.class_id] for b in parsed})
            if args.check_images and (problem := check_image(img)):
                errors.append(f"{split}/{problem}")

        image_stems = {p.stem for p in images}
        orphans = [p.name for p in label_dir.glob("*.txt") if p.stem not in image_stems]
        if orphans:
            warnings.append(f"{split}: {len(orphans)} label files without an image (e.g. {orphans[0]})")

        summary[split] = {"images": len(images), "background": background, "boxes": boxes, "img": img_per_class}

    leaked = sum(1 for splits in hashes.values() if len(splits) > 1)
    if leaked:
        errors.append(f"{leaked} identical images appear in more than one split (data leakage)")

    print(f"Dataset: {root}\n")
    print(f"{'class':<11}" + "".join(f"{s + ' images':>14}{s + ' boxes':>13}" for s in summary))
    for c in WASTE_CLASSES:
        print(f"{c:<11}" + "".join(f"{v['img'][c]:>14}{v['boxes'][c]:>13}" for v in summary.values()))
    print(f"{'TOTAL':<11}" + "".join(f"{v['images']:>14}{sum(v['boxes'].values()):>13}" for v in summary.values()))
    print(f"{'background':<11}" + "".join(f"{v['background']:>14}{'':>13}" for v in summary.values()))

    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors[:MAX_ERRORS_SHOWN]:
            print(f"  - {e}")
        if len(errors) > MAX_ERRORS_SHOWN:
            print(f"  ... and {len(errors) - MAX_ERRORS_SHOWN} more")
        print("\nDataset verification FAILED")
        return 1

    print("\nDataset verification PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
