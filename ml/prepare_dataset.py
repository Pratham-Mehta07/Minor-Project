"""Convert raw source datasets into the canonical 6-class YOLO dataset.

Steps: read source annotations -> map labels explicitly (config class_map/drop)
-> COCO to YOLO boxes -> drop exact duplicate images -> stratified 70/20/10 split
(per source, so adding a source never reshuffles another) -> write images/labels
+ manifest.csv + dataset_report.json.

Usage:
    python ml/prepare_dataset.py                                   # default sources
    python ml/prepare_dataset.py --source ml/configs/sources/garbage_classification_3.yaml --overwrite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from utils.classes import CLASS_TO_ID, WASTE_CLASSES
from utils.coco import read_image_bytes
from utils.images import normalise_image
from utils.paths import PROJECT_ROOT, resolve_path
from utils.sources import SourceConfig, load_source, normalise_label
from utils.splitting import SPLITS, stratified_split
from utils.yolo_labels import YoloBox, from_xywh_pixels, write_label_file

DEFAULT_SOURCES = [
    "ml/configs/sources/garbage_classification_3.yaml",
    "ml/configs/sources/taco.yaml",
]
DEFAULT_OUT = "datasets/waste"
KEEP_FILES = {".gitkeep"}
WORKERS = 8


@dataclass
class Sample:
    source: SourceConfig
    container: Path
    member: str
    size: tuple[int, int]  # annotated (width, height)
    out_name: str  # file name in the output dataset
    boxes: list[YoloBox] = field(default_factory=list)
    sha256: str = ""

    @property
    def classes(self) -> set[str]:
        return {WASTE_CLASSES[b.class_id] for b in self.boxes}


@dataclass
class Stats:
    dropped_boxes: Counter = field(default_factory=Counter)  # by source label
    unknown_labels: Counter = field(default_factory=Counter)
    degenerate_boxes: int = 0
    duplicates_removed: int = 0


def output_name(source: SourceConfig, member: str) -> str:
    """Flatten the member path (TACO reuses file names across batch folders)."""
    path = PurePosixPath(member)
    stem = "_".join(path.with_suffix("").parts)
    suffix = ".jpg" if source.max_image_size else path.suffix.lower()
    return f"{source.prefix}_{stem}{suffix}"


def load_samples(source: SourceConfig, stats: Stats) -> list[Sample]:
    missing = [p for p in source.required_paths() if not p.exists()]
    if missing:
        hint = "Run: python ml/download_dataset.py" if source.files else f"Place the data in {source.root}"
        raise FileNotFoundError(f"Missing files for '{source.name}': {[str(m) for m in missing]}. {hint}")

    samples: list[Sample] = []
    for img in source.read_images():
        sample = Sample(source, img.container, img.member, (img.width, img.height), output_name(source, img.member))
        for box in img.boxes:
            label = normalise_label(box.label)
            if label in source.drop:
                stats.dropped_boxes[f"{source.name}:{box.label}"] += 1
                continue
            if label not in source.class_map:
                stats.unknown_labels[f"{source.name}:{box.label}"] += 1
                continue
            yolo = from_xywh_pixels(
                CLASS_TO_ID[source.class_map[label]], box.x, box.y, box.w, box.h, img.width, img.height
            )
            if yolo is None:
                stats.degenerate_boxes += 1
            else:
                sample.boxes.append(yolo)
        samples.append(sample)
    return samples


def hash_images(samples: list[Sample]) -> None:
    open_zips: dict[Path, zipfile.ZipFile] = {}
    try:
        for s in samples:
            s.sha256 = hashlib.sha256(read_image_bytes(s.container, s.member, open_zips)).hexdigest()
    finally:
        for zf in open_zips.values():
            zf.close()


def deduplicate(samples: list[Sample], stats: Stats) -> list[Sample]:
    """Keep one copy of byte-identical images so no image can leak across splits."""
    seen: set[str] = set()
    unique: list[Sample] = []
    for s in sorted(samples, key=lambda s: s.out_name):
        if s.sha256 in seen:
            stats.duplicates_removed += 1
            continue
        seen.add(s.sha256)
        unique.append(s)
    return unique


def split_per_source(samples: list[Sample], sources: list[SourceConfig], ratios: list[float], seed: int) -> list[str]:
    splits = [""] * len(samples)
    for source in sources:
        indices = [i for i, s in enumerate(samples) if s.source is source]
        for i, split in zip(indices, stratified_split([samples[i].classes for i in indices], ratios, seed)):
            splits[i] = split
    return splits


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    split_dirs = [out_dir / kind / split for kind in ("images", "labels") for split in SPLITS]
    existing = [f for d in split_dirs if d.exists() for f in d.iterdir() if f.name not in KEEP_FILES]
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already contains {len(existing)} files. Re-run with --overwrite to replace them.")
    for f in existing:
        f.unlink()
    for d in split_dirs:
        d.mkdir(parents=True, exist_ok=True)


def write_dataset(samples: list[Sample], splits: list[str], out_dir: Path) -> None:
    def write(sample: Sample, split: str, data: bytes) -> None:
        if sample.source.max_image_size:
            data = normalise_image(data, sample.size, sample.source.max_image_size)
        (out_dir / "images" / split / sample.out_name).write_bytes(data)
        write_label_file(out_dir / "labels" / split / (Path(sample.out_name).stem + ".txt"), sample.boxes)

    pairs = list(zip(samples, splits))
    open_zips: dict[Path, zipfile.ZipFile] = {}
    try:  # zip members are copied sequentially (shared handles are not thread-safe)
        for s, split in pairs:
            if not s.container.is_dir():
                write(s, split, read_image_bytes(s.container, s.member, open_zips))
    finally:
        for zf in open_zips.values():
            zf.close()

    folder_pairs = [(s, split) for s, split in pairs if s.container.is_dir()]
    with ThreadPoolExecutor(WORKERS) as pool:  # decoding/resizing large photos dominates
        futures = [pool.submit(write, s, split, (s.container / s.member).read_bytes()) for s, split in folder_pairs]
        for future in futures:
            future.result()  # re-raise any failure


def summarise(samples: list[Sample], splits: list[str]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for split in SPLITS:
        chosen = [s for s, sp in zip(samples, splits) if sp == split]
        boxes = Counter(WASTE_CLASSES[b.class_id] for s in chosen for b in s.boxes)
        images = Counter(c for s in chosen for c in s.classes)
        summary[split] = {
            "images": len(chosen),
            "background_images": sum(1 for s in chosen if not s.boxes),
            "boxes": sum(boxes.values()),
            "images_per_class": {c: images[c] for c in WASTE_CLASSES},
            "boxes_per_class": {c: boxes[c] for c in WASTE_CLASSES},
        }
    return summary


def write_manifest(samples: list[Sample], splits: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "split", "source", "source_file", "sha256", "boxes", *WASTE_CLASSES])
        rows = sorted(zip(samples, splits), key=lambda p: (SPLITS.index(p[1]), p[0].out_name))
        for s, split in rows:
            counts = Counter(b.class_id for b in s.boxes)
            source_file = f"{s.container.name}/{s.member}"
            writer.writerow([s.out_name, split, s.source.name, source_file, s.sha256, len(s.boxes),
                             *(counts[i] for i in range(len(WASTE_CLASSES)))])


def print_summary(title: str, summary: dict[str, dict]) -> None:
    header = f"{'class':<11}" + "".join(f"{s + ' img/box':>18}" for s in SPLITS)
    print(f"\n{title}\n{header}\n" + "-" * len(header))
    for c in WASTE_CLASSES:
        cells = "".join(f"{summary[s]['images_per_class'][c]:>9}/{summary[s]['boxes_per_class'][c]:<8}" for s in SPLITS)
        print(f"{c:<11}{cells}")
    print("-" * len(header))
    print(f"{'images':<11}" + "".join(f"{summary[s]['images']:>18}" for s in SPLITS))
    print(f"{'background':<11}" + "".join(f"{summary[s]['background_images']:>18}" for s in SPLITS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", help="source config YAML (repeatable; default: all known sources)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output dataset directory")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing prepared dataset")
    args = parser.parse_args()

    sources = [load_source(p) for p in (args.source or DEFAULT_SOURCES)]
    out_dir = resolve_path(args.out)
    stats = Stats()

    samples: list[Sample] = []
    for source in sources:
        print(f"Reading '{source.name}' ...")
        samples.extend(load_samples(source, stats))
    if stats.unknown_labels:
        raise SystemExit(
            f"Unmapped source labels {dict(stats.unknown_labels)}. Add each one to class_map or drop "
            "in the source config - labels are never silently remapped."
        )

    names = Counter(s.out_name for s in samples)
    clashes = [n for n, c in names.items() if c > 1]
    if clashes:
        raise SystemExit(f"Output file name collisions: {clashes[:10]}")

    print(f"Hashing {len(samples)} images ...")
    hash_images(samples)
    samples = deduplicate(samples, stats)

    splits = split_per_source(samples, sources, args.ratios, args.seed)
    prepare_output(out_dir, args.overwrite)
    print(f"Writing dataset to {out_dir} ...")
    write_dataset(samples, splits, out_dir)

    summary = summarise(samples, splits)
    by_source = {}
    for source in sources:
        keep = [(s, sp) for s, sp in zip(samples, splits) if s.source is source]
        by_source[source.name] = summarise([s for s, _ in keep], [sp for _, sp in keep])

    write_manifest(samples, splits, out_dir / "manifest.csv")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": list(WASTE_CLASSES),
        "seed": args.seed,
        "ratios": dict(zip(SPLITS, args.ratios)),
        "sources": [
            {"name": s.name, "config": s.config_path.relative_to(PROJECT_ROOT).as_posix(),
             "license": s.license, "attribution": s.attribution, "max_image_size": s.max_image_size,
             "class_map": s.class_map, "drop": sorted(s.drop)}
            for s in sources
        ],
        "total_images": len(samples),
        "duplicates_removed": stats.duplicates_removed,
        "degenerate_boxes_removed": stats.degenerate_boxes,
        "dropped_boxes_by_label": dict(sorted(stats.dropped_boxes.items())),
        "splits": summary,
        "splits_by_source": by_source,
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, source_summary in by_source.items():
        print_summary(f"Source: {name}", source_summary)
    print_summary("Combined dataset", summary)
    print(f"\nDuplicates removed: {stats.duplicates_removed}   degenerate boxes removed: {stats.degenerate_boxes}"
          f"   dropped boxes: {sum(stats.dropped_boxes.values())}")
    print(f"Manifest: {out_dir / 'manifest.csv'}\nReport:   {out_dir / 'dataset_report.json'}")
    print("Next: python ml/verify_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
