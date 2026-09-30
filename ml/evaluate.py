"""Evaluate a trained detector on the held-out test split.

Reports overall and per-class Precision, Recall, F1, mAP@50 and mAP@50-95, saves
Ultralytics plots (confusion matrix, PR / F1 / P / R curves) and exports
metrics.json + per_class.csv + summary.txt. By default the test split is also
evaluated per source dataset (e.g. clean product photos vs. real-world TACO litter).

By default Ultralytics' standard thresholds are used: mAP/PR curves at confidence
0.001 (the full precision-recall curve) and the confusion matrix at 0.25.
Do NOT evaluate mAP with the API's deployment threshold (e.g. 0.35-0.50) - that
truncates the precision-recall curve and under-reports mAP. Note that passing
--conf also moves the confusion-matrix threshold to that value.

Usage:
    python ml/evaluate.py
    python ml/evaluate.py --weights runs/detect/yolov8s_baseline/weights/best.pt --split val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from utils.classes import WASTE_CLASSES
from utils.device import resolve_device
from utils.paths import RUNS_DIR, resolve_path
from utils.run_info import dataset_info, environment_info, file_sha256, resolve_data_yaml
from utils.runtime import configure_dataloader
from utils.yolo_labels import label_dir_for, parse_label_file

DEFAULT_WEIGHTS = "models/best.pt"
DEFAULT_DATA = "datasets/waste/data.yaml"
PLOT_FILES = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, help="default: training image size from the model metadata, else 640")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=None,
                        help="override confidence threshold (default: 0.001 for mAP, 0.25 for confusion matrix)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--name", help="output folder under runs/evaluate/ (default: <model>_<split>)")
    parser.add_argument("--no-by-source", action="store_true", help="skip per-source evaluation")
    return parser.parse_args()


def load_model_metadata(weights: Path) -> dict:
    meta_path = weights.with_suffix(".json")  # written by ml/promote.py
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    run_config = weights.parent.parent / "run_config.json"  # a runs/detect/<name>/weights/*.pt file
    if run_config.exists():
        cfg = json.loads(run_config.read_text(encoding="utf-8"))
        return {"model_version": f"{weights.parent.parent.name}-{weights.stem}", "train_config": cfg.get("config")}
    return {"model_version": weights.stem}


def split_images(data_yaml: Path, split: str) -> list[Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    image_dir = Path(data["path"]) / data[split]
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.name != ".gitkeep")


def ground_truth_counts(images: list[Path]) -> dict[str, dict[str, int]]:
    """Images and instances per class, straight from the label files."""
    img_counts, box_counts = Counter(), Counter()
    for img in images:
        boxes, _ = parse_label_file(label_dir_for(img.parent) / f"{img.stem}.txt", len(WASTE_CLASSES))
        box_counts.update(WASTE_CLASSES[b.class_id] for b in boxes)
        img_counts.update({WASTE_CLASSES[b.class_id] for b in boxes})
    return {c: {"images": img_counts[c], "instances": box_counts[c]} for c in WASTE_CLASSES}


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def extract_metrics(metrics, images: list[Path]) -> dict:
    box = metrics.box
    gt = ground_truth_counts(images)
    per_class = {}
    for pos, cls_idx in enumerate(metrics.ap_class_index):
        name = WASTE_CLASSES[int(cls_idx)]
        p, r = float(box.p[pos]), float(box.r[pos])
        per_class[name] = {
            **gt[name],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1(p, r), 4),
            "map50": round(float(box.ap50[pos]), 4),
            "map50_95": round(float(box.ap[pos]), 4),
        }
    for name in WASTE_CLASSES:  # classes absent from this split have no metrics
        per_class.setdefault(name, {**gt[name], "precision": None, "recall": None, "f1": None,
                                    "map50": None, "map50_95": None})
    return {
        "images": len(images),
        "instances": sum(v["instances"] for v in gt.values()),
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "f1": round(f1(float(box.mp), float(box.mr)), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
        },
        "per_class": per_class,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in metrics.speed.items()},
    }


def run_val(model, data_path: Path, split: str, args: argparse.Namespace, imgsz: int, device: str, out: Path):
    """Run Ultralytics validation, saving plots directly into `out`."""
    overrides = {"conf": args.conf} if args.conf is not None else {}  # unset -> Ultralytics standard thresholds
    return model.val(
        data=str(data_path), split=split, imgsz=imgsz, batch=args.batch, device=device, workers=args.workers,
        iou=args.iou, plots=True, project=str(out.parent), name=out.name, exist_ok=True, verbose=False,
        **overrides,
    )


def format_table(title: str, result: dict) -> str:
    o = result["overall"]
    lines = [
        title,
        "-" * 64,
        f"Images: {result['images']}   Instances: {result['instances']}",
        "",
        f"Overall Precision: {o['precision']:.3f}",
        f"Overall Recall:    {o['recall']:.3f}",
        f"Overall F1:        {o['f1']:.3f}",
        f"mAP@50:            {o['map50']:.3f}",
        f"mAP@50-95:         {o['map50_95']:.3f}",
        "",
        "Per-Class Results",
        "",
        f"{'Class':<11}{'Images':>7}{'Inst.':>7}{'Precision':>11}{'Recall':>8}{'F1':>7}{'mAP50':>8}{'mAP50-95':>10}",
    ]
    fmt = lambda v, w: f"{v:>{w}.3f}" if v is not None else f"{'n/a':>{w}}"  # noqa: E731
    for name in WASTE_CLASSES:
        c = result["per_class"][name]
        lines.append(f"{name.capitalize():<11}{c['images']:>7}{c['instances']:>7}{fmt(c['precision'], 11)}"
                     f"{fmt(c['recall'], 8)}{fmt(c['f1'], 7)}{fmt(c['map50'], 8)}{fmt(c['map50_95'], 10)}")
    return "\n".join(lines)


def write_per_class_csv(path: Path, results: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subset", "class", "images", "instances", "precision", "recall", "f1", "map50", "map50_95"])
        for subset, result in results.items():
            o = result["overall"]
            writer.writerow([subset, "all", result["images"], result["instances"], o["precision"], o["recall"],
                             o["f1"], o["map50"], o["map50_95"]])
            for name in WASTE_CLASSES:
                c = result["per_class"][name]
                writer.writerow([subset, name, c["images"], c["instances"], c["precision"], c["recall"],
                                 c["f1"], c["map50"], c["map50_95"]])


def source_subsets(data_yaml: Path, split: str, images: list[Path]) -> dict[str, list[Path]]:
    """Group split images by source dataset using the committed manifest."""
    manifest = data_yaml.parent / "manifest.csv"
    if not manifest.exists():
        return {}
    source_of = {}
    with manifest.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == split:
                source_of[row["file"]] = row["source"]
    groups: dict[str, list[Path]] = defaultdict(list)
    for img in images:
        groups[source_of.get(img.name, "unknown")].append(img)
    return dict(groups) if len(groups) > 1 else {}


def main() -> int:
    args = parse_args()
    weights = resolve_path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}. Train first (python ml/train.py --promote) or pass --weights.")

    configure_dataloader()
    from ultralytics import YOLO

    meta = load_model_metadata(weights)
    imgsz = args.imgsz or (meta.get("train_config") or {}).get("imgsz", 640)
    device = resolve_device(args.device)
    name = args.name or f"{meta['model_version']}_{args.split}"
    out_dir = RUNS_DIR / "evaluate" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = resolve_path(args.data)
    resolved = resolve_data_yaml(data_yaml, out_dir / "data_resolved.yaml")
    images = split_images(resolved, args.split)

    model = YOLO(str(weights))
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != list(WASTE_CLASSES):
        raise SystemExit(f"Model classes {model_names} do not match canonical {list(WASTE_CLASSES)}")

    map_conf = args.conf if args.conf is not None else 0.001
    cm_conf = args.conf if args.conf is not None else 0.25
    print(f"Evaluating {weights} on '{args.split}' ({len(images)} images, imgsz={imgsz}, "
          f"mAP conf={map_conf}, confusion-matrix conf={cm_conf})")
    results = {"all": extract_metrics(run_val(model, resolved, args.split, args, imgsz, device, out_dir), images)}

    if not args.no_by_source:
        for source, subset in source_subsets(data_yaml, args.split, images).items():
            sub_dir = out_dir / f"source_{source}"
            sub_dir.mkdir(exist_ok=True)
            list_file = sub_dir / f"{args.split}_images.txt"
            list_file.write_text("\n".join(str(p) for p in subset) + "\n", encoding="utf-8")
            sub_yaml = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            sub_yaml[args.split] = str(list_file)
            sub_data = sub_dir / "data.yaml"
            sub_data.write_text(yaml.safe_dump(sub_yaml, sort_keys=False), encoding="utf-8")
            print(f"Evaluating subset '{source}' ({len(subset)} images)")
            results[source] = extract_metrics(
                run_val(model, sub_data, args.split, args, imgsz, device, sub_dir), subset
            )

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": meta.get("model_version"),
        "weights": weights.as_posix(),
        "weights_sha256": file_sha256(weights),
        "split": args.split,
        "params": {"imgsz": imgsz, "map_conf": map_conf, "confusion_matrix_conf": cm_conf, "iou": args.iou,
                   "batch": args.batch, "device": device},
        "dataset": dataset_info(data_yaml),
        "environment": environment_info(),
        "results": results,
        "plots": [p for p in PLOT_FILES if (out_dir / p).exists()],
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_per_class_csv(out_dir / "per_class.csv", results)

    title = f"Model Evaluation - {meta.get('model_version')} - {args.split} split"
    text = "\n\n".join(
        [format_table(title, results["all"])]
        + [format_table(f"Subset: {src} ({args.split})", res) for src, res in results.items() if src != "all"]
    )
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nSaved to {out_dir}:\n  metrics.json, per_class.csv, summary.txt, " + ", ".join(report["plots"]))
    return 0


if __name__ == "__main__":  # required on Windows: dataloader workers re-import this module
    sys.exit(main())
