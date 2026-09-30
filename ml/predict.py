"""Run the trained waste detector on one image or a folder of images.

Prints detections, object counts, count-based composition, recyclable % and
object-count contamination %, and saves an annotated image + JSON per image.

Usage:
    python ml/predict.py --image path/to/photo.jpg
    python ml/predict.py --image path/to/folder --show        # pop-up window per image
    python ml/predict.py --image photo.jpg --conf 0.25 --weights runs/detect/<run>/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import utils.classes  # noqa: F401  (puts backend/ on sys.path for the shared statistics services)
from app.services.contamination_service import calculate_contamination
from app.services.statistics_service import composition_percent, count_by_class
from utils.classes import WASTE_CLASSES
from utils.device import resolve_device
from utils.paths import RUNS_DIR, resolve_path
from utils.yolo_labels import IMAGE_SUFFIXES

DEFAULT_WEIGHTS = "models/best.pt"
# Mean F1 on the test set peaks at confidence ~0.36 for the current model (BoxF1_curve.png).
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.7  # Ultralytics default NMS IoU, same as used in evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="image file or folder of images")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="minimum detection confidence")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, help="default: training image size from models/best.json, else 640")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", help="output folder (default: runs/predict/<timestamp>)")
    parser.add_argument("--show", action="store_true", help="display each annotated image (press any key)")
    return parser.parse_args()


def model_info(weights: Path) -> tuple[str, int | None]:
    meta_path = weights.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("model_version", weights.stem), (meta.get("train_config") or {}).get("imgsz")
    return weights.stem, None


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    raise SystemExit(f"Image or folder not found: {source}")


def analyse(result, conf: float, model_version: str, image: Path) -> dict:
    """Turn one Ultralytics result into the same structure the API will return."""
    detections = []
    for cls_id, score, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
        detections.append({
            "class": result.names[int(cls_id)],
            "confidence": round(float(score), 4),
            "bbox": [round(v) for v in box],  # x1, y1, x2, y2 in original image pixels
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    counts = count_by_class(d["class"] for d in detections)
    contamination = calculate_contamination(counts)
    return {
        "image": str(image),
        "model_version": model_version,
        "confidence_threshold": conf,
        "total_objects": contamination.total_objects,
        "counts": counts,
        "composition_percent": composition_percent(counts),
        "recyclable_percent": contamination.recyclable_percent,
        "contamination_percent": contamination.contamination_percent,
        "contamination_severity": contamination.severity.value,
        "detections": detections,
    }


def print_report(report: dict) -> None:
    print(f"\n=== {Path(report['image']).name} ===")
    if not report["detections"]:
        print("No waste objects detected above the confidence threshold "
              f"({report['confidence_threshold']}). No composition could be calculated.")
        return
    print(f"{'#':>3}  {'class':<10} {'conf':>6}  bbox [x1, y1, x2, y2]")
    for i, d in enumerate(report["detections"], 1):
        print(f"{i:>3}  {d['class']:<10} {d['confidence']:>6.2f}  {d['bbox']}")
    print(f"\nTotal objects detected: {report['total_objects']}\n")
    for c in WASTE_CLASSES:
        print(f"  {c.capitalize() + ':':<11} {report['counts'][c]:>3}   ({report['composition_percent'][c]:.2f}%)")
    print(f"\nRecyclable objects:            {report['recyclable_percent']:.2f}%")
    print(f"Contamination (object count):  {report['contamination_percent']:.2f}%  "
          f"severity={report['contamination_severity']}")


def annotate(result, report: dict):
    """Ultralytics box drawing + a summary banner added above the image (BGR numpy image)."""
    import cv2

    boxes = result.plot(line_width=2)
    present = [f"{c} {report['counts'][c]}" for c in WASTE_CLASSES if report["counts"][c]]
    lines = [
        f"Objects: {report['total_objects']}   " + ("  ".join(present) if present else "none detected"),
        f"Recyclable: {report['recyclable_percent']:.1f}%   "
        f"Contamination (count): {report['contamination_percent']:.1f}%",
    ]
    scale = max(0.45, boxes.shape[1] / 1400)
    thickness = max(1, round(scale * 2))
    line_h = int(28 * scale)
    # Extend the canvas instead of drawing over the image, so no box or label is hidden.
    image = cv2.copyMakeBorder(boxes, line_h * len(lines) + 10, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for i, text in enumerate(lines):
        cv2.putText(image, text, (8, line_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def main() -> int:
    args = parse_args()
    weights = resolve_path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}. Train and promote a model first.")

    from ultralytics import YOLO
    import cv2

    model_version, trained_imgsz = model_info(weights)
    imgsz = args.imgsz or trained_imgsz or 640
    device = resolve_device(args.device)
    images = collect_images(resolve_path(args.image))
    out_dir = resolve_path(args.out) if args.out else RUNS_DIR / "predict" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))  # loaded once for all images
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != list(WASTE_CLASSES):
        raise SystemExit(f"Model classes {model_names} do not match canonical {list(WASTE_CLASSES)}")
    print(f"Model {model_version} | conf>={args.conf} | imgsz={imgsz} | device={device} | {len(images)} image(s)")

    for image in images:
        result = model.predict(str(image), conf=args.conf, iou=args.iou, imgsz=imgsz, device=device, verbose=False)[0]
        report = analyse(result, args.conf, model_version, image)
        print_report(report)

        annotated = annotate(result, report)
        cv2.imwrite(str(out_dir / f"{image.stem}_annotated.jpg"), annotated)
        (out_dir / f"{image.stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.show:
            cv2.imshow("WasteWise AI - press any key", annotated)
            cv2.waitKey(0)
    if args.show:
        cv2.destroyAllWindows()

    print(f"\nAnnotated images and JSON saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
