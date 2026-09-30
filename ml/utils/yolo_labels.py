"""YOLO detection label format: one `class xc yc w h` line per box, normalised to [0, 1]."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_BOX_PIXELS = 1.0


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def from_xywh_pixels(class_id: int, x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> YoloBox | None:
    """Convert a top-left pixel box to YOLO format, clipping to the image.

    Returns None for boxes that are degenerate after clipping.
    """
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
    if x2 - x1 < MIN_BOX_PIXELS or y2 - y1 < MIN_BOX_PIXELS:
        return None
    return YoloBox(
        class_id,
        (x1 + x2) / 2 / img_w,
        (y1 + y2) / 2 / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    )


def write_label_file(path: Path, boxes: list[YoloBox]) -> None:
    text = "\n".join(b.to_line() for b in boxes)
    path.write_text(text + "\n" if text else "", encoding="utf-8")


def parse_label_file(path: Path, num_classes: int) -> tuple[list[YoloBox], list[str]]:
    """Parse and validate a label file. Returns (boxes, error messages)."""
    boxes: list[YoloBox] = []
    errors: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{where}: expected 5 values, got {len(parts)} (segmentation labels are not supported)")
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{where}: non-numeric value in '{line.strip()}'")
            continue
        if not 0 <= cls < num_classes:
            errors.append(f"{where}: class id {cls} outside 0..{num_classes - 1}")
        elif not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
            errors.append(f"{where}: coordinates must be in [0, 1] with positive width/height")
        else:
            boxes.append(YoloBox(cls, xc, yc, w, h))
    return boxes, errors


def label_dir_for(image_dir: Path) -> Path:
    """Ultralytics convention: replace the last 'images' path component with 'labels'."""
    parts = list(image_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    raise ValueError(f"Image directory '{image_dir}' has no 'images' component")
