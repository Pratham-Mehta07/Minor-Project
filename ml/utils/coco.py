"""Reading COCO-format detection datasets, either zipped (Roboflow export) or as a folder."""
from __future__ import annotations

import json
import posixpath
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CocoBox:
    label: str
    x: float  # top-left, absolute pixels
    y: float
    w: float
    h: float


@dataclass
class CocoImage:
    container: Path  # zip archive or dataset root directory
    member: str  # posix path of the image inside the container
    width: int
    height: int
    boxes: list[CocoBox] = field(default_factory=list)


def _parse(data: dict, container: Path, base: str, members: set[str] | None) -> list[CocoImage]:
    categories = {c["id"]: c["name"] for c in data["categories"]}
    images: dict[int, CocoImage] = {}
    for img in data["images"]:
        member = posixpath.join(base, img["file_name"]) if base else img["file_name"]
        exists = member in members if members is not None else (container / member).is_file()
        if not exists:
            raise FileNotFoundError(f"{container}: annotation references missing image '{member}'")
        images[img["id"]] = CocoImage(container, member, int(img["width"]), int(img["height"]))

    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        images[ann["image_id"]].boxes.append(CocoBox(categories[ann["category_id"]], x, y, w, h))
    return list(images.values())


def read_coco_archive(archive: Path) -> list[CocoImage]:
    """Parse the single COCO annotation file inside a zip archive."""
    with zipfile.ZipFile(archive) as zf:
        json_members = [m for m in zf.namelist() if m.endswith(".json")]
        if len(json_members) != 1:
            raise ValueError(f"{archive}: expected exactly one COCO .json file, found {json_members}")
        data = json.loads(zf.read(json_members[0]))
        members = set(zf.namelist())
    return _parse(data, archive, posixpath.dirname(json_members[0]), members)


def read_coco_dir(root: Path, annotation_file: str) -> list[CocoImage]:
    """Parse a folder dataset whose image paths are relative to `root`."""
    data = json.loads((root / annotation_file).read_text(encoding="utf-8"))
    return _parse(data, root, "", None)


def read_image_bytes(container: Path, member: str, open_zips: dict[Path, zipfile.ZipFile]) -> bytes:
    """Read raw image bytes from a zip (cached handle) or a directory."""
    if container.is_dir():
        return (container / member).read_bytes()
    zf = open_zips.get(container)
    if zf is None:
        zf = open_zips[container] = zipfile.ZipFile(container)
    return zf.read(member)
