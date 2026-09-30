"""Loading and validation of source-dataset configs (ml/configs/sources/*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from utils.classes import canonical_name
from utils.coco import CocoImage, read_coco_archive, read_coco_dir
from utils.paths import DATASETS_DIR, resolve_path

SUPPORTED_FORMATS = {"coco_zip", "coco_dir"}
RAW_DIR = DATASETS_DIR / "raw"


@dataclass(frozen=True)
class DownloadFile:
    path: str
    sha256: str | None = None


@dataclass(frozen=True)
class SourceConfig:
    name: str
    prefix: str
    format: str
    license: str
    attribution: str
    base_url: str
    files: tuple[DownloadFile, ...]
    archives: tuple[str, ...]  # coco_zip: archives relative to raw_dir
    root: Path | None  # coco_dir: folder holding images + annotation file
    annotation_file: str | None  # coco_dir: relative to root
    max_image_size: int | None  # longest side after preparation; None = copy bytes unchanged
    class_map: dict[str, str]  # normalised source label -> canonical class
    drop: frozenset[str]  # normalised source labels deliberately discarded
    config_path: Path

    @property
    def raw_dir(self) -> Path:
        return RAW_DIR / self.name

    def required_paths(self) -> list[Path]:
        if self.format == "coco_dir":
            return [self.root / self.annotation_file]
        return [self.raw_dir / a for a in self.archives]

    def read_images(self) -> list[CocoImage]:
        if self.format == "coco_dir":
            return read_coco_dir(self.root, self.annotation_file)
        return [img for archive in self.required_paths() for img in read_coco_archive(archive)]


def normalise_label(label: str) -> str:
    return label.strip().lower()


def load_source(path: str | Path) -> SourceConfig:
    config_path = resolve_path(path)
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    fmt = raw["format"]
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"{config_path.name}: unsupported format '{fmt}' (supported: {sorted(SUPPORTED_FORMATS)})")
    if fmt == "coco_dir" and not (raw.get("root") and raw.get("annotation_file")):
        raise ValueError(f"{config_path.name}: coco_dir sources need 'root' and 'annotation_file'")

    # Validate targets up front so a typo in the mapping fails before any work is done.
    class_map = {normalise_label(src): canonical_name(dst) for src, dst in raw["class_map"].items()}
    drop = frozenset(normalise_label(x) for x in raw.get("drop") or [])
    overlap = drop & class_map.keys()
    if overlap:
        raise ValueError(f"{config_path.name}: labels both mapped and dropped: {sorted(overlap)}")

    download = raw.get("download") or {}
    return SourceConfig(
        name=raw["name"],
        prefix=raw["prefix"],
        format=fmt,
        license=raw.get("license", "unknown"),
        attribution=" ".join(str(raw.get("attribution", "")).split()),
        base_url=download.get("base_url", "").rstrip("/"),
        files=tuple(DownloadFile(f["path"], f.get("sha256")) for f in download.get("files", [])),
        archives=tuple(raw.get("archives") or []),
        root=resolve_path(raw["root"]) if raw.get("root") else None,
        annotation_file=raw.get("annotation_file"),
        max_image_size=raw.get("max_image_size"),
        class_map=class_map,
        drop=drop,
        config_path=config_path,
    )
