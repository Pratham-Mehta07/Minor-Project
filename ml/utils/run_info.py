"""Reproducibility metadata recorded alongside training/evaluation outputs."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
from pathlib import Path

import torch
import yaml

from utils.paths import PROJECT_ROOT


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "ultralytics": metadata.version("ultralytics"),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def resolve_data_yaml(data_yaml: Path, out_path: Path) -> Path:
    """Write a copy of data.yaml with an absolute `path`, so Ultralytics never guesses the root."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"]) if data.get("path") else data_yaml.parent
    data["path"] = str(root if root.is_absolute() else (data_yaml.parent / root).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out_path


def dataset_info(data_yaml: Path) -> dict:
    """Fingerprint the prepared dataset via its manifest (the frozen split)."""
    root = data_yaml.parent
    info: dict = {"data_yaml": data_yaml.as_posix()}
    manifest = root / "manifest.csv"
    if manifest.exists():
        info["manifest_sha256"] = file_sha256(manifest)
    report = root / "dataset_report.json"
    if report.exists():
        rep = json.loads(report.read_text(encoding="utf-8"))
        info["sources"] = [s["name"] for s in rep.get("sources", [])]
        info["images"] = {k: v["images"] for k, v in rep.get("splits", {}).items()}
    return info
