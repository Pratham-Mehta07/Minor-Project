"""Verify the Python environment for WasteWise AI.

Usage:
    python ml/check_env.py                 # full check incl. a tiny YOLOv8n inference
    python ml/check_env.py --skip-inference

The inference smoke test downloads the official pretrained yolov8n.pt (~6 MB)
into models/pretrained/ and runs it on an image bundled with Ultralytics.
It only proves the stack works; it is not our waste model.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import platform
import sys
import time

from utils.paths import PRETRAINED_DIR

MIN_PYTHON = (3, 10)

# (import name, distribution name) - backend packages are optional for ML-only
# environments such as Colab/Kaggle, so they only produce warnings.
BACKEND_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic_settings", "pydantic-settings"),
    ("multipart", "python-multipart"),
    ("sqlalchemy", "sqlalchemy"),
    ("alembic", "alembic"),
    ("psycopg", "psycopg"),
    ("pytest", "pytest"),
    ("httpx", "httpx"),
]


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def check_core() -> bool:
    ok = True
    print(f"Python        : {platform.python_version()} ({sys.executable})")
    print(f"Platform      : {platform.platform()}")
    if sys.version_info < MIN_PYTHON:
        print(f"  ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required.")
        ok = False

    try:
        import torch
    except ImportError:
        print("torch         : NOT INSTALLED - install PyTorch first (see README).")
        return False

    print(f"torch         : {torch.__version__}")
    print(f"CUDA build    : {torch.version.cuda or 'none (CPU-only build)'}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    ultra = _version("ultralytics")
    print(f"ultralytics   : {ultra}")
    if ultra == "not installed":
        ok = False

    from utils.device import describe_device, resolve_device

    device = resolve_device("auto")
    print(f"Auto device   : {describe_device(device)}")
    if device == "cpu" and torch.version.cuda is None:
        print("  NOTE: CPU-only PyTorch build. If this machine has an NVIDIA GPU,")
        print("        reinstall PyTorch using the CUDA command in the README.")
    return ok


def check_backend() -> None:
    print("\nBackend packages (optional for ML-only environments):")
    for module, dist in BACKEND_PACKAGES:
        found = importlib.util.find_spec(module) is not None
        status = _version(dist) if found else "MISSING"
        print(f"  {dist:<18} {status}")


def smoke_inference() -> bool:
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    from utils.device import resolve_device

    weights = PRETRAINED_DIR / "yolov8n.pt"
    image = ASSETS / "bus.jpg"
    device = resolve_device("auto")
    print(f"\nSmoke test: {weights.name} on {image.name} (device={device})")

    model = YOLO(str(weights))
    start = time.perf_counter()
    result = model.predict(str(image), device=device, verbose=False)[0]
    elapsed = time.perf_counter() - start

    names = [result.names[int(c)] for c in result.boxes.cls.tolist()]
    print(f"  {len(names)} objects in {elapsed:.2f}s (includes warm-up): {', '.join(names)}")
    return len(names) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-inference", action="store_true", help="skip the YOLOv8n smoke test")
    args = parser.parse_args()

    ok = check_core()
    check_backend()
    if ok and not args.skip_inference:
        ok = smoke_inference()

    print("\nEnvironment OK" if ok else "\nEnvironment check FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
