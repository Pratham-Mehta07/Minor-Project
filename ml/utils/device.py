"""Device selection shared by training, evaluation and prediction."""
from __future__ import annotations

import torch

AUTO = "auto"


def resolve_device(requested: str | None = AUTO) -> str:
    """Return an Ultralytics device string.

    "auto" (or empty) picks the first CUDA GPU ("0") when available, else "cpu".
    Explicit values such as "cpu", "0" or "0,1" are validated and passed through.
    """
    value = (requested or AUTO).strip().lower()
    if value == AUTO:
        return "0" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{requested}' was requested but CUDA is not available. "
            "Use --device auto or --device cpu."
        )
    return value


def describe_device(device: str) -> str:
    """Human-readable description of a resolved device string."""
    if device == "cpu":
        return "CPU"
    first = int(device.split(",")[0])
    props = torch.cuda.get_device_properties(first)
    return f"CUDA:{device} ({props.name}, {props.total_memory / 1024**3:.1f} GB)"
