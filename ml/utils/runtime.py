"""Process-level settings that must be applied before Ultralytics is imported."""
from __future__ import annotations

import os
import platform


def configure_dataloader() -> None:
    """Disable pinned memory on Windows (override with PIN_MEMORY=True).

    Pinned-memory transfer crashed with "CUDA error: resource already mapped" on
    Windows under RAM pressure; its speed benefit is negligible for this workload.
    Ultralytics reads PIN_MEMORY at import time, so call this before importing it.
    """
    if platform.system() == "Windows":
        os.environ.setdefault("PIN_MEMORY", "False")
