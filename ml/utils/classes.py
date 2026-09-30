"""Expose the backend's canonical class list to the ML scripts (one source of truth)."""
from __future__ import annotations

import sys

from utils.paths import BACKEND_DIR

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.label_map import CLASS_TO_ID, WASTE_CLASSES, UnknownWasteClassError, canonical_name  # noqa: E402

__all__ = ["CLASS_TO_ID", "WASTE_CLASSES", "UnknownWasteClassError", "canonical_name"]
