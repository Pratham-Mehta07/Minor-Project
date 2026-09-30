"""Canonical WasteWise waste classes - the single source of truth.

The order defines the YOLO class ids and must never change without retraining:
datasets/waste/data.yaml, the trained model's `names` and this tuple must match.

This module is dependency-free so the ML scripts can import it too.
"""
from __future__ import annotations

WASTE_CLASSES: tuple[str, ...] = (
    "plastic",
    "paper",
    "glass",
    "metal",
    "cardboard",
    "organic",
)

CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(WASTE_CLASSES)}


class UnknownWasteClassError(ValueError):
    """Raised when a label is not one of the canonical waste classes."""


def canonical_name(label: str) -> str:
    """Normalise a label (case/whitespace) and ensure it is a canonical class."""
    name = label.strip().lower()
    if name not in CLASS_TO_ID:
        raise UnknownWasteClassError(f"Unknown waste class '{label}'. Expected one of {WASTE_CLASSES}.")
    return name


def class_id(label: str) -> int:
    return CLASS_TO_ID[canonical_name(label)]
