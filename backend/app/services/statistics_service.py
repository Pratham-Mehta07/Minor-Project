"""Deterministic object-count statistics (no ML, no I/O).

Quantity in this project means *number of detected objects*; percentages are
count-based, never weight-based.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.ml.label_map import WASTE_CLASSES, canonical_name

PERCENT_DECIMALS = 2


def count_by_class(labels: Iterable[str]) -> dict[str, int]:
    """Count detections per canonical class; every class is present (0 if absent).

    Raises UnknownWasteClassError for labels outside the six canonical classes.
    """
    counts = dict.fromkeys(WASTE_CLASSES, 0)
    for label in labels:
        counts[canonical_name(label)] += 1
    return counts


def percent(part: int, total: int) -> float:
    """part / total * 100, rounded; 0.0 when total is 0 (no division by zero)."""
    return round(part / total * 100, PERCENT_DECIMALS) if total else 0.0


def composition_percent(counts: Mapping[str, int]) -> dict[str, float]:
    """Share of each class in the total object count.

    Values are rounded individually, so they may not sum to exactly 100.
    """
    total = sum(counts.get(c, 0) for c in WASTE_CLASSES)
    return {c: percent(counts.get(c, 0), total) for c in WASTE_CLASSES}
