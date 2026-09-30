"""Deterministic, object-count-based recycling contamination rules.

contamination_percent = contaminating objects / total detected objects * 100.
This is an OBJECT-COUNT percentage, not a mass-based contamination rate.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from app.ml.label_map import WASTE_CLASSES
from app.services.statistics_service import percent

# Classes accepted in the recycling / dry-waste stream; anything else contaminates it.
RECYCLING_STREAM_ALLOWED: frozenset[str] = frozenset({"plastic", "paper", "glass", "metal", "cardboard"})


class Severity(str, Enum):
    NONE = "none"  # nothing detected, no reliable composition
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True)
class SeverityThresholds:
    """Low < moderate <= x <= high < High (percent). Defaults follow the project spec."""

    moderate: float = 10.0
    high: float = 20.0


@dataclass(frozen=True)
class ContaminationResult:
    total_objects: int
    recyclable_objects: int
    contaminant_objects: int
    recyclable_percent: float
    contamination_percent: float
    severity: Severity


def classify_severity(contamination_percent: float, total_objects: int,
                      thresholds: SeverityThresholds = SeverityThresholds()) -> Severity:
    if total_objects == 0:
        return Severity.NONE
    if contamination_percent > thresholds.high:
        return Severity.HIGH
    if contamination_percent >= thresholds.moderate:
        return Severity.MODERATE
    return Severity.LOW


def calculate_contamination(
    counts: Mapping[str, int],
    allowed: frozenset[str] = RECYCLING_STREAM_ALLOWED,
    thresholds: SeverityThresholds = SeverityThresholds(),
) -> ContaminationResult:
    total = sum(counts.get(c, 0) for c in WASTE_CLASSES)
    recyclable = sum(counts.get(c, 0) for c in WASTE_CLASSES if c in allowed)
    contaminants = total - recyclable
    contamination = percent(contaminants, total)
    return ContaminationResult(
        total_objects=total,
        recyclable_objects=recyclable,
        contaminant_objects=contaminants,
        recyclable_percent=percent(recyclable, total),
        contamination_percent=contamination,
        severity=classify_severity(contamination, total, thresholds),
    )
