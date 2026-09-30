"""Deterministic, class-stratified train/val/test splitting for multi-label images."""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

SPLITS = ("train", "val", "test")
BACKGROUND_KEY = "__background__"


def stratified_split(class_sets: Sequence[set[str]], ratios: Sequence[float], seed: int) -> list[str]:
    """Assign each item to train/val/test.

    Each item is stratified by the *rarest* class it contains (by image frequency),
    so minority classes get their fair share in every split even inside mixed images.
    Returns one split name per item, in input order.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6 or min(ratios) < 0:
        raise ValueError(f"ratios must be three non-negative numbers summing to 1, got {ratios}")

    frequency = Counter(c for classes in class_sets for c in classes)

    def key(classes: set[str]) -> str:
        return min(classes, key=lambda c: (frequency[c], c)) if classes else BACKGROUND_KEY

    strata: dict[str, list[int]] = {}
    for idx, classes in enumerate(class_sets):
        strata.setdefault(key(classes), []).append(idx)

    _, val_ratio, test_ratio = ratios
    assignment = [""] * len(class_sets)
    for stratum, indices in sorted(strata.items()):
        rng = random.Random(f"{seed}:{stratum}")  # str seeds are stable across runs/platforms
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_test = round(len(shuffled) * test_ratio)
        n_val = round(len(shuffled) * val_ratio)
        for pos, idx in enumerate(shuffled):
            assignment[idx] = "test" if pos < n_test else "val" if pos < n_test + n_val else "train"
    return assignment
