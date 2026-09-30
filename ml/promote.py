"""Promote a training run to the deployed model (models/best.pt + models/best.json).

Usage:
    python ml/promote.py runs/detect/yolov8s_baseline
"""
from __future__ import annotations

import argparse
import sys

from utils.paths import resolve_path
from utils.promotion import promote_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="training run directory containing weights/best.pt")
    args = parser.parse_args()

    meta = promote_run(resolve_path(args.run_dir))
    print(f"Promoted {meta['source_run']} -> models/best.pt")
    print(f"model_version: {meta['model_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
