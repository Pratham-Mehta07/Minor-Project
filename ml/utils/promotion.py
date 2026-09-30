"""Promote a training run's best weights to models/best.pt (+ models/best.json)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from utils.paths import MODELS_DIR, PROJECT_ROOT
from utils.run_info import file_sha256


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def promote_run(run_dir: Path, target: Path = MODELS_DIR / "best.pt") -> dict:
    """Copy <run_dir>/weights/best.pt to `target` and write metadata next to it.

    The model version is "<run name>-<first 8 hex of weights sha256>", so two
    different weight files can never share a version string.
    """
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No best.pt in {run_dir / 'weights'}")

    sha = file_sha256(weights)
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}

    metadata = {
        "model_version": f"{run_dir.name}-{sha[:8]}",
        "weights_sha256": sha,
        "source_run": _relative(run_dir),
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": run_config.get("classes"),
        "train_config": run_config.get("config"),
        "dataset": run_config.get("dataset"),
        "val_metrics": run_config.get("final_val_metrics"),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, target)
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
