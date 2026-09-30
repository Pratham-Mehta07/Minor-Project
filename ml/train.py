"""Train the WasteWise YOLOv8 detector.

Defaults come from ml/configs/train.yaml; command-line flags override them.

Usage:
    python ml/train.py
    python ml/train.py --epochs 3 --name smoke_test              # quick pipeline check
    python ml/train.py --epochs 10 --fraction 0.5 --name proto    # fast low-accuracy prototype
    python ml/train.py --model yolov8n.pt --batch 8 --device cpu
    python ml/train.py --name yolov8s_baseline --resume           # continue an interrupted run
    python ml/train.py --promote                                  # copy best.pt to models/ afterwards

Outputs (runs/detect/<name>/): weights/best.pt, weights/last.pt, results.csv,
plots, args.yaml (Ultralytics) and run_config.json (config + environment + dataset + metrics).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from utils.classes import WASTE_CLASSES
from utils.device import describe_device, resolve_device
from utils.paths import PRETRAINED_DIR, RUNS_DIR, resolve_path
from utils.promotion import promote_run
from utils.run_info import dataset_info, environment_info, resolve_data_yaml
from utils.runtime import configure_dataloader

DEFAULT_CONFIG = "ml/configs/train.yaml"
PROJECT_DIR = RUNS_DIR / "detect"
OVERRIDABLE = ("model", "data", "name", "epochs", "patience", "imgsz", "batch", "device", "workers", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="training config YAML")
    parser.add_argument("--model", help="starting weights, e.g. yolov8n.pt / yolov8s.pt or a path")
    parser.add_argument("--data", help="dataset YAML")
    parser.add_argument("--name", help="experiment name (run folder)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", help="auto, cpu, 0, 0,1 ...")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fraction", type=float, help="use this fraction of the training images (quick prototypes)")
    parser.add_argument("--resume", action="store_true", help="resume runs/detect/<name>/weights/last.pt")
    parser.add_argument("--promote", action="store_true", help="copy best.pt to models/best.pt when done")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    for key in OVERRIDABLE:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config.setdefault("extra", {})
    if args.fraction is not None:
        if not 0 < args.fraction <= 1:
            raise SystemExit("--fraction must be in (0, 1]")
        config["extra"]["fraction"] = args.fraction
    return config


def resolve_model(model: str) -> str:
    """Existing paths are used as-is; bare official names are downloaded into models/pretrained/."""
    path = resolve_path(model)
    if path.exists() or Path(model).suffix == ".yaml":
        return str(path) if path.exists() else model
    if Path(model).parent == Path("."):
        return str(PRETRAINED_DIR / model)
    raise FileNotFoundError(f"Model weights not found: {path}")


def check_classes(data_yaml: Path) -> None:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    if names != list(WASTE_CLASSES):
        raise SystemExit(f"{data_yaml}: class names {names} do not match canonical {list(WASTE_CLASSES)}")


def write_run_config(save_dir: Path, payload: dict) -> None:
    (save_dir / "run_config.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args)

    configure_dataloader()
    from ultralytics import YOLO  # imported late (after dataloader config) so --help is instant

    data_yaml = resolve_path(config["data"])
    check_classes(data_yaml)
    device = resolve_device(str(config["device"]))
    resolved_data = resolve_data_yaml(data_yaml, PROJECT_DIR / ".data" / f"{config['name']}.yaml")

    payload: dict = {
        "config": config,
        "resolved_device": device,
        "device_description": describe_device(device),
        "classes": list(WASTE_CLASSES),
        "environment": environment_info(),
        "dataset": dataset_info(data_yaml),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if args.resume:
        last = PROJECT_DIR / config["name"] / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Cannot resume: {last} not found")
        model = YOLO(str(last))
        train_kwargs: dict = {"resume": True}
    else:
        model = YOLO(resolve_model(str(config["model"])))
        train_kwargs = {
            "data": str(resolved_data),
            "epochs": config["epochs"],
            "patience": config["patience"],
            "imgsz": config["imgsz"],
            "batch": config["batch"],
            "device": device,
            "workers": config["workers"],
            "seed": config["seed"],
            "deterministic": config.get("deterministic", True),
            "project": str(PROJECT_DIR),
            "name": config["name"],
            **config["extra"],
        }

    def on_pretrain_end(trainer) -> None:
        # Record the configuration before the first epoch so interrupted runs are documented too.
        save_dir = Path(trainer.save_dir)
        shutil.copy2(resolved_data, save_dir / "data_resolved.yaml")
        write_run_config(save_dir, payload)

    model.add_callback("on_pretrain_routine_end", on_pretrain_end)

    print(f"Training {config['model']} on {data_yaml} using {payload['device_description']}")
    start = time.perf_counter()
    model.train(**train_kwargs)
    trainer = model.trainer
    save_dir = Path(trainer.save_dir)

    payload.update({
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_minutes": round((time.perf_counter() - start) / 60, 1),
        "epochs_completed": trainer.epoch + 1,
        "best_fitness": float(trainer.best_fitness) if trainer.best_fitness is not None else None,
        # Validation-split metrics of best.pt; the held-out test split is evaluated by ml/evaluate.py.
        "final_val_metrics": {k: round(float(v), 5) for k, v in (trainer.metrics or {}).items()},
    })
    write_run_config(save_dir, payload)

    print(f"\nRun directory: {save_dir}")
    print(f"Best weights:  {save_dir / 'weights' / 'best.pt'}")
    for key, value in payload["final_val_metrics"].items():
        print(f"  {key:<24} {value}")

    if args.promote:
        meta = promote_run(save_dir)
        print(f"\nPromoted to models/best.pt as model_version={meta['model_version']}")
    else:
        print(f"\nTo deploy: python ml/promote.py {save_dir.relative_to(RUNS_DIR.parent).as_posix()}")
    print("Next: python ml/evaluate.py")
    return 0


if __name__ == "__main__":  # required on Windows: dataloader workers re-import this module
    sys.exit(main())
