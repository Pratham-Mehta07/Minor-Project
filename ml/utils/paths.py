"""Project-relative paths so scripts behave the same from any working directory."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_DIR = PROJECT_ROOT / "ml"
BACKEND_DIR = PROJECT_ROOT / "backend"
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
RUNS_DIR = PROJECT_ROOT / "runs"


def resolve_path(path: str | Path) -> Path:
    """Resolve a user-supplied path; relative paths are taken from the project root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
