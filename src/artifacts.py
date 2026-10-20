"""
Loading the exported artefact package.

This module is identical in all ten projects. It exists to enforce one rule:

    The application reads its settings from model/config.json.
    It never hard-codes a value that training decided.

If you find yourself typing an image size, a normalisation constant, a class
name or a threshold anywhere in src/, stop: that value belongs in config.json,
exported by the training notebook. A value typed twice is a value that will
eventually disagree with itself, and the application will then produce
confidently wrong answers without raising a single error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"


class ArtifactError(RuntimeError):
    """Raised when the exported artefact package is missing or incomplete."""


def _read_json(name: str, required: bool = True) -> Any:
    path = MODEL_DIR / name
    if not path.exists():
        if not required:
            return None
        raise ArtifactError(
            f"Missing artefact: model/{name}\n"
            f"  Export it from the training notebook, or run:\n"
            f"      python tools/make_dummy_artifacts.py\n"
            f"  to create placeholder artefacts so the application starts."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"model/{name} is not valid JSON: {exc}") from exc


def load_config() -> dict:
    """Every preprocessing and inference parameter, as exported by training."""
    config = _read_json("config.json")
    if not isinstance(config, dict):
        raise ArtifactError("model/config.json must contain a JSON object.")
    return config


def load_classes() -> list[str]:
    """Class names in the exact index order the model outputs.

    Never retype this list by hand. Export it from the training dataset so the
    ordering cannot drift; a reordered list relabels every prediction silently.
    """
    classes = _read_json("classes.json")
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise ArtifactError("model/classes.json must contain a JSON list of strings.")
    return classes


def checkpoint_path(name: str = "best_model.pth") -> Path:
    path = MODEL_DIR / name
    if not path.exists():
        raise ArtifactError(
            f"Missing checkpoint: model/{name}\n"
            f"  Download it from Colab into the model/ directory, or run:\n"
            f"      python tools/make_dummy_artifacts.py"
        )
    return path


def extra_path(name: str, required: bool = True) -> Path | None:
    """A project-specific artefact: a scaler, a tokenizer, an index map..."""
    path = MODEL_DIR / name
    if not path.exists():
        if not required:
            return None
        raise ArtifactError(f"Missing artefact: model/{name}")
    return path


def describe(config: dict) -> dict:
    """Small summary shown in the page footer, so the interface always states
    which model produced the result the user is looking at."""
    return {
        "architecture": config.get("architecture", "unknown"),
        "trained_on": config.get("trained_on", "unknown"),
        "untrained": bool(config.get("untrained_placeholder", False)),
    }
