"""
Scoring an image for abnormality.

    anomaly score = how badly the autoencoder failed to rebuild this image

Two scoring rules are provided because they behave very differently, and the
brief asks you to compare them:

  mean        Average squared error over the whole image. Stable, but a small
              defect is diluted by a large expanse of correctly rebuilt
              background — it is easy to miss a scratch this way.
  max_patch   The worst local patch, after average-pooling the error map.
              Much more sensitive to small, concentrated defects, and
              correspondingly more prone to false alarms on noise.

The threshold is NOT a round number someone liked. It comes from the
distribution of scores on normal validation images — typically a high
percentile — so that you can state your expected false-alarm rate before you
ever see a defect.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from src import artifacts
from src.model import build_model
from src.preprocessing import preprocess


@lru_cache(maxsize=1)
def load():
    config = artifacts.load_config()
    model = build_model(config)
    state = torch.load(artifacts.checkpoint_path(), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


def _score(error_map: torch.Tensor, rule: str, patch: int) -> float:
    """error_map: [1, 1, H, W] of per-pixel squared error."""
    if rule == "max_patch":
        pooled = F.avg_pool2d(error_map, kernel_size=patch, stride=max(1, patch // 2))
        return float(pooled.max())
    return float(error_map.mean())


@torch.inference_mode()
def predict(raw: bytes) -> dict:
    model, config = load()
    tensor, original = preprocess(raw, config)

    reconstruction = model(tensor)
    error_map = ((tensor - reconstruction) ** 2).mean(dim=1, keepdim=True)

    rule = config.get("scoring_rule", "mean")
    score = _score(error_map, rule, int(config.get("patch_size", 16)))
    threshold = float(config["anomaly_threshold"])
    anomalous = score >= threshold

    return {
        "label": "ANOMALY" if anomalous else "NORMAL",
        "anomalous": anomalous,
        "score": score,
        "threshold": threshold,
        "scoring_rule": rule,
        # How far past (or short of) the threshold, as a readable multiple.
        "ratio": score / threshold if threshold > 0 else float("inf"),
        "tensors": {"input": tensor, "reconstruction": reconstruction, "error": error_map},
        "image": original,
    }


def warm_up() -> None:
    try:
        predict(example_input())
    except Exception:  # noqa: BLE001
        pass


def example_input() -> bytes:
    return (Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "example.png").read_bytes()
