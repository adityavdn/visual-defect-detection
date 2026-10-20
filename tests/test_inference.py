"""Tests for the inspection prototype. Run: python tests/test_inference.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import artifacts, inference  # noqa: E402

NORMAL = ROOT / "tests" / "fixtures" / "example.png"
DEFECTIVE = ROOT / "tests" / "fixtures" / "example_defective.png"


def test_threshold_is_exported():
    config = artifacts.load_config()
    assert config["anomaly_threshold"] > 0, "the threshold must come from validation data"
    assert config["scoring_rule"] in {"mean", "max_patch"}


def test_reconstruction_has_the_same_shape_as_the_input():
    """A shape mismatch here means the encoder and decoder depths disagree."""
    result = inference.predict(NORMAL.read_bytes())
    tensors = result["tensors"]
    assert tensors["input"].shape == tensors["reconstruction"].shape


def test_score_is_non_negative_and_verdict_follows_threshold():
    result = inference.predict(NORMAL.read_bytes())
    assert result["score"] >= 0.0
    assert result["anomalous"] == (result["score"] >= result["threshold"])


def test_both_scoring_rules_run():
    """max_patch should never score below mean: it takes the worst local patch."""
    from src.inference import _score

    result = inference.predict(DEFECTIVE.read_bytes())
    error_map = result["tensors"]["error"]
    mean_score = _score(error_map, "mean", 16)
    patch_score = _score(error_map, "max_patch", 16)
    assert patch_score >= mean_score


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("PASS", name)
    print("\nAll checks passed.")
