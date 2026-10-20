"""Placeholder autoencoder so the interface can be built on day one.

    python tools/make_dummy_artifacts.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import build_model  # noqa: E402

# --------------------------------------------------------------------------
CATEGORY = "bottle"              # the one MVTec AD category your group chose
INPUT_SIZE = [128, 128]
SCORING_RULE = "mean"            # or "max_patch"
ANOMALY_THRESHOLD = 0.01         # DERIVE THIS from normal validation scores
# --------------------------------------------------------------------------

config = {
    "task": "anomaly_detection",
    "architecture": "conv_autoencoder",
    "category": CATEGORY,
    "input_size": INPUT_SIZE,
    "base_channels": 32,
    "latent_channels": 128,
    "depth": 4,
    "scoring_rule": SCORING_RULE,
    "patch_size": 16,
    "anomaly_threshold": ANOMALY_THRESHOLD,
    "threshold_rationale": "Replace with your own: which percentile of normal validation scores, "
                           "and what false-alarm rate does that imply?",
    "trained_on": str(date.today()),
    "untrained_placeholder": True,
}

model_dir = ROOT / "model"
model_dir.mkdir(exist_ok=True)
torch.save(build_model(config).state_dict(), model_dir / "best_model.pth")
(model_dir / "classes.json").write_text(json.dumps(["normal", "anomaly"], indent=2))
(model_dir / "config.json").write_text(json.dumps(config, indent=2))

fixtures = ROOT / "tests" / "fixtures"
fixtures.mkdir(parents=True, exist_ok=True)
for name, defect in (("example.png", False), ("example_defective.png", True)):
    image = Image.new("RGB", (256, 256), (205, 208, 212))
    draw = ImageDraw.Draw(image)
    draw.ellipse((48, 48, 208, 208), fill=(150, 160, 172), outline=(110, 120, 132), width=3)
    if defect:
        draw.line((96, 150, 168, 104), fill=(46, 30, 26), width=5)     # a "scratch"
    image.save(fixtures / name)

print("Wrote placeholder artefacts and two fixtures (normal and defective).")
print("Next: python tools/check_setup.py")
