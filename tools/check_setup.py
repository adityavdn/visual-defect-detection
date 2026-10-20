"""
Check that this project is ready to run, before you touch Flask.

    python tools/check_setup.py

It verifies, in order: the Python version, the required packages, the exported
artefact package, that the model loads on CPU, and that a single prediction
runs end to end. Fix the first failure it reports, then run it again.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "
failures = 0


def report(status: str, message: str, detail: str = "") -> None:
    global failures
    print(f"[{status}] {message}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")
    if status == BAD:
        failures += 1


print("=" * 68)
print("Project readiness check")
print("=" * 68)

# 1 --------------------------------------------------------------- Python
if sys.version_info >= (3, 9):
    report(OK, f"Python {sys.version_info.major}.{sys.version_info.minor}")
else:
    report(BAD, f"Python {sys.version_info.major}.{sys.version_info.minor} is too old — use 3.9 or newer")

in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
report(OK if in_venv else WARN,
       "Running inside a virtual environment" if in_venv
       else "Not in a virtual environment — create one with: python -m venv .venv")

# 2 ------------------------------------------------------------- packages
required = [line.split("=")[0].split(">")[0].split("[")[0].strip()
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")]

IMPORT_NAME = {"pillow": "PIL", "scikit-learn": "sklearn", "facenet-pytorch": "facenet_pytorch",
               "python-dateutil": "dateutil", "opencv-python": "cv2"}

for package in required:
    module = IMPORT_NAME.get(package.lower(), package.replace("-", "_"))
    try:
        importlib.import_module(module)
        report(OK, f"package: {package}")
    except ImportError:
        report(BAD, f"package missing: {package}", "pip install -r requirements.txt")

# 3 ------------------------------------------------------------ artefacts
try:
    from src import artifacts

    config = artifacts.load_config()
    report(OK, "model/config.json loaded", f"architecture: {config.get('architecture')}")
    if config.get("untrained_placeholder"):
        report(WARN, "These are PLACEHOLDER artefacts with random weights",
               "Predictions will be meaningless until you export a trained model from Colab.")
    # Most projects export a single best_model.pth; project 7 exports a
    # transformers directory instead, named in config["checkpoint"].
    checkpoint = config.get("checkpoint", "best_model.pth")
    path = artifacts.MODEL_DIR / checkpoint
    if path.exists():
        report(OK, "model/{} present".format(checkpoint))
    else:
        report(BAD, "model/{} is missing".format(checkpoint),
               "Export it from the notebook, or run: python tools/make_dummy_artifacts.py")
except Exception as exc:  # noqa: BLE001 - surface any artefact problem plainly
    report(BAD, "artefact package incomplete", str(exc))

# 4 ------------------------------------------------------- load + predict
if failures == 0:
    try:
        from src import inference

        inference.load()
        report(OK, "model loaded on CPU and set to eval mode")
    except Exception as exc:  # noqa: BLE001
        report(BAD, "model failed to load", str(exc))

    try:
        from src import inference

        fixture = inference.example_input()
        result = inference.predict(fixture)
        preview = ", ".join(f"{k}={v}" for k, v in list(result.items())[:3]
                            if not isinstance(v, (list, dict)))
        report(OK, "one prediction ran end to end", preview)
    except Exception as exc:  # noqa: BLE001
        report(BAD, "prediction failed", str(exc))

print("=" * 68)
if failures:
    print(f"{failures} problem(s) found. Fix the first one and run this again.")
    sys.exit(1)
print("Ready. Start the application with:  python app.py")
print("Then open http://127.0.0.1:5000")
