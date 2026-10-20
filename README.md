# Project 9 — Visual Defect and Anomaly Detection

*Finding faults when you have almost no examples of them*

| | |
|---|---|
| **Problem type** | One-class (unsupervised) anomaly detection |
| **Data** | MVTec AD — one category. Accept the licence and download it yourselves |
| **Model** | Convolutional autoencoder trained on normal images only |
| **Headline metrics** | ROC-AUC, then precision/recall/F1 at your threshold |

Read your full brief in **Part D of the student handbook**. This file covers the code.

---

## Quick start — before you have trained anything

You can have this application running in about five minutes, with placeholder
weights. Do this on day one: it lets the Application Lead build the entire
interface while the model is still training.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tools/make_dummy_artifacts.py   # placeholder model + test fixtures
python tools/check_setup.py            # verifies everything end to end
python app.py                          # http://127.0.0.1:5000
```

The page footer will say **untrained placeholder weights**. Predictions are
meaningless until you replace `model/` with your Colab export — which is
exactly the point.

> Install the CPU build of PyTorch to keep the environment small:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

---

## The two-week path

| When | Do this |
|---|---|
| Lab 1 | Open `notebook/training.ipynb` in Colab. Run it end to end on the placeholder data so you know the pipeline works. Agree roles and the artefact contract. |
| Week 1 | Collect or load your real data. Train. Evaluate. Export. |
| Lab 2 | Unzip the export into `model/`, run `python tools/check_setup.py`, complete the parity check, connect and test the interface. |
| Week 2 | Error analysis, README, rehearsal **with the network disconnected**, backup video. |

---

## What your group must change

1. Choose your category and justify it.
2. Derive the threshold from a percentile of normal validation scores, and justify the percentile.
3. Compare two scoring rules (`mean` against `max_patch`) and report which separates better.

Everything else is scaffolding. Marks come from these decisions and your
ability to defend them — not from the code you were handed.

---

## The artefact contract

Your Colab export must produce exactly these files in `model/`:

- best_model.pth
- classes.json
- config.json — input size, scoring rule, **anomaly threshold**

`src/` reads every parameter from `config.json`. **Never hard-code a value that
training decided** — a number typed in two places will eventually disagree with
itself, and the application will then produce confidently wrong answers without
raising a single error.

### The parity check — do not skip this

Before Lab 2 ends: run three examples through the notebook, record the output
probabilities, and check that your local `predict()` reproduces them to two or
three decimal places. `tests/test_inference.py` has a place to paste them.
This one check catches most integration bugs.

---

## Files

```
app.py                      Flask routes only — no model code
src/artifacts.py            loads config.json / classes.json (identical in all projects)
src/model.py                builds the architecture from config
src/preprocessing.py        the SAME transforms as training
src/inference.py            load() and predict() — the boundary of the ML system
src/webio.py                upload validation and image encoding
templates/index.html        extends templates/_base.html
static/css/style.css        shared stylesheet, readable when projected
model/                      your exported artefacts go here
notebook/training.ipynb     the Colab notebook
tools/make_dummy_artifacts.py   placeholder artefacts and fixtures
tools/check_setup.py        readiness check — run this first when something breaks
tests/test_inference.py     smoke tests and the parity check
```

---

## At the showcase

1. Run one unseen normal image and one unseen defective image.
2. Show the heatmap — it is what makes this demonstration memorable.
3. Explain how the threshold was chosen, and what would change if no defect could be allowed to escape.

---

## Troubleshooting

**Everything scores as normal**  
The bottleneck is too wide: the autoencoder is copying pixels and rebuilding defects perfectly. Reduce `latent_channels`.

**Everything scores as anomalous**  
Under-trained, or the threshold came from the wrong distribution. It must come from held-out NORMAL images.

**The heatmap lights up on shadows**  
A real finding about what the model has learned. Put it in your error analysis.

For anything else, run `python tools/check_setup.py` first — it reports the
first thing that is wrong, in order. The handbook's Part G has the full
troubleshooting table.

---

## Responsible use

Automated inspection displaces human judgement. State what happens to a flagged item and who is accountable, and note that thresholds do not transfer between products or lighting.
