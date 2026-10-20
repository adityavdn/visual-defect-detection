"""Flask entry point for the visual inspection prototype.  python app.py"""

from __future__ import annotations

import os

from flask import Flask, render_template, request

from src import artifacts, inference
from src.visualise import error_heatmap, tensor_to_image
from src.webio import UploadError, image_data_uri, read_upload

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def page(**context):
    for key in ("result", "error", "views"):
        context.setdefault(key, None)
    context["meta"] = artifacts.describe(artifacts.load_config())
    return render_template("index.html", **context)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method != "POST":
        return page()
    try:
        raw = read_upload(request.files.get("image"), ALLOWED)
        result = inference.predict(raw)
        tensors = result.pop("tensors")
        original = result.pop("image")

        views = {
            "original": image_data_uri(tensor_to_image(tensors["input"])),
            "reconstruction": image_data_uri(tensor_to_image(tensors["reconstruction"])),
            "heatmap": image_data_uri(error_heatmap(
                tensors["error"], tensor_to_image(tensors["input"]))),
        }
        return page(result=result, views=views)
    except UploadError as exc:
        return page(error=str(exc))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("scoring failed")
        return page(error="Could not process that image ({}).".format(type(exc).__name__))


@app.errorhandler(413)
def too_large(_):
    return page(error="That file is too large (8 MB maximum)."), 413


if __name__ == "__main__":
    inference.warm_up()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
