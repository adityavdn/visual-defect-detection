"""
Small helpers for moving files and pictures between the browser and the model.

Identical in every project. Nothing here knows anything about deep learning —
keeping web plumbing out of inference.py is what makes the inference path
testable without Flask.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # keep in step with MAX_CONTENT_LENGTH in app.py


class UploadError(ValueError):
    """Raised for anything the user can fix by choosing a different file."""


def read_upload(storage, allowed_extensions: set[str], max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Validate a Werkzeug FileStorage and return its bytes.

    Never trust an uploaded file: check that something was actually chosen,
    that the extension is one you support, and that the size is sane. The
    messages raised here are shown to the user, so they say what to do next.
    """
    if storage is None or not getattr(storage, "filename", ""):
        raise UploadError("No file was selected.")

    suffix = Path(storage.filename).suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        given = suffix if suffix else "none"
        raise UploadError("Unsupported file type '{}'. Accepted: {}.".format(given, allowed))

    raw = storage.read()
    if not raw:
        raise UploadError("That file is empty.")
    if len(raw) > max_bytes:
        raise UploadError(f"That file is larger than {max_bytes // (1024 * 1024)} MB.")
    return raw


def data_uri_from_bytes(raw: bytes, mime: str) -> str:
    """Inline a file in the page, so results render without writing to disk."""
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def image_data_uri(image, fmt: str = "PNG") -> str:
    """Inline a PIL image."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    mime = "image/jpeg" if fmt.upper() in {"JPG", "JPEG"} else f"image/{fmt.lower()}"
    return data_uri_from_bytes(buffer.getvalue(), mime)


def figure_data_uri(figure) -> str:
    """Inline a Matplotlib figure.

    Always select the Agg backend before importing pyplot in a web application;
    the default interactive backend expects a window system and will hang or
    crash inside a server process.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    buffer.seek(0)
    try:
        import matplotlib.pyplot as plt

        plt.close(figure)  # release memory: a server runs for hours
    except Exception:  # pragma: no cover - plotting is optional in some projects
        pass
    return data_uri_from_bytes(buffer.getvalue(), "image/png")
