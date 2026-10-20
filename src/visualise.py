"""
Turning the error map into the picture that makes this project memorable.

The heatmap is not decoration: it is the system explaining itself. It shows
WHERE the reconstruction failed, which is the closest thing an autoencoder has
to an explanation of its own verdict. Use it in your error analysis — a
heatmap that lights up on a shadow rather than the defect tells you something
important about what the model has learned.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")    # no display in a server process — set this BEFORE pyplot

import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def error_heatmap(error_map: torch.Tensor, original: Image.Image, alpha: float = 0.55) -> Image.Image:
    """Overlay the per-pixel error on the input image."""
    error = error_map[0, 0].numpy()
    spread = error.max() - error.min()
    normalised = (error - error.min()) / spread if spread > 1e-12 else np.zeros_like(error)

    coloured = (cm.inferno(normalised)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(coloured).resize(original.size, Image.BILINEAR)
    return Image.blend(original.convert("RGB"), heat, alpha)
