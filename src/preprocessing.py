"""
Preparing inspection images.

No ImageNet normalisation here: the autoencoder's output layer is a sigmoid
producing values in [0, 1], so the input must live in [0, 1] too. Normalising
the input but not the target is a common and confusing mistake — the loss
then never drops properly.
"""

from __future__ import annotations

import io

import torch
from PIL import Image, ImageOps
from torchvision import transforms


def load_image(raw: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def build_transform(config: dict) -> transforms.Compose:
    height, width = config["input_size"]
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),          # [0, 1], matching the sigmoid output
    ])


def preprocess(raw: bytes, config: dict) -> tuple[torch.Tensor, Image.Image]:
    image = load_image(raw)
    tensor = build_transform(config)(image)
    return tensor.unsqueeze(0), image
