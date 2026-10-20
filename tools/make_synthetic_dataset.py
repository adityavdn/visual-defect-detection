"""Recreate the synthetic dataset the notebook trained on.

    python tools/make_synthetic_dataset.py

Same drawing code and the same seed (42) as notebook cell 3, so the images
match the ones generated in Colab when MVTec AD was not found.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "synthetic-data" / "bottle"


def make_product(defective: bool = False, size: int = 256) -> Image.Image:
    image = Image.new('RGB', (size, size), (208, 210, 214))
    draw = ImageDraw.Draw(image)
    centre = size // 2
    draw.ellipse((centre - 84, centre - 84, centre + 84, centre + 84),
                 fill=(150, 158, 170), outline=(112, 120, 132), width=3)
    for _ in range(60):                      # texture, so reconstruction is non-trivial
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        draw.point((x, y), fill=(random.randint(120, 200),) * 3)
    if defective:
        kind = random.choice(['scratch', 'chip', 'stain'])
        if kind == 'scratch':
            x = random.randint(centre - 50, centre + 10)
            draw.line((x, centre - 40, x + random.randint(30, 70), centre + 30),
                      fill=(46, 32, 28), width=random.randint(3, 6))
        elif kind == 'chip':
            x, y = random.randint(centre - 60, centre + 40), random.randint(centre - 60, centre + 40)
            draw.ellipse((x, y, x + 26, y + 22), fill=(90, 70, 60))
        else:
            x, y = random.randint(centre - 50, centre + 30), random.randint(centre - 50, centre + 30)
            draw.ellipse((x, y, x + 40, y + 36), fill=(128, 128, 110))
    return image.filter(ImageFilter.GaussianBlur(0.4))


if __name__ == "__main__":
    random.seed(SEED)
    for folder, count, defective in [('train/good', 220, False), ('test/good', 40, False),
                                     ('test/defect', 40, True)]:
        target = OUT / folder
        target.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            make_product(defective).save(target / f'{i:03d}.png')
    print(f"Wrote 300 images to {OUT}")
