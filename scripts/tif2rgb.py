#!/usr/bin/env python3

import rasterio
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm


# ================================
# 1. Load the paths
# ================================
"""
The dataset are in 4 bands: Shape [C,H,W]: (4, 256, 256) with 4 bands:
- Band 1 = Red
- Band 2 = Green
- Band 3 = Blue
- Band 4 = Near Infrared

But the SAM 3 model only takes 3 bands Shape [H, W, 3] or PIL.Image RGB:
- Band 1 = Red
- Band 2 = Green
- Band 3 = Blue
"""

ROOT = Path("data")
RAW = ROOT / "raw" / "FTW_Vietnam"
PROCESSED = ROOT / "processed"

img_dir = RAW / "s2_images" / "window_a"
lst_tif_files = sorted(img_dir.glob("*.tif"))

sem3_dir = RAW / "label_masks" / "semantic_3class"
inst_dir = RAW / "label_masks" / "instance"

rgb_dir = PROCESSED / "rgb" / "window_a"


# ================================
# 2. Prepare the images for SAM 3
# ================================
# Now we will change all the images to the right shape to test with SAM 3

def normalize(band, lower=2, upper=98):
    low, high = np.percentile(band, (2, 98))
    band = np.clip(band, low, high)
    band = (band - low) / (high - low + 1e-8)  # scale the value to range(0, 1)
    return (band * 255).astype(np.uint8)  # SAM 3 uses uint8

def convert_tif2rgb():
    for tif_path in tqdm(lst_tif_files):
        with rasterio.open(tif_path) as src:
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)


        rgb = np.stack(
            [
                normalize(red),
                normalize(green),
                normalize(blue)
            ],
             axis=-1  # stack along the last dimension
        ).astype(np.uint8)

        img = Image.fromarray(rgb).convert("RGB")

        out_path = rgb_dir / f"{tif_path.stem}.png"
        img.save(out_path)


def main():
    convert_tif2rgb()

if __name__ == "__main__":
    main()