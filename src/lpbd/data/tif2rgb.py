"""Utilities for converting FTW Sentinel-2 GeoTIFF chips to RGB PNG images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from tqdm.auto import tqdm


def normalize_to_uint8(band: np.ndarray, lower: float = 2, upper: float = 98) -> np.ndarray:
    """Contrast-stretch one image band to uint8 using percentile cutoffs."""
    low, high = np.percentile(band, (lower, upper))
    band = np.clip(band, low, high)
    band = (band - low) / (high - low + 1e-8)
    return (band * 255).astype(np.uint8)


def read_tif_as_rgb(
    tif_path: Path,
    *,
    band_indexes: tuple[int, int, int] = (1, 2, 3),
    lower: float = 2,
    upper: float = 98,
) -> Image.Image:
    """Read a Sentinel-2 chip as a normalized PIL RGB image.

    The FTW files store B04, B03, B02, B08, so the default band indexes
    produce true-color RGB from Red, Green, and Blue.
    """
    tif_path = Path(tif_path)

    with rasterio.open(tif_path) as src:
        red = src.read(band_indexes[0])
        green = src.read(band_indexes[1])
        blue = src.read(band_indexes[2])

    rgb = np.stack(
        [
            normalize_to_uint8(red, lower=lower, upper=upper),
            normalize_to_uint8(green, lower=lower, upper=upper),
            normalize_to_uint8(blue, lower=lower, upper=upper),
        ],
        axis=-1,
    )

    return Image.fromarray(rgb).convert("RGB")


def convert_tif_to_rgb(
    tif_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
    lower: float = 2,
    upper: float = 98,
) -> Path:
    """Convert one GeoTIFF chip to one RGB PNG file."""
    tif_path = Path(tif_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = read_tif_as_rgb(tif_path, lower=lower, upper=upper)
    image.save(output_path)
    return output_path


def convert_tif_folder_to_rgb(
    input_dir: Path,
    output_dir: Path,
    *,
    pattern: str = "*.tif",
    overwrite: bool = False,
    lower: float = 2,
    upper: float = 98,
) -> list[Path]:
    """Convert every matching GeoTIFF in a folder to RGB PNG files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    tif_files = sorted(input_dir.glob(pattern))
    if not tif_files:
        raise RuntimeError(f"No files matching {pattern!r} found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for tif_path in tqdm(tif_files, desc="Converting TIF to RGB"):
        output_path = output_dir / f"{tif_path.stem}.png"
        output_paths.append(
            convert_tif_to_rgb(
                tif_path,
                output_path,
                overwrite=overwrite,
                lower=lower,
                upper=upper,
            )
        )

    return output_paths
