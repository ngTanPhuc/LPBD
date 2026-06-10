"""Utilities for converting FTW instance masks to polygon annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from skimage import measure


@dataclass(frozen=True)
class InstanceObject:
    """One parcel instance extracted from an instance-mask raster."""

    instance_id: int
    segmentation: list[list[float]]
    bbox: list[int]
    area: int


def read_mask(mask_path: Path) -> np.ndarray:
    """Read a single-band instance mask from a GeoTIFF file."""
    with rasterio.open(mask_path) as src:
        return src.read(1)


def binary_mask_to_polygons(binary_mask: np.ndarray, min_points: int = 3) -> list[list[float]]:
    """Convert a binary object mask to COCO-style polygon lists.

    Each polygon is stored as [x1, y1, x2, y2, ...]. Raster contours are
    returned as row/column pairs, so this function flips them to x/y pairs.
    """
    polygons: list[list[float]] = []
    contours = measure.find_contours(binary_mask.astype(np.uint8), level=0.5)
    height, width = binary_mask.shape

    for contour in contours:
        if len(contour) < min_points:
            continue

        polygon: list[float] = []
        for row, col in contour:
            x = float(np.clip(col, 0, width - 1))
            y = float(np.clip(row, 0, height - 1))
            polygon.extend([x, y])

        if len(polygon) >= min_points * 2:
            polygons.append(polygon)

    return polygons


def bbox_from_binary_mask(binary_mask: np.ndarray) -> list[int] | None:
    """Return COCO bbox [x, y, width, height] for a binary mask."""
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    y_min = int(ys.min())
    box_width = int(xs.max() - x_min + 1)
    box_height = int(ys.max() - y_min + 1)
    return [x_min, y_min, box_width, box_height]


def extract_instance_objects(mask_path: Path, min_area: int = 5) -> list[InstanceObject]:
    """Extract parcel instances from an FTW instance-mask GeoTIFF."""
    mask = read_mask(mask_path)
    instance_ids = np.unique(mask)
    objects: list[InstanceObject] = []

    for instance_id in instance_ids:
        if instance_id == 0:
            continue

        binary_mask = mask == instance_id
        area = int(binary_mask.sum())
        if area < min_area:
            continue

        bbox = bbox_from_binary_mask(binary_mask)
        if bbox is None:
            continue

        segmentation = binary_mask_to_polygons(binary_mask)
        if not segmentation:
            continue

        objects.append(
            InstanceObject(
                instance_id=int(instance_id),
                segmentation=segmentation,
                bbox=bbox,
                area=area,
            )
        )

    return objects


def extract_object_segmentations(mask_path: Path, min_area: int = 1) -> list[list[float]]:
    """Return one polygon list per extracted object from an instance-mask file.

    If an object has multiple contour polygons, they are flattened into separate
    polygon entries so the return value remains list[list[float]].
    """
    segmentations: list[list[float]] = []
    for obj in extract_instance_objects(mask_path, min_area=min_area):
        segmentations.extend(obj.segmentation)
    return segmentations
