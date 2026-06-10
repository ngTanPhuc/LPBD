#!/usr/bin/env python3
"""Prepare a filtered FTW Vietnam dataset for SAM 3 training.

The pipeline starts by counting parcel instances in each mask and keeps only
chips with parcel_count <= --max-parcels. RGB conversion, image staging, and
annotation writing are then performed only for the filtered chips.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lpbd.data.mask_annotations import extract_instance_objects
from lpbd.data.tif2rgb import convert_tif_to_rgb

VALID_SPLITS = {"train", "val", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare filtered FTW Vietnam RGB images and annotations for SAM 3."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing data/, scripts/, and src/.",
    )
    parser.add_argument(
        "--window",
        default="window_a",
        choices=("window_a", "window_b"),
        help="FTW Sentinel-2 image window to prepare.",
    )
    parser.add_argument(
        "--output-name",
        default="sam3_ftw",
        help="Folder name under data/processed for the filtered prepared dataset.",
    )
    parser.add_argument(
        "--max-parcels",
        type=int,
        default=200,
        help="Keep only images with this many parcel instances or fewer.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move RGB PNGs instead of copying them. Default is copy to preserve data/processed/rgb.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the prepared SAM 3 folder.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output folder before preparing the filtered dataset.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected row has missing image, RGB, or mask files.",
    )
    parser.add_argument("--lower", type=float, default=2, help="Lower percentile for RGB conversion.")
    parser.add_argument("--upper", type=float, default=98, help="Upper percentile for RGB conversion.")
    parser.add_argument(
        "--text-prompt",
        default="agricultural field",
        help="Text prompt written to image text_input and annotation noun_phrase.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=5,
        help="Minimum parcel instance area, in pixels, to count and include in annotations.",
    )
    return parser.parse_args()


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_split_table(chips_path: Path) -> pd.DataFrame:
    require_path(chips_path, "chips parquet split file")

    df = pd.read_parquet(chips_path)
    required_columns = {"aoi_id", "split"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{chips_path} is missing required columns: {sorted(missing_columns)}")

    df = df[["aoi_id", "split"]].copy()
    df["aoi_id"] = df["aoi_id"].astype(str)
    df["split"] = df["split"].astype(str).str.lower()

    invalid_splits = sorted(set(df["split"]) - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Invalid split values in {chips_path}: {invalid_splits}")

    duplicate_ids = df[df.duplicated("aoi_id", keep=False)]["aoi_id"].unique()
    if len(duplicate_ids):
        raise ValueError(f"Duplicate aoi_id values in {chips_path}: {duplicate_ids[:10].tolist()}")

    return df.sort_values(["split", "aoi_id"]).reset_index(drop=True)


def count_parcels(mask_path: Path, min_area: int) -> int:
    """Count non-background parcel instances in one FTW instance mask."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    instance_ids, counts = np.unique(mask, return_counts=True)
    keep = (instance_ids != 0) & (counts >= min_area)
    return int(keep.sum())


def filter_rows_by_parcel_count(
    split_df: pd.DataFrame,
    instance_mask_dir: Path,
    *,
    max_parcels: int,
    min_area: int,
    strict: bool,
) -> pd.DataFrame:
    """Keep only rows whose instance mask has <= max_parcels objects."""
    kept_rows: list[dict[str, object]] = []
    missing_masks: list[str] = []
    over_limit = 0

    for row in tqdm(split_df.itertuples(index=False), total=len(split_df), desc="Counting parcels"):
        aoi_id = row.aoi_id
        mask_path = instance_mask_dir / f"{aoi_id}.tif"

        if not mask_path.exists():
            missing_masks.append(aoi_id)
            continue

        parcel_count = count_parcels(mask_path, min_area=min_area)
        if parcel_count <= max_parcels:
            kept_rows.append(
                {
                    "aoi_id": aoi_id,
                    "split": row.split,
                    "parcel_count": parcel_count,
                }
            )
        else:
            over_limit += 1

    if missing_masks:
        message = f"Missing {len(missing_masks)} instance masks. Sample: {missing_masks[:10]}"
        if strict:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}")

    filtered_df = pd.DataFrame(kept_rows, columns=["aoi_id", "split", "parcel_count"])
    print(f"Kept {len(filtered_df)} chips with <= {max_parcels} parcels")
    print(f"Skipped {over_limit} chips with > {max_parcels} parcels")
    return filtered_df.sort_values(["split", "aoi_id"]).reset_index(drop=True)


def prepare_directories(output_root: Path, *, clean_output: bool) -> dict[str, Path]:
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)

    annotations_dir = output_root / "annotations"
    images_root = output_root / "images"
    split_dirs = {split: images_root / split for split in sorted(VALID_SPLITS)}

    annotations_dir.mkdir(parents=True, exist_ok=True)
    for path in split_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return split_dirs


def ensure_rgb_images(
    filtered_df: pd.DataFrame,
    *,
    tif_dir: Path,
    rgb_dir: Path,
    lower: float,
    upper: float,
    strict: bool,
) -> list[str]:
    """Convert RGB only for selected chips that do not already have PNGs."""
    rgb_dir.mkdir(parents=True, exist_ok=True)
    missing_tifs: list[str] = []

    for row in tqdm(filtered_df.itertuples(index=False), total=len(filtered_df), desc="Checking RGB inputs"):
        aoi_id = row.aoi_id
        tif_path = tif_dir / f"{aoi_id}.tif"
        rgb_path = rgb_dir / f"{aoi_id}.png"

        if rgb_path.exists():
            continue

        if not tif_path.exists():
            missing_tifs.append(aoi_id)
            continue

        convert_tif_to_rgb(tif_path, rgb_path, overwrite=False, lower=lower, upper=upper)

    if missing_tifs:
        message = f"Missing {len(missing_tifs)} source TIF files. Sample: {missing_tifs[:10]}"
        if strict:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}")

    return missing_tifs


def stage_rgb_images(
    filtered_df: pd.DataFrame,
    rgb_dir: Path,
    split_dirs: dict[str, Path],
    *,
    move: bool,
    overwrite: bool,
    strict: bool,
) -> tuple[dict[str, int], list[str]]:
    action = shutil.move if move else shutil.copy2
    counts = {split: 0 for split in sorted(VALID_SPLITS)}
    missing: list[str] = []

    for row in tqdm(filtered_df.itertuples(index=False), total=len(filtered_df), desc="Staging RGB images"):
        aoi_id = row.aoi_id
        split = row.split
        src = rgb_dir / f"{aoi_id}.png"
        dst = split_dirs[split] / src.name

        if not src.exists():
            missing.append(aoi_id)
            continue

        if dst.exists():
            if overwrite:
                dst.unlink()
            else:
                counts[split] += 1
                continue

        action(src, dst)
        counts[split] += 1

    if missing:
        message = f"Missing {len(missing)} selected RGB images. Sample: {missing[:10]}"
        if strict:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}")

    return counts, missing


def write_split_annotations(
    filtered_df: pd.DataFrame,
    *,
    output_root: Path,
    split_dirs: dict[str, Path],
    instance_mask_dir: Path,
    text_prompt: str,
    min_area: int,
    strict: bool,
) -> dict[str, int]:
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    category = {"id": 1, "name": text_prompt}
    annotation_counts: dict[str, int] = {split: 0 for split in sorted(VALID_SPLITS)}

    for split in sorted(VALID_SPLITS):
        images = []
        annotations = []
        image_id = 1
        annotation_id = 1
        split_rows = filtered_df[filtered_df["split"] == split]

        for row in tqdm(split_rows.itertuples(index=False), total=len(split_rows), desc=f"Writing {split} annotations"):
            aoi_id = row.aoi_id
            image_path = split_dirs[split] / f"{aoi_id}.png"
            mask_path = instance_mask_dir / f"{aoi_id}.tif"

            if not image_path.exists():
                continue

            with Image.open(image_path) as image:
                width, height = image.size

            images.append(
                {
                    "id": image_id,
                    "file_name": f"images/{split}/{image_path.name}",
                    "width": width,
                    "height": height,
                    "text_input": text_prompt,
                    "parcel_count": int(row.parcel_count),
                }
            )

            if not mask_path.exists():
                message = f"Missing instance mask for {aoi_id}: {mask_path}"
                if strict:
                    raise FileNotFoundError(message)
                print(f"WARNING: {message}")
                image_id += 1
                continue

            for obj in extract_instance_objects(mask_path, min_area=min_area):
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": obj.bbox,
                        "area": obj.area,
                        "segmentation": obj.segmentation,
                        "iscrowd": 0,
                        "noun_phrase": text_prompt,
                    }
                )
                annotation_id += 1

            image_id += 1

        payload = {
            "images": images,
            "annotations": annotations,
            "categories": [category],
        }

        out_path = annotations_dir / f"{split}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

        annotation_counts[split] = len(annotations)

    return annotation_counts


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    raw_root = project_root / "data" / "raw" / "FTW_Vietnam"
    processed_root = project_root / "data" / "processed"
    tif_dir = raw_root / "s2_images" / args.window
    rgb_dir = processed_root / "rgb" / args.window
    output_root = processed_root / args.output_name
    chips_path = raw_root / "chips_vietnam.parquet"
    instance_mask_dir = raw_root / "label_masks" / "instance"

    require_path(raw_root, "FTW Vietnam raw folder")
    require_path(tif_dir, "FTW Sentinel-2 image folder")
    require_path(instance_mask_dir, "FTW instance mask folder")

    # Step 1: determine which chips satisfy parcel_count <= max_parcels.
    split_df = load_split_table(chips_path)
    filtered_df = filter_rows_by_parcel_count(
        split_df,
        instance_mask_dir,
        max_parcels=args.max_parcels,
        min_area=args.min_area,
        strict=args.strict,
    )

    # Step 2: only create/check RGB PNGs for selected chips.
    ensure_rgb_images(
        filtered_df,
        tif_dir=tif_dir,
        rgb_dir=rgb_dir,
        lower=args.lower,
        upper=args.upper,
        strict=args.strict,
    )

    # Step 3: create the filtered SAM 3 folder layout.
    split_dirs = prepare_directories(output_root, clean_output=args.clean_output)

    # Step 4: only stage selected RGB images into split folders.
    counts, missing = stage_rgb_images(
        filtered_df,
        rgb_dir,
        split_dirs,
        move=args.move,
        overwrite=args.overwrite,
        strict=args.strict,
    )

    # Step 5: only write annotations for selected images.
    annotation_counts = write_split_annotations(
        filtered_df,
        output_root=output_root,
        split_dirs=split_dirs,
        instance_mask_dir=instance_mask_dir,
        text_prompt=args.text_prompt,
        min_area=args.min_area,
        strict=args.strict,
    )

    print("Filtered SAM 3 FTW dataset prepared")
    print(f"Output root: {output_root}")
    print(f"Max parcels per image: {args.max_parcels}")
    for split in sorted(counts):
        print(f"  {split}: {counts[split]} images, {annotation_counts[split]} annotations")
    if missing:
        print(f"  missing selected RGB entries: {len(missing)}")


if __name__ == "__main__":
    main()
