#!/usr/bin/env python3
"""Prepare the FTW Vietnam RGB image split for SAM 3 training.

This script expects the FTW raw data under data/raw/FTW_Vietnam and uses
chips_vietnam.parquet to place RGB images into train/val/test folders.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lpbd.data.mask_annotations import extract_instance_objects
from lpbd.data.tif2rgb import convert_tif_folder_to_rgb

VALID_SPLITS = {"train", "val", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FTW Vietnam RGB images in SAM 3 train/val/test layout."
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
        help="Folder name under data/processed for the prepared dataset.",
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
        "--strict",
        action="store_true",
        help="Fail if any parquet aoi_id has no matching RGB image.",
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
        default=1,
        help="Minimum parcel instance area, in pixels, to include in annotations.",
    )
    return parser.parse_args()


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def expected_tif_stems(tif_dir: Path) -> set[str]:
    return {path.stem for path in tif_dir.glob("*.tif")}


def existing_rgb_stems(rgb_dir: Path) -> set[str]:
    return {path.stem for path in rgb_dir.glob("*.png")}


def ensure_rgb_images(
    raw_root: Path,
    rgb_dir: Path,
    window: str,
    *,
    lower: float,
    upper: float,
) -> None:
    tif_dir = raw_root / "s2_images" / window
    require_path(tif_dir, "raw Sentinel-2 image folder")

    expected = expected_tif_stems(tif_dir)
    if not expected:
        raise RuntimeError(f"No .tif files found in {tif_dir}")

    rgb_dir.mkdir(parents=True, exist_ok=True)
    existing = existing_rgb_stems(rgb_dir)
    missing = sorted(expected - existing)

    if not missing:
        print(f"RGB check passed: {len(existing & expected)}/{len(expected)} images exist in {rgb_dir}")
        return

    print(f"RGB check found {len(missing)} missing PNGs in {rgb_dir}")
    print("Converting missing RGB inputs through lpbd.data.tif2rgb utilities...")

    convert_tif_folder_to_rgb(tif_dir, rgb_dir, overwrite=False, lower=lower, upper=upper)

    existing = existing_rgb_stems(rgb_dir)
    missing = sorted(expected - existing)
    if missing:
        sample = ", ".join(missing[:10])
        raise RuntimeError(
            f"RGB conversion finished but {len(missing)} PNGs are still missing. "
            f"Sample: {sample}"
        )

    print(f"RGB conversion complete: {len(existing & expected)}/{len(expected)} images exist")


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


def prepare_directories(output_root: Path) -> dict[str, Path]:
    annotations_dir = output_root / "annotations"
    images_root = output_root / "images"
    split_dirs = {split: images_root / split for split in sorted(VALID_SPLITS)}

    annotations_dir.mkdir(parents=True, exist_ok=True)
    for path in split_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return split_dirs


def stage_rgb_images(
    split_df: pd.DataFrame,
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

    for row in tqdm(split_df.itertuples(index=False), total=len(split_df), desc="Staging RGB images"):
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
        message = f"Missing {len(missing)} RGB images referenced by parquet. Sample: {missing[:10]}"
        if strict:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}")

    return counts, missing


def write_split_annotations(
    split_df: pd.DataFrame,
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
        split_rows = split_df[split_df["split"] == split]

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

    # Resolve the raw FTW input paths and the processed output locations
    raw_root = project_root / "data" / "raw" / "FTW_Vietnam"
    processed_root = project_root / "data" / "processed"
    rgb_dir = processed_root / "rgb" / args.window
    output_root = processed_root / args.output_name
    chips_path = raw_root / "chips_vietnam.parquet"
    instance_mask_dir = raw_root / "label_masks" / "instance"

    # Ensure the raw dataset exists, then create RGB PNGs if they are missing
    require_path(raw_root, "FTW Vietnam raw folder")
    require_path(instance_mask_dir, "FTW instance mask folder")
    ensure_rgb_images(raw_root, rgb_dir, args.window, lower=args.lower, upper=args.upper)

    # Read train/val/test assignments from the FTW chip metadata parquet
    split_df = load_split_table(chips_path)

    # Create the SAM 3 folder layout: annotations/ and images/{train,val,test}/
    split_dirs = prepare_directories(output_root)

    # Copy or move each RGB image into the split folder defined by the parquet
    counts, missing = stage_rgb_images(
        split_df,
        rgb_dir,
        split_dirs,
        move=args.move,
        overwrite=args.overwrite,
        strict=args.strict,
    )

    # Convert instance-mask TIF files into SAM 3/COCO-style JSON annotations
    annotation_counts = write_split_annotations(
        split_df,
        output_root=output_root,
        split_dirs=split_dirs,
        instance_mask_dir=instance_mask_dir,
        text_prompt=args.text_prompt,
        min_area=args.min_area,
        strict=args.strict,
    )

    print("SAM 3 FTW image dataset prepared")
    print(f"Output root: {output_root}")
    for split in sorted(counts):
        print(f"  {split}: {counts[split]} images, {annotation_counts[split]} annotations")
    if missing:
        print(f"  missing parquet entries without RGB: {len(missing)}")


if __name__ == "__main__":
    main()
