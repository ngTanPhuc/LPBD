import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "sam3_ftw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam3_ftw_testset_eval"

import sys

sys.path.insert(0, str(SRC_ROOT))

from test_sam3_ckpt import load_model, preprocess_image, run_inference, visualize
from lpbd.utils.metrics import calculate_field_iou, calculate_field_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM 3 LPBD inference on the sam3_ftw test split and report field metrics."
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        type=Path,
        help="Fine-tuned checkpoint path",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        type=Path,
        help="sam3_ftw data root containing images/test and annotations/test.json",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory for overlays and metrics CSV",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip saving per-image prediction overlays",
    )
    return parser.parse_args()


def load_test_annotations(annotation_path: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    images = sorted(payload.get("images", []), key=lambda x: x["file_name"])
    anns_by_image_id: dict[int, list[dict]] = {image["id"]: [] for image in images}
    for ann in payload.get("annotations", []):
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    return images, anns_by_image_id


def rasterize_field_mask(image_info: dict, annotations: list[dict]) -> np.ndarray:
    width = int(image_info["width"])
    height = int(image_info["height"])
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for ann in annotations:
        segmentation = ann.get("segmentation") or []
        if isinstance(segmentation, dict):
            raise ValueError(
                "RLE segmentations are not supported by this lightweight evaluator. "
                f"Image id: {image_info['id']}"
            )
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=1)

    return np.asarray(mask_img, dtype=np.uint8)


def write_metrics_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "num_gt_instances",
        "num_pred_instances",
        "field_iou",
        "field_recall",
        "mean_score",
        "max_score",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    overlay_dir = output_dir / "overlays"
    metrics_csv_path = output_dir / "metrics.csv"

    images, anns_by_image_id = load_test_annotations(data_root / "annotations" / "train.json")
    if not images:
        raise RuntimeError(f"No test images found in {data_root / 'annotations' / 'test.json'}")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt_path, device)

    rows = []
    for image_idx, image_info in enumerate(images, start=1):
        image_path = data_root / image_info["file_name"]
        print(f"[{image_idx}/{len(images)}] {image_path}")

        img_batch, image_np, original_size = preprocess_image(image_path, device)
        predictions = run_inference(model, img_batch, original_size, device)
        gt_mask = rasterize_field_mask(image_info, anns_by_image_id.get(image_info["id"], []))

        pred_masks = predictions["masks"] > 0.2
        field_iou = calculate_field_iou(pred_masks, gt_mask, field_values=(1,))
        field_recall = calculate_field_recall(pred_masks, gt_mask, field_values=(1,))
        scores = predictions["scores"].numpy()

        if not args.no_overlays:
            overlay_path = overlay_dir / f"{Path(image_info['file_name']).stem}_overlay.png"
            visualize(image_np, predictions, overlay_path)

        row = {
            "file_name": image_info["file_name"],
            "num_gt_instances": len(anns_by_image_id.get(image_info["id"], [])),
            "num_pred_instances": len(predictions["scores"]),
            "field_iou": field_iou,
            "field_recall": field_recall,
            "mean_score": float(scores.mean()) if scores.size else 0.0,
            "max_score": float(scores.max()) if scores.size else 0.0,
        }
        rows.append(row)
        print(
            "  "
            f"IoU={field_iou:.4f} "
            f"Recall={field_recall:.4f} "
            f"GT={row['num_gt_instances']} "
            f"Pred={row['num_pred_instances']}"
        )

    write_metrics_csv(rows, metrics_csv_path)

    mean_iou = float(np.mean([row["field_iou"] for row in rows]))
    mean_recall = float(np.mean([row["field_recall"] for row in rows]))
    print("=" * 80)
    print(f"Images evaluated: {len(rows)}")
    print(f"Mean field IoU: {mean_iou:.4f}")
    print(f"Mean field recall: {mean_recall:.4f}")
    print(f"Metrics CSV: {metrics_csv_path}")
    if not args.no_overlays:
        print(f"Overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
