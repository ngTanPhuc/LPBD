"""Segmentation metrics and mask helpers."""

from pathlib import Path
import numpy as np


def masks_to_binary_field_mask(masks, target_shape):
    """Union SAM3 instance masks into one binary field mask."""
    if masks is None:
        return np.zeros(target_shape, dtype=bool)

    if hasattr(masks, "detach") and hasattr(masks, "cpu"):
        masks = masks.detach().cpu().numpy()
    elif isinstance(masks, (list, tuple)):
        if len(masks) == 0:
            return np.zeros(target_shape, dtype=bool)
        masks = [
            m.detach().cpu().numpy() if hasattr(m, "detach") and hasattr(m, "cpu") else np.asarray(m)
            for m in masks
        ]
        masks = np.stack(masks, axis=0)
    else:
        masks = np.asarray(masks)

    if masks.size == 0:
        return np.zeros(target_shape, dtype=bool)

    if masks.ndim == 2:
        pred_mask = masks.astype(bool)
    elif masks.ndim == 3:
        pred_mask = np.any(masks.astype(bool), axis=0)
    else:
        raise ValueError(f"Expected masks with shape [H, W] or [N, H, W], got {masks.shape}")

    if pred_mask.shape != tuple(target_shape):
        raise ValueError(f"Predicted mask shape {pred_mask.shape} does not match target shape {target_shape}")

    return pred_mask


def calculate_field_iou(pred_masks, gt_semantic_mask, field_values=(1, 2), empty_score=1.0):
    """Calculate binary Field IoU between predicted masks and an FTW semantic mask.

    Args:
        pred_masks: Predicted masks, shape [N, H, W] or [H, W].
        gt_semantic_mask: FTW semantic mask, shape [H, W]. Values 1 and 2 are
            treated as field pixels by default.
        field_values: Semantic class values counted as fields. Use None to treat
            all non-zero values as fields.
        empty_score: IoU returned when prediction and ground truth are both empty.
    """
    if hasattr(gt_semantic_mask, "detach") and hasattr(gt_semantic_mask, "cpu"):
        gt_semantic_mask = gt_semantic_mask.detach().cpu().numpy()

    gt_semantic_mask = np.asarray(gt_semantic_mask)
    gt_field_mask = gt_semantic_mask > 0 if field_values is None else np.isin(gt_semantic_mask, field_values)
    pred_field_mask = masks_to_binary_field_mask(pred_masks, gt_field_mask.shape)

    intersection = np.logical_and(pred_field_mask, gt_field_mask).sum()
    union = np.logical_or(pred_field_mask, gt_field_mask).sum()

    if union == 0:
        return float(empty_score)

    return float(intersection / union)


def calculate_field_recall(pred_masks, gt_semantic_mask, field_values=(1, 2), empty_score=1.0):
    """Calculate binary field recall between predicted masks and an FTW semantic mask.

    Recall is true positive field pixels divided by all ground-truth field pixels.
    """
    if hasattr(gt_semantic_mask, "detach") and hasattr(gt_semantic_mask, "cpu"):
        gt_semantic_mask = gt_semantic_mask.detach().cpu().numpy()

    gt_semantic_mask = np.asarray(gt_semantic_mask)
    gt_field_mask = gt_semantic_mask > 0 if field_values is None else np.isin(gt_semantic_mask, field_values)
    pred_field_mask = masks_to_binary_field_mask(pred_masks, gt_field_mask.shape)

    gt_pixels = gt_field_mask.sum()
    if gt_pixels == 0:
        return float(empty_score)

    true_positive_pixels = np.logical_and(pred_field_mask, gt_field_mask).sum()
    return float(true_positive_pixels / gt_pixels)


def load_semantic_mask_for_rgb(rgb_path, semantic_dir):
    """Load the FTW semantic mask that matches a converted RGB PNG."""
    import rasterio

    mask_path = Path(semantic_dir) / f"{Path(rgb_path).stem}.tif"
    with rasterio.open(mask_path) as src:
        return src.read(1)
