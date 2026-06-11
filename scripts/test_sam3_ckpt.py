import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = PROJECT_ROOT / "external" / "sam3"

sys.path.insert(0, str(SAM3_ROOT))

from sam3.model.data_misc import (  # noqa: E402
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)
from sam3.model_builder_lpbd import build_lpbd_sam3_image_model  # noqa: E402

RESOLUTION = 1008
TEXT_PROMPT = "agricultural field"
SCORE_THRESHOLD = 0.5
MAX_DETECTIONS = 200
MASK_ALPHA = 0.45

MASK_COLORS = np.array(
    [
        (230, 57, 70),
        (42, 157, 143),
        (38, 70, 83),
        (244, 162, 97),
        (233, 196, 106),
        (29, 53, 87),
        (69, 123, 157),
        (168, 218, 220),
        (241, 91, 181),
        (0, 166, 251),
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fine-tuned SAM 3 LPBD checkpoint on one image and save an overlay."
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        type=Path,
        help="Fine-tuned checkpoint path",
    )
    parser.add_argument(
        "--test-img-path",
        required=True,
        type=Path,
        help="Input test image path",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Output visualization image path",
    )
    return parser.parse_args()


def load_model(ckpt_path: Path, device: str) -> torch.nn.Module:
    base_ckpt_path = PROJECT_ROOT / "models" / "pretrained" / "sam3" / "sam3.pt"
    bpe_path = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ckpt_path}")
    if not base_ckpt_path.is_file():
        raise FileNotFoundError(f"Base SAM 3 checkpoint not found: {base_ckpt_path}")
    if not bpe_path.is_file():
        raise FileNotFoundError(f"BPE vocabulary not found: {bpe_path}")

    print("Device:", device)
    print("Loading base model:", base_ckpt_path)
    model = build_lpbd_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=str(base_ckpt_path),
        load_from_HF=False,
        enable_segmentation=True,
        device=device,
        eval_mode=True,
        freeze_backbone=True,
    )

    print("Loading fine-tuned checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Epoch:", ckpt.get("epoch") if isinstance(ckpt, dict) else None)
    print("Steps:", ckpt.get("steps") if isinstance(ckpt, dict) else None)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.eval()
    return model


def preprocess_image(image_path: Path, device: str) -> tuple[torch.Tensor, np.ndarray, tuple[int, int]]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image_for_model = image.resize((RESOLUTION, RESOLUTION), Image.BILINEAR)

    image_np = np.asarray(image, dtype=np.uint8)
    x = np.asarray(image_for_model, dtype=np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).to(device)
    return x, image_np, (orig_h, orig_w)


def make_batched_datapoint(img_batch: torch.Tensor, original_size: tuple[int, int]) -> BatchedDatapoint:
    device = img_batch.device
    find_input = FindStage(
        img_ids=torch.tensor([0], dtype=torch.long, device=device),
        text_ids=torch.tensor([0], dtype=torch.long, device=device),
        input_boxes=torch.zeros((0, 1, 4), dtype=torch.float32, device=device),
        input_boxes_mask=torch.ones((1, 0), dtype=torch.bool, device=device),
        input_boxes_label=torch.zeros((0, 1), dtype=torch.long, device=device),
        input_points=torch.empty((1, 0, 257), dtype=torch.float32, device=device),
        input_points_mask=torch.ones((1, 0), dtype=torch.bool, device=device),
        object_ids=[[]],
    )
    find_target = BatchedFindTarget(
        num_boxes=torch.tensor([0], dtype=torch.long, device=device),
        boxes=torch.zeros((0, 4), dtype=torch.float32, device=device),
        boxes_padded=torch.zeros((1, 0, 4), dtype=torch.float32, device=device),
        repeated_boxes=torch.zeros((0, 4), dtype=torch.float32, device=device),
        segments=torch.zeros((0, RESOLUTION, RESOLUTION), dtype=torch.bool, device=device),
        semantic_segments=torch.zeros((0, RESOLUTION, RESOLUTION), dtype=torch.bool, device=device),
        is_valid_segment=torch.zeros((0,), dtype=torch.bool, device=device),
        is_exhaustive=torch.tensor([False], dtype=torch.bool, device=device),
        object_ids=torch.zeros((0,), dtype=torch.long, device=device),
        object_ids_padded=torch.zeros((1, 0), dtype=torch.long, device=device),
    )
    metadata = BatchedInferenceMetadata(
        coco_image_id=torch.tensor([0], dtype=torch.long, device=device),
        original_image_id=torch.tensor([0], dtype=torch.long, device=device),
        original_category_id=torch.tensor([1], dtype=torch.int, device=device),
        original_size=torch.tensor([original_size], dtype=torch.long, device=device),
        object_id=torch.tensor([-1], dtype=torch.long, device=device),
        frame_index=torch.tensor([0], dtype=torch.long, device=device),
        is_conditioning_only=[False],
    )
    return BatchedDatapoint(
        img_batch=img_batch,
        find_text_batch=[TEXT_PROMPT],
        find_inputs=[find_input],
        find_targets=[find_target],
        find_metadatas=[metadata],
        raw_images=None,
    )


def run_inference(
    model: torch.nn.Module,
    img_batch: torch.Tensor,
    original_size: tuple[int, int],
    device: str,
) -> dict[str, torch.Tensor]:
    datapoint = make_batched_datapoint(img_batch, original_size)
    autocast_enabled = device == "cuda"

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            model_output = model(datapoint)

    out = model_output[0]
    scores = out["pred_logits"].sigmoid().amax(dim=-1)[0]
    if "presence_logit_dec" in out:
        scores = scores * out["presence_logit_dec"].sigmoid()[0]

    boxes = out["pred_boxes_xyxy"][0].detach().float().cpu()
    masks = out["pred_masks"].detach().float()
    masks = F.interpolate(masks, size=original_size, mode="bilinear", align_corners=False)[0]
    masks = masks.sigmoid().cpu()
    scores = scores.detach().float().cpu()

    keep = scores >= SCORE_THRESHOLD
    if keep.any():
        kept_idx = torch.nonzero(keep, as_tuple=False).flatten()
    else:
        kept_idx = torch.topk(scores, k=1).indices

    kept_scores = scores[kept_idx]
    order = torch.argsort(kept_scores, descending=True)[:MAX_DETECTIONS]
    kept_idx = kept_idx[order]

    return {
        "scores": scores[kept_idx],
        "boxes": boxes[kept_idx],
        "masks": masks[kept_idx],
    }


def visualize(
    image_np: np.ndarray,
    predictions: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = image_np.shape[:2]
    overlay = image_np.astype(np.float32).copy()
    masks = predictions["masks"].numpy() > 0.5
    scores = predictions["scores"].numpy()
    boxes = predictions["boxes"].numpy()

    for idx, mask in enumerate(masks):
        color = MASK_COLORS[idx % len(MASK_COLORS)]
        overlay[mask] = overlay[mask] * (1.0 - MASK_ALPHA) + color * MASK_ALPHA

    output = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)

    for idx, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
        color = tuple(MASK_COLORS[idx % len(MASK_COLORS)].astype(np.uint8).tolist())
        x0, y0, x1, y1 = box * np.array([w, h, w, h], dtype=np.float32)
        x0, y0, x1, y1 = np.clip([x0, y0, x1, y1], [0, 0, 0, 0], [w, h, w, h])
        draw.rectangle(
            (float(x0), float(y0), float(x1), float(y1)),
            outline=color,
            width=max(1, round(min(w, h) / 256)),
        )
        if mask.any():
            ys, xs = np.where(mask)
            label = f"{idx + 1}:{score:.2f}"
            tx = int(xs.mean())
            ty = int(ys.mean())
            text_box = draw.textbbox((tx, ty), label)
            pad = 2
            bg_box = (
                text_box[0] - pad,
                text_box[1] - pad,
                text_box[2] + pad,
                text_box[3] + pad,
            )
            draw.rectangle(bg_box, fill=color)
            draw.text((tx, ty), label, fill=(255, 255, 255))

    output.save(output_path)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.ckpt_path, device)
    img_batch, image_np, original_size = preprocess_image(args.test_img_path, device)
    predictions = run_inference(model, img_batch, original_size, device)
    visualize(image_np, predictions, args.output_path)

    print(f"Saved visualization: {args.output_path}")
    print(f"Detections visualized: {len(predictions['scores'])}")
    print("Scores:", ", ".join(f"{s:.3f}" for s in predictions["scores"].tolist()))


if __name__ == "__main__":
    main()
