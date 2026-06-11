# !!! DO NOT run this script directly. It is meant to be imported and used by the config file (e.g., configs/sam3/lpbd_ftw.yaml).
# So after you clone the SAM 3 repo, copy this file into the `externals/sam3/sam3/model_builder.py` file, and then reference the `build_lpbd_sam3_image_model` function in your config file.

from sam3.model_builder import build_sam3_image_model as _build_sam3_image_model


def build_lpbd_sam3_image_model(
    *args,
    freeze_backbone: bool = True,
    freeze_prefixes=None,
    **kwargs,
):
    model = _build_sam3_image_model(*args, **kwargs)

    if freeze_prefixes is None:
        freeze_prefixes = (
            "backbone.vision_backbone.trunk",
            "backbone.language_backbone",
        )

    if freeze_backbone:
        frozen_params = 0
        frozen_elems = 0

        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_prefixes):
                param.requires_grad_(False)
                frozen_params += 1
                frozen_elems += param.numel()

        print("=" * 80)
        print("[LPBD] Frozen SAM 3 prefixes:")
        for prefix in freeze_prefixes:
            print(f"  - {prefix}")
        print(f"[LPBD] Frozen parameter tensors: {frozen_params}")
        print(f"[LPBD] Frozen parameter elements: {frozen_elems:,}")
        print("=" * 80)

    total_elems = sum(p.numel() for p in model.parameters())
    trainable_elems = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80)
    print(f"[LPBD] Total parameters:     {total_elems:,}")
    print(f"[LPBD] Trainable parameters: {trainable_elems:,}")
    print(f"[LPBD] Trainable ratio:      {100 * trainable_elems / total_elems:.2f}%")
    print("=" * 80)

    return model