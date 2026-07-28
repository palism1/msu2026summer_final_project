# FILE MAP | Albumentations train/val pipelines. Two things are now parameterised, and BOTH
#   default to the values every published run used:
#     - normalization: "imagenet" for U-Net + SAM ViT-H/ViT-B (SAM's own Sam.preprocess is the
#       same constants x255); "minmax" for MedSAM, fine-tuned on per-image [0,1] inputs.
#     - color_jitter: the photometric augmentation policy. Parameterised ONLY because min-max
#       normalization is affine-invariant, so brightness/contrast cannot reach a min-max model;
#       the medsam_ctrl arm needs an ImageNet-normalized run with them disabled to match.
#   Math lives in src/normalization.py (pure numpy, unit-tested); this file only wires it in.
#   [DO NOT TOUCH] the defaults. Protocol per model is bound in src/config.MODEL_SPECS.
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import STANDARD_COLOR_JITTER
from src.normalization import (
    IMAGENET_MEAN, IMAGENET_STD, check_normalization, min_max_normalize,
)


def _min_max_image(image, **params):
    """A.Lambda image callback (Lambda calls fn(img, **params))."""
    return min_max_normalize(image)


def _normalize_op(normalization: str):
    """One transform either way, at the same pipeline index, so swapping modes neither adds
    nor removes an augmentation RNG draw."""
    check_normalization(normalization)
    if normalization == "minmax":
        # Deliberately NOT A.Normalize(normalization="min_max"): that kwarg postdates the pinned
        # albumentations>=1.3.0 and albucore implements (max-min+1e-4) rather than MedSAM's
        # clip(max-min, 1e-8). A.Lambda keeps the formula exact and version-proof.
        return A.Lambda(image=_min_max_image, name="min_max_normalize", p=1.0)
    return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _color_jitter_op(color_jitter: dict | None):
    cj = STANDARD_COLOR_JITTER if color_jitter is None else color_jitter
    # brightness/contrast of 0.0 -> factor range [1, 1] (identity) while keeping the transform
    # in the pipeline, so the number of RNG draws is unchanged. If a future albumentations
    # rejects 0.0, pass the explicit tuple (1.0, 1.0) instead — same semantics.
    return A.ColorJitter(
        brightness=cj["brightness"], contrast=cj["contrast"],
        saturation=cj["saturation"], hue=cj["hue"], p=cj["p"],
    )


def get_train_transform(img_size: int = 352, normalization: str = "imagenet",
                        color_jitter: dict | None = None) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        _color_jitter_op(color_jitter),
        A.Resize(img_size, img_size),  # always last spatial op to guarantee fixed output size
        _normalize_op(normalization),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 352, normalization: str = "imagenet") -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        _normalize_op(normalization),
        ToTensorV2(),
    ])
