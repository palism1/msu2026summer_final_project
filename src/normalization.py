# FILE MAP | Pure-numpy input-normalization math, shared by the training transform
#   (src/data/transforms.py) and the zero-shot oracle path (src/models/zeroshot.py).
#   numpy ONLY — no torch, no albumentations — so it is unit-testable on any machine and
#   importable from both packages without a GPU stack.
#   [DO NOT TOUCH] min_max_normalize's formula: it reproduces MedSAM's own preprocessing
#   (bowang-lab/MedSAM: MedSAM_Inference.py, train_one_gpu.py) exactly.
"""Input normalization primitives (pure numpy)."""

from __future__ import annotations

import numpy as np

# ImageNet statistics == SAM's own Sam.preprocess constants / 255
# (pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Kept in sync with src/config.NORMALIZATION_CHOICES (config.py stays numpy-free by contract).
NORMALIZATION_CHOICES = ("imagenet", "minmax")

MINMAX_EPS = 1e-8   # MedSAM's np.clip(max - min, a_min=1e-8, a_max=None)


def check_normalization(name: str) -> str:
    if name not in NORMALIZATION_CHOICES:
        raise ValueError(f"normalization must be one of {NORMALIZATION_CHOICES}, got {name!r}")
    return name


def min_max_normalize(image: np.ndarray, eps: float = MINMAX_EPS) -> np.ndarray:
    """Scale one image to [0, 1] using its GLOBAL min/max over all pixels AND channels.

    Reproduces MedSAM's preprocessing:
        img = (img - img.min()) / np.clip(img.max() - img.min(), 1e-8, None)
    Per-image, NOT per-channel — a per-channel variant would alter the colour balance
    MedSAM's encoder was fit on. A constant image maps to zeros (never NaN). Output float32.

    NOTE (load-bearing for docs/MEDSAM_INVESTIGATION.md experiment C): this is exactly
    invariant to any global affine map y = a*x + b with a > 0, which is precisely what
    A.ColorJitter's brightness (a*x) and contrast (f*x + mean*(1-f)) are. Under "minmax"
    those two augmentations cannot reach the network except through uint8 clipping.
    """
    x = np.asarray(image, dtype=np.float32)
    lo = float(x.min())
    hi = float(x.max())
    return ((x - lo) / max(hi - lo, eps)).astype(np.float32)


def imagenet_normalize(image: np.ndarray) -> np.ndarray:
    """Reference implementation of (img/255 - mean)/std, for tests and analysis only.

    The training pipeline uses A.Normalize (unchanged); tests/test_transforms.py asserts the
    two agree. Kept here so torch-free tests can reason about both normalizations.
    """
    x = np.asarray(image, dtype=np.float32) / 255.0
    return ((x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)).astype(np.float32)
