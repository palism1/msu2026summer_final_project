# FILE MAP | Unit tests for the PURE normalization math in src/normalization.py.
#   No torch/albumentations needed. Covers: the min-max formula matches MedSAM's own
#   preprocessing exactly, the ImageNet reference matches SAM's pixel stats, and the
#   confound this module exists to document — min-max normalization is exactly invariant
#   to A.ColorJitter's brightness/contrast, ImageNet normalization is not. See
#   docs/MEDSAM_INVESTIGATION.md experiment C for why that asymmetry matters.
"""GPU-free tests for input normalization primitives."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.normalization import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NORMALIZATION_CHOICES,
    check_normalization,
    imagenet_normalize,
    min_max_normalize,
)

rng = np.random.default_rng(0)
IMG = rng.integers(0, 256, (24, 20, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Formula / shape
# ---------------------------------------------------------------------------

def test_min_max_matches_medsam_formula():
    x = IMG.astype(np.float32)
    expected = (x - x.min()) / np.clip(x.max() - x.min(), a_min=1e-8, a_max=None)
    np.testing.assert_allclose(min_max_normalize(IMG), expected, atol=1e-6)


def test_min_max_is_global_not_per_channel():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 0] = rng.integers(0, 256, (10, 10))       # ch0 spans 0-255
    img[..., 1] = rng.integers(100, 121, (10, 10))     # ch1 spans 100-120
    img[..., 2] = 50
    out = min_max_normalize(img)
    assert out[..., 1].min() > 0.0
    assert out[..., 1].max() < 1.0


def test_min_max_output_range_and_dtype():
    out = min_max_normalize(IMG)
    assert out.dtype == np.float32
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_constant_image_maps_to_zeros_not_nan():
    img = np.full((5, 5, 3), 100, dtype=np.uint8)
    out = min_max_normalize(img)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, 0.0)


def test_imagenet_constants_are_sam_pixel_stats_over_255():
    sam_pixel_mean = np.array([123.675, 116.28, 103.53]) / 255.0
    sam_pixel_std = np.array([58.395, 57.12, 57.375]) / 255.0
    np.testing.assert_allclose(IMAGENET_MEAN, sam_pixel_mean, atol=1e-4)
    np.testing.assert_allclose(IMAGENET_STD, sam_pixel_std, atol=1e-4)


# ---------------------------------------------------------------------------
# Confound tests — the BLOCKING-1 mechanism guard (torch-free)
# ---------------------------------------------------------------------------

def _brightness(img, a):            # adjust_brightness_torchvision: multiply(img, a)
    return np.clip(np.asarray(img, np.float32) * a, 0, 255)


def _contrast(img, f):              # adjust_contrast_torchvision: multiply_add(img, f, mean*(1-f))
    x = np.asarray(img, np.float32)
    mean = x.mean(axis=2).mean()    # grayscale mean, a scalar
    return np.clip(x * f + mean * (1 - f), 0, 255)


def _saturation(img, f):            # adjust_saturation_torchvision: addWeighted(img, f, gray, 1-f)
    x = np.asarray(img, np.float32)
    gray = x.mean(axis=2, keepdims=True).repeat(3, axis=2)
    return np.clip(x * f + gray * (1 - f), 0, 255)


def test_min_max_exactly_cancels_brightness_jitter():
    a = 0.8  # no clipping at this factor
    np.testing.assert_allclose(
        min_max_normalize(_brightness(IMG, a)), min_max_normalize(IMG), atol=1e-6
    )


def test_min_max_exactly_cancels_contrast_jitter():
    f = 0.8
    np.testing.assert_allclose(
        min_max_normalize(_contrast(IMG, f)), min_max_normalize(IMG), atol=1e-6
    )


def test_imagenet_normalization_does_not_cancel_the_same_jitter():
    # THIS IS THE ASYMMETRY BLOCKING 1 IS ABOUT, and it runs locally.
    jittered = _brightness(IMG, 0.8)
    imagenet_diff = np.mean(np.abs(imagenet_normalize(jittered) - imagenet_normalize(IMG)))
    minmax_diff = np.mean(np.abs(min_max_normalize(jittered) - min_max_normalize(IMG)))
    assert imagenet_diff > 0.3
    assert minmax_diff < 1e-5


def test_min_max_does_not_cancel_saturation():
    jittered = _saturation(IMG, 0.6)
    minmax_diff = np.mean(np.abs(min_max_normalize(jittered) - min_max_normalize(IMG)))
    assert minmax_diff > 0.005


def test_uint8_rounding_leaves_only_a_tiny_residue():
    jittered_uint8 = _brightness(IMG, 0.8).astype(np.uint8)
    minmax_diff = np.mean(np.abs(min_max_normalize(jittered_uint8) - min_max_normalize(IMG)))
    assert minmax_diff < 0.005


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def test_check_normalization_rejects_unknown():
    with pytest.raises(ValueError):
        check_normalization("zscore")


def test_normalization_choices_agree_with_config():
    from src.config import NORMALIZATION_CHOICES as CONFIG_CHOICES

    assert tuple(NORMALIZATION_CHOICES) == tuple(CONFIG_CHOICES)


def test_pure_modules_import_without_torch_or_albumentations():
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys, src.normalization, src.models.zeroshot; "
        "assert 'torch' not in sys.modules; "
        "assert 'albumentations' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
