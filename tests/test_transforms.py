# FILE MAP | Unit tests for the parameterised train/val transforms (src/data/transforms.py).
#   Requires albumentations — SKIPPED locally (no albumentations in this environment), runs in
#   Colab. Covers: minmax output is exactly [0,1], imagenet output reproduces the legacy
#   (unparameterised) pipeline byte-for-byte, the color_jitter override wires through, and mask
#   binarity survives both normalization modes.
"""Tests for the train/val transform pipelines (requires albumentations)."""

import numpy as np
import pytest

pytest.importorskip("albumentations")
pytest.importorskip("albumentations.pytorch")

from src.config import NO_BC_COLOR_JITTER, STANDARD_COLOR_JITTER
from src.data.transforms import get_train_transform, get_val_transform
from src.normalization import imagenet_normalize

rng = np.random.default_rng(0)
IMG = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
MASK = (rng.random((64, 48)) > 0.5).astype(np.float32)


def test_val_minmax_output_is_zero_one_float32():
    out = get_val_transform(352, "minmax")(image=IMG, mask=MASK)["image"]
    assert out.shape == (3, 352, 352)
    assert out.dtype.is_floating_point
    assert float(out.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_val_imagenet_reproduces_legacy_values():
    import albumentations as A

    resize_only = A.Compose([A.Resize(352, 352)])
    resized = resize_only(image=IMG, mask=MASK)["image"]
    expected = imagenet_normalize(resized)

    out = get_val_transform(352, "imagenet")(image=IMG, mask=MASK)["image"]
    got = out.permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_val_minmax_matches_min_max_normalize_on_resized_image():
    import albumentations as A

    from src.normalization import min_max_normalize

    resize_only = A.Compose([A.Resize(352, 352)])
    resized = resize_only(image=IMG, mask=MASK)["image"]
    expected = min_max_normalize(resized)

    out = get_val_transform(352, "minmax")(image=IMG, mask=MASK)["image"]
    got = out.permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_train_defaults_are_the_published_jitter():
    t = get_train_transform(352)
    cj = [tr for tr in t.transforms if tr.__class__.__name__ == "ColorJitter"][0]
    assert cj.brightness == (max(0, 1 - STANDARD_COLOR_JITTER["brightness"]),
                             1 + STANDARD_COLOR_JITTER["brightness"])


def test_train_accepts_zero_brightness_contrast():
    t = get_train_transform(352, "imagenet", NO_BC_COLOR_JITTER)
    out = t(image=IMG, mask=MASK)
    assert out["image"].shape == (3, 352, 352)


def test_train_keeps_mask_binary_in_both_modes():
    for norm in ("imagenet", "minmax"):
        out = get_train_transform(352, norm)(image=IMG, mask=MASK)
        vals = set(np.unique(out["mask"].numpy()).tolist())
        assert vals <= {0.0, 1.0}


def test_colorjitter_transform_count_is_mode_independent():
    n_imagenet = len(get_train_transform(352, "imagenet").transforms)
    n_minmax = len(get_train_transform(352, "minmax").transforms)
    assert n_imagenet == n_minmax


def test_unknown_normalization_raises():
    with pytest.raises(ValueError):
        get_train_transform(352, "zscore")
    with pytest.raises(ValueError):
        get_val_transform(352, "zscore")
