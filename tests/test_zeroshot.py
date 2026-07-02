# FILE MAP | GPU-free unit tests for the zero-shot baselines' PURE prompt-derivation math
#   (src/models/zeroshot.py). No torch, no segment-anything, no GPU — thanks to the lazy
#   src/models/__init__.py, importing box_from_mask/point_from_mask needs only numpy.
#   [TWEAK] add cases freely; the empty-mask -> None invariant is load-bearing (drives the
#   "empty GT -> empty prediction" path in ZeroShotSAM.predict_prob).
"""GPU-free tests for zero-shot prompt derivation."""

import numpy as np
import pytest

from src.models.zeroshot import box_from_mask, is_zeroshot, point_from_mask


@pytest.fixture
def square_mask():
    m = np.zeros((32, 32), np.float32)
    m[8:24, 10:20] = 1.0   # rows 8..23 (y), cols 10..19 (x)
    return m


def test_box_from_mask_tight_xyxy(square_mask):
    box = box_from_mask(square_mask, padding=0)
    # XYXY: x0,y0,x1,y1 = col_min, row_min, col_max, row_max
    assert box.tolist() == [10.0, 8.0, 19.0, 23.0]


def test_box_padding_clamps_to_bounds(square_mask):
    # Large padding must not run past the image edges (0..31).
    box = box_from_mask(square_mask, padding=100)
    assert box.tolist() == [0.0, 0.0, 31.0, 31.0]


def test_box_padding_expands_within_bounds(square_mask):
    box = box_from_mask(square_mask, padding=3)
    assert box.tolist() == [7.0, 5.0, 22.0, 26.0]


def test_box_empty_mask_returns_none():
    assert box_from_mask(np.zeros((16, 16), np.float32)) is None


def test_box_single_pixel():
    m = np.zeros((16, 16), np.float32)
    m[5, 7] = 1.0
    assert box_from_mask(m, padding=0).tolist() == [7.0, 5.0, 7.0, 5.0]


def test_box_ignores_soft_values_below_threshold():
    # 0.4 is background (threshold is 0.5); only the 1.0 block counts.
    m = np.full((16, 16), 0.4, np.float32)
    m[2:5, 2:5] = 1.0
    assert box_from_mask(m, padding=0).tolist() == [2.0, 2.0, 4.0, 4.0]


def test_point_from_mask_is_centroid(square_mask):
    pt = point_from_mask(square_mask)
    assert pt.shape == (1, 2)
    # centroid: x over cols 10..19 -> 14.5 ; y over rows 8..23 -> 15.5
    assert pt[0].tolist() == [pytest.approx(14.5), pytest.approx(15.5)]


def test_point_empty_mask_returns_none():
    assert point_from_mask(np.zeros((16, 16), np.float32)) is None


def test_is_zeroshot_false_for_plain_objects():
    assert is_zeroshot(object()) is False
    assert is_zeroshot(None) is False
