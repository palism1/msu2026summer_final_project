# FILE MAP | Unit tests for the PURE evaluation math in src/metrics/segmentation.py.
#   No torch, no GPU — these are the "eval is testable without a GPU" guarantee.
#   [TWEAK] add cases freely; [DO NOT TOUCH] the perfect/empty invariants below.
"""GPU-free tests for the segmentation metrics."""

import numpy as np
import pytest

from src.metrics import (
    MetricTracker,
    dice_score,
    e_measure,
    iou_score,
    mae_score,
    s_measure,
    weighted_f_measure,
)


@pytest.fixture
def disk_mask():
    gt = np.zeros((32, 32), np.float32)
    gt[8:24, 8:24] = 1.0
    return gt


def test_perfect_prediction_scores_one(disk_mask):
    assert dice_score(disk_mask, disk_mask) == pytest.approx(1.0, abs=1e-4)
    assert iou_score(disk_mask, disk_mask) == pytest.approx(1.0, abs=1e-4)
    assert mae_score(disk_mask, disk_mask) == pytest.approx(0.0, abs=1e-6)
    assert weighted_f_measure(disk_mask, disk_mask) == pytest.approx(1.0, abs=1e-3)


def test_disjoint_prediction_scores_zero(disk_mask):
    pred = np.zeros_like(disk_mask)
    pred[0:4, 0:4] = 1.0  # no overlap with the central disk
    assert dice_score(pred, disk_mask) == pytest.approx(0.0, abs=1e-4)
    assert iou_score(pred, disk_mask) == pytest.approx(0.0, abs=1e-4)


def test_dice_known_half_overlap():
    gt = np.zeros((10, 10), np.float32); gt[:, :5] = 1.0     # 50 px
    pred = np.zeros((10, 10), np.float32); pred[:, 2:7] = 1.0  # 50 px, 30 overlap
    # Dice = 2*30 / (50+50) = 0.6 ; IoU = 30 / (100-30) = 3/7
    assert dice_score(pred, gt) == pytest.approx(0.6, abs=1e-3)
    assert iou_score(pred, gt) == pytest.approx(3 / 7, abs=1e-3)


def test_empty_gt_and_empty_pred_edge_cases():
    empty = np.zeros((16, 16), np.float32)
    # weighted F: both empty -> perfect; s/e measures stay bounded in [0,1]
    assert weighted_f_measure(empty, empty) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= s_measure(empty, empty) <= 1.0
    assert 0.0 <= e_measure(empty, empty) <= 1.0


def test_e_measure_within_unit_range(disk_mask):
    rng = np.random.default_rng(0)
    for _ in range(20):
        pred = rng.uniform(0, 1, disk_mask.shape).astype(np.float32)
        val = e_measure(pred, disk_mask)
        assert 0.0 <= val <= 1.0


def test_tracker_reports_all_six_means(disk_mask):
    tracker = MetricTracker()
    for _ in range(5):
        tracker.update(disk_mask, disk_mask)  # perfect each time
    out = tracker.compute()
    assert set(out) == {"dice", "iou", "mae", "wfm", "sm", "em"}
    assert out["dice"] == pytest.approx(1.0, abs=1e-3)
    assert out["mae"] == pytest.approx(0.0, abs=1e-6)


def test_tracker_thresholds_probabilities(disk_mask):
    # A soft prediction just under/over 0.5 should binarize correctly.
    tracker = MetricTracker()
    soft = np.where(disk_mask > 0, 0.9, 0.1).astype(np.float32)
    tracker.update(soft, disk_mask)
    assert tracker.compute()["dice"] == pytest.approx(1.0, abs=1e-3)
