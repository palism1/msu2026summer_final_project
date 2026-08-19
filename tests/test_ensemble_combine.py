# FILE MAP | Unit tests for pure ensemble math (src/ensemble/combine.py): weighted averaging,
#   the six-metric evaluation shim, and the derived-payload metadata mergers. No torch, no GPU.
"""GPU-free tests for ensemble combination and evaluation."""

import numpy as np
import pytest

from src.ensemble.combine import (
    average_probs,
    calibration_stats,
    evaluate_stack,
    merge_backbones,
    merge_normalization,
    normalize_weights,
)
from src.metrics import MetricTracker


def test_average_probs_single_member_returns_it_unchanged():
    p = np.random.default_rng(0).random((3, 4, 4)).astype(np.float32)
    out = average_probs([p], [1.0])
    assert np.array_equal(out, p)


def test_average_probs_uniform_equals_arithmetic_mean():
    rng = np.random.default_rng(1)
    a = rng.random((2, 4, 4)).astype(np.float32)
    b = rng.random((2, 4, 4)).astype(np.float32)
    out = average_probs([a, b], [0.5, 0.5])
    assert np.allclose(out, (a + b) / 2, atol=1e-6)


def test_normalize_weights_rejects_negative():
    with pytest.raises(ValueError):
        normalize_weights([0.5, -0.5])


def test_normalize_weights_rejects_all_zero():
    with pytest.raises(ValueError):
        normalize_weights([0.0, 0.0])


def test_average_probs_raises_on_shape_mismatch():
    a = np.zeros((2, 4, 4), np.float32)
    b = np.zeros((2, 5, 5), np.float32)
    with pytest.raises(ValueError):
        average_probs([a, b], [0.5, 0.5])


def test_evaluate_stack_perfect_prediction_dice_above_0999():
    gt = (np.random.default_rng(2).random((5, 16, 16)) > 0.5).astype(np.float32)
    probs = gt.copy()
    result = evaluate_stack(probs, gt, MetricTracker)
    assert result["dice"] > 0.999


def test_evaluate_stack_matches_manual_metric_tracker_loop():
    rng = np.random.default_rng(3)
    probs = rng.random((5, 16, 16)).astype(np.float32)
    gt = (rng.random((5, 16, 16)) > 0.5).astype(np.float32)

    result = evaluate_stack(probs, gt, MetricTracker)

    manual = MetricTracker()
    for i in range(5):
        manual.update(probs[i], gt[i])
    expected = manual.compute()

    for key in ("dice", "iou", "mae", "wfm", "sm", "em"):
        assert result[key] == pytest.approx(expected[key])


def test_merge_normalization_all_agree():
    assert merge_normalization(["imagenet", "imagenet", "imagenet"]) == "imagenet"


def test_merge_normalization_disagree_is_mixed():
    assert merge_normalization(["imagenet", "minmax"]) == "mixed"


def test_merge_backbones_joins_with_plus():
    assert merge_backbones(["resnet34", "vit_h", "vit_b"]) == "resnet34+vit_h+vit_b"


def test_calibration_stats_shape():
    probs = np.array([0.5, 0.5, 0.0, 1.0], np.float32).reshape(1, 2, 2)
    stats = calibration_stats(probs)
    assert set(stats) == {"mean_prob", "frac_uncertain"}
    assert stats["mean_prob"] == pytest.approx(0.5)
    assert stats["frac_uncertain"] == pytest.approx(0.5)   # only the two 0.5 pixels
