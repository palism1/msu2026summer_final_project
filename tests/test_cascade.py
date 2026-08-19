# FILE MAP | Unit tests for the pure box cascade (src/ensemble/cascade.py). No torch, no
#   segment_anything, no GPU. Covers largest-component connectivity/tie-breaking, box derivation,
#   padding clamping, the two invariant checks in build_cascade_plan, and the empty/multiblob
#   bookkeeping through a stub segmenter.
"""GPU-free tests for the box cascade."""

import numpy as np
import pytest

from src.config import build_cascade_plan
from src.ensemble.cascade import box_from_prediction, largest_component, run_cascade_split
from src.models.zeroshot import box_from_mask


def test_largest_component_all_zero_returns_none():
    assert largest_component(np.zeros((16, 16), dtype=bool)) is None


def test_largest_component_picks_the_bigger_blob():
    m = np.zeros((16, 16), dtype=bool)
    m[0:3, 0:3] = True    # 9 pixels
    m[10:12, 10:12] = True  # 4 pixels
    comp = largest_component(m)
    assert comp.sum() == 9
    assert comp[0:3, 0:3].all()
    assert not comp[10:12, 10:12].any()


def test_largest_component_tie_break_is_lowest_label_and_repeatable():
    m = np.zeros((16, 16), dtype=bool)
    m[0:2, 0:2] = True    # 4 pixels, label 1 (scanned first)
    m[10:12, 10:12] = True  # 4 pixels, label 2
    comp1 = largest_component(m)
    comp2 = largest_component(m)
    assert np.array_equal(comp1, comp2)
    assert comp1[0:2, 0:2].all()
    assert not comp1[10:12, 10:12].any()


def test_largest_component_8connectivity_joins_diagonal_pixels():
    m = np.zeros((8, 8), dtype=bool)
    m[2, 2] = True
    m[3, 3] = True   # diagonal neighbor — joins under 8-connectivity, not 4
    comp = largest_component(m)
    assert comp.sum() == 2


def test_box_from_prediction_matches_box_from_mask_on_single_blob():
    prob = np.zeros((16, 16), dtype=np.float32)
    prob[4:8, 5:9] = 0.9
    box = box_from_prediction(prob, threshold=0.5, padding=3)
    expected = box_from_mask(prob >= 0.5, padding=3)
    assert box.tolist() == expected.tolist()


def test_box_from_prediction_below_threshold_returns_none():
    prob = np.full((16, 16), 0.3, dtype=np.float32)
    assert box_from_prediction(prob, threshold=0.5) is None


def test_box_from_prediction_ignores_small_second_blob():
    prob = np.zeros((16, 16), dtype=np.float32)
    prob[2:10, 2:10] = 0.9    # 8x8 main blob
    prob[14, 14] = 0.9        # 1-pixel stray blob, far away
    box = box_from_prediction(prob, threshold=0.5, padding=0)
    assert box.tolist() == [2.0, 2.0, 9.0, 9.0]


def test_box_from_prediction_padding_clamps_to_bounds():
    prob = np.zeros((16, 16), dtype=np.float32)
    prob[7:9, 7:9] = 0.9
    box = box_from_prediction(prob, threshold=0.5, padding=100)
    assert box.tolist() == [0.0, 0.0, 15.0, 15.0]


# ---------------------------------------------------------------------------
# build_cascade_plan invariants
# ---------------------------------------------------------------------------

def test_build_cascade_plan_rejects_gt_box_with_derived_tier():
    base_cfg = {
        "data": {"root": "data/polyp", "img_size": 352},
        "zeroshot": {"box_padding": 5, "checkpoints": {"vanilla_medsam_minmax": "medsam_vit_b.pth"}},
    }
    with pytest.raises(ValueError):
        from src.config import CascadeSpec, _check_cascade_invariants
        _check_cascade_invariants(
            "bad_spec", CascadeSpec(None, "vanilla_medsam_minmax", "gt", "zero", "derived", "bad")
        )


def test_build_cascade_plan_known_specs_resolve():
    cfg = {
        "data": {"root": "data/polyp", "img_size": 352},
        "zeroshot": {"box_padding": 5, "checkpoints": {"vanilla_medsam_minmax": "medsam_vit_b.pth"}},
        "sam": {"model_type": "vit_h", "checkpoint": "sam_vit_h_4b8939.pth"},
        "model": {"encoder": "resnet34"},
    }
    plan = build_cascade_plan(cfg, "casc_unet_medsam", seed=42)
    assert plan.tier == "derived"
    assert plan.box_source == "prediction"

    ceiling = build_cascade_plan(cfg, "oracle_casc_gtbox_medsam")
    assert ceiling.tier == "oracle"
    assert ceiling.box_source == "gt"
    assert ceiling.seed == 0


# ---------------------------------------------------------------------------
# run_cascade_split with a stub segmenter (no torch, no segment_anything)
# ---------------------------------------------------------------------------

class _StubSegmenter:
    """Returns a fixed, distinguishable probability map for every box it is asked to segment."""

    def predict_prob_from_box(self, image_uint8, box, out_hw):
        return np.full(out_hw, 0.77, dtype=np.float32)


def _make_split(n, h, w, empty_indices=()):
    rng = np.random.default_rng(0)
    gt = np.zeros((n, h, w), dtype=np.float32)
    detector_probs = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
        if i in empty_indices:
            continue
        gt[i, 2:5, 2:5] = 1.0
        detector_probs[i, 2:5, 2:5] = 0.9
    return gt, detector_probs


def test_run_cascade_split_empty_policy_zero(tmp_path, monkeypatch):
    n, h, w = 3, 8, 8
    gt, detector_probs = _make_split(n, h, w, empty_indices=(1,))
    image_paths = [str(tmp_path / f"img_{i}.png") for i in range(n)]

    monkeypatch.setattr("src.ensemble.cascade.load_image_uint8",
                        lambda path, img_size: np.zeros((img_size, img_size, 3), np.uint8))

    result = run_cascade_split(
        _StubSegmenter(), image_paths, detector_probs, gt, img_size=8, box_padding=0,
        detector_threshold=0.5, empty_policy="zero", box_source="prediction",
    )
    assert result.n_empty == 1
    assert np.all(result.probs[1] == 0.0)          # empty index -> all-zero, no detection invented
    assert np.all(result.probs[0] == 0.77)          # non-empty -> stub segmenter's fixed output


def test_run_cascade_split_empty_policy_passthrough(tmp_path, monkeypatch):
    n, h, w = 3, 8, 8
    gt, detector_probs = _make_split(n, h, w, empty_indices=(2,))
    detector_probs[2] = 0.42   # distinguishable passthrough value
    image_paths = [str(tmp_path / f"img_{i}.png") for i in range(n)]

    monkeypatch.setattr("src.ensemble.cascade.load_image_uint8",
                        lambda path, img_size: np.zeros((img_size, img_size, 3), np.uint8))

    result = run_cascade_split(
        _StubSegmenter(), image_paths, detector_probs, gt, img_size=8, box_padding=0,
        detector_threshold=0.5, empty_policy="passthrough", box_source="prediction",
    )
    assert result.n_empty == 1
    assert np.allclose(result.probs[2], 0.42)       # passthrough copies the detector map
