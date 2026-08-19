# FILE MAP | Unit tests for weight fitting (src/ensemble/combine.py Phase 3): the contiguous
#   fit/holdout split, the simplex grid, fit_weights' tie-break, and build_fit_stack's leakage
#   guard. No torch, no GPU.
"""GPU-free tests for ensemble weight fitting."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.ensemble.combine import (
    SEEN_SPLITS,
    build_fit_stack,
    fit_holdout_indices,
    fit_weights,
    simplex_grid,
)


def test_fit_holdout_indices_n100():
    fit, holdout = fit_holdout_indices(100)
    assert fit == list(range(50))
    assert holdout == list(range(50, 100))


def test_fit_holdout_indices_n62():
    fit, holdout = fit_holdout_indices(62)
    assert fit == list(range(31))
    assert holdout == list(range(31, 62))


@pytest.mark.parametrize("n", range(10))
def test_fit_holdout_disjoint_and_complete(n):
    fit, holdout = fit_holdout_indices(n)
    assert sorted(fit + holdout) == list(range(n))
    assert set(fit).isdisjoint(holdout)
    if fit and holdout:
        assert max(fit) < min(holdout)


def test_simplex_grid_two_members_half_step():
    grid = simplex_grid(2, 0.5)
    assert grid == [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    for cand in grid:
        assert abs(sum(cand) - 1.0) < 1e-9


def test_simplex_grid_three_members_count():
    grid = simplex_grid(3, 0.5)
    for cand in grid:
        assert abs(sum(cand) - 1.0) < 1e-9
    assert len(grid) == 6   # C(2+2, 2) = 6 compositions of 2 into 3 non-negative parts


def test_fit_weights_favors_the_accurate_member():
    rng = np.random.default_rng(4)
    gt = (rng.random((20, 8, 8)) > 0.5).astype(np.float32)
    member_a = gt.copy()          # perfect
    member_b = 1.0 - gt           # perfectly wrong (inverted) — a true drag on the average

    result = fit_weights([member_a, member_b], gt, step=0.05, threshold=0.5)
    assert result.weights[0] > result.weights[1]
    assert result.fit_dice > result.uniform_dice


def test_fit_weights_tie_break_prefers_uniform():
    # Both members are identical, so every weight split ties on Dice; the tie-break must
    # return the point closest to uniform, i.e. (0.5, 0.5).
    rng = np.random.default_rng(5)
    gt = (rng.random((10, 8, 8)) > 0.5).astype(np.float32)
    member = gt.copy()

    result = fit_weights([member, member], gt, step=0.25, threshold=0.5)
    assert result.weights == (0.5, 0.5)


def test_build_fit_stack_leakage_guard_only_reads_seen_splits():
    plan = SimpleNamespace(members=("unet", "sam_lora"))
    rng = np.random.default_rng(6)
    gt = (rng.random((10, 4, 4)) > 0.5).astype(np.float32)

    def read_cache(member, split):
        if split not in SEEN_SPLITS:
            raise AssertionError(f"read_cache called with non-seen split {split!r}")
        probs = rng.random((10, 4, 4)).astype(np.float32)
        return SimpleNamespace(probs=probs, gt=gt)

    member_stacks, gt_stack, meta = build_fit_stack(plan, read_cache)
    assert meta["splits"] == list(SEEN_SPLITS)
    assert len(member_stacks) == 2
    assert gt_stack.shape[0] == meta["n_fit_images"]


def test_build_fit_stack_rejects_unseen_split():
    plan = SimpleNamespace(members=("unet",))

    def read_cache(member, split):
        raise AssertionError("should not be called: validation happens before any read")

    with pytest.raises(ValueError):
        build_fit_stack(plan, read_cache, splits=("cvc_colondb",))
