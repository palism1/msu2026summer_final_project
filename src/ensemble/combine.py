# FILE MAP | Pure numpy/stdlib ensemble math (docs/PLAN_ENSEMBLE.md Phase 2/3): weighted
#   averaging, the six-metric evaluation shim over a probability stack, calibration diagnostics,
#   the derived-payload metadata mergers, and weight fitting on a leakage-free fit slice.
#   MetricTracker is imported lazily (inside evaluate_stack), so importing this module needs only
#   numpy + scipy, no torch. [DO NOT TOUCH] fit_holdout_indices' contiguous (not interleaved)
#   split — see docs/PLAN_ENSEMBLE.md interpretation 3 (CVC-ClinicDB video-sequence leakage).
"""Torch-free ensemble combination, evaluation, and weight fitting."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

# Split groupings an ensemble may fit weights on. Mirrors src/results_summary.SEEN_SPLITS.
# [DO NOT TOUCH] the leakage guard in build_fit_stack depends on this exact tuple.
SEEN_SPLITS = ("seen_kvasir", "seen_clinicdb")


# ---------------------------------------------------------------------------
# Weighted averaging
# ---------------------------------------------------------------------------

def normalize_weights(weights: Iterable[float]) -> np.ndarray:
    """Return a non-negative weight vector summing to 1.

    Raises ValueError on a negative entry or an all-zero vector — both are meaningless as
    ensemble weights.
    """
    w = np.asarray(list(weights), dtype=np.float64)
    if np.any(w < 0):
        raise ValueError(f"weights must be non-negative, got {w.tolist()}")
    total = w.sum()
    if total <= 0:
        raise ValueError(f"weights must sum to a positive value, got {w.tolist()}")
    return w / total


def average_probs(stacks: list[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    """Weighted mean of member probability stacks, each shaped (N, H, W).

    Raises ValueError when the members' shapes disagree.
    """
    shapes = {s.shape for s in stacks}
    if len(shapes) > 1:
        raise ValueError(f"member shape mismatch: {sorted(shapes)}")
    w = normalize_weights(weights)
    if len(w) != len(stacks):
        raise ValueError(f"got {len(stacks)} stacks but {len(w)} weights")
    out = np.zeros_like(stacks[0], dtype=np.float64)
    for s, wi in zip(stacks, w):
        out += wi * s.astype(np.float64)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation shim over a probability stack
# ---------------------------------------------------------------------------

def evaluate_stack(probs: np.ndarray, gt: np.ndarray, tracker_factory: Callable) -> dict:
    """Run every per-image pair through tracker_factory() (typically MetricTracker) and return
    the dataset-level mean of all six metrics: {dice, iou, mae, wfm, sm, em}."""
    tracker = tracker_factory()
    for i in range(probs.shape[0]):
        tracker.update(probs[i], gt[i])
    return tracker.compute()


def calibration_stats(probs: np.ndarray) -> dict:
    """Cheap calibration diagnostic for a probability stack: mean predicted probability, and the
    fraction of pixels in the "uncertain" band [0.05, 0.95]."""
    p = np.asarray(probs, dtype=np.float64)
    uncertain = (p >= 0.05) & (p <= 0.95)
    return {
        "mean_prob": float(p.mean()),
        "frac_uncertain": float(uncertain.mean()),
    }


# ---------------------------------------------------------------------------
# Derived-payload metadata mergers — never hardcode "mixed" or a backbone string elsewhere.
# ---------------------------------------------------------------------------

def merge_backbones(values: list[str]) -> str:
    """Join member backbones with '+', e.g. ["resnet34", "vit_h", "vit_b"] -> "resnet34+vit_h+vit_b"."""
    return "+".join(values)


def merge_normalization(values: list[str]) -> str:
    """A single normalization string when every member agrees, else the literal "mixed".

    This is the only place that string is produced — callers must never hardcode "mixed" or a
    backbone string themselves (docs/PLAN_ENSEMBLE.md Phase 2)."""
    uniq = set(values)
    if len(uniq) == 1:
        return next(iter(uniq))
    return "mixed"


# ---------------------------------------------------------------------------
# Weight fitting (Phase 3)
# ---------------------------------------------------------------------------

def fit_holdout_indices(n: int) -> tuple[list[int], list[int]]:
    """Contiguous split: fit = [0, n//2), holdout = [n//2, n).

    Contiguous, not interleaved — CVC-ClinicDB frames come from 29 video sequences, so an
    interleaved split would place near-duplicate frames on both sides. See docs/PLAN_ENSEMBLE.md
    interpretation 3."""
    half = n // 2
    return list(range(half)), list(range(half, n))


def simplex_grid(k: int, step: float) -> list[tuple[float, ...]]:
    """Every non-negative rational k-tuple on the grid of multiples of ``step`` that sums to 1.

    Returned in the order itertools.product yields the first k-1 coordinates, with the last
    coordinate implied. E.g. simplex_grid(2, 0.5) == [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]."""
    n_steps = round(1.0 / step)
    if abs(n_steps * step - 1.0) > 1e-9:
        raise ValueError(f"step {step} must divide 1.0 evenly")

    out: list[tuple[float, ...]] = []
    if k == 1:
        return [(1.0,)]

    def _rec(remaining_steps: int, coords: list[int], slots_left: int):
        if slots_left == 1:
            out.append(tuple(round((c) * step, 10) for c in coords + [remaining_steps]))
            return
        for i in range(remaining_steps + 1):
            _rec(remaining_steps - i, coords + [i], slots_left - 1)

    _rec(n_steps, [], k)
    return out


@dataclass(frozen=True)
class WeightFit:
    weights: tuple[float, ...]
    fit_dice: float
    uniform_dice: float
    n_fit_images: int
    grid_step: float
    n_candidates: int


def _thresholded_mean_dice(probs: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    from src.metrics import dice_score
    pred = (probs >= threshold).astype(np.float32)
    return float(np.mean([dice_score(pred[i], gt[i]) for i in range(pred.shape[0])]))


def fit_weights(member_stacks: list[np.ndarray], gt: np.ndarray, step: float = 0.05,
                threshold: float = 0.5) -> WeightFit:
    """Maximize mean per-image Dice of the thresholded weighted average over a simplex grid.

    Tie-break, in order: highest Dice, then smallest L2 distance to the uniform vector, then
    lexicographic order of the weight tuple — biasing toward the null hypothesis pre-empts a
    cherry-picked-optimum objection.
    """
    k = len(member_stacks)
    n = member_stacks[0].shape[0]
    candidates = simplex_grid(k, step)
    uniform = np.full(k, 1.0 / k)

    best = None  # (dice, l2_to_uniform, weights_tuple)
    uniform_dice = None
    for cand in candidates:
        probs = average_probs(member_stacks, cand)
        dice = _thresholded_mean_dice(probs, gt, threshold)
        l2 = float(np.linalg.norm(np.asarray(cand) - uniform))
        key = (-dice, l2, cand)   # minimize: -dice (== maximize dice), then l2, then lexicographic
        if best is None or key < best[0]:
            best = (key, dice, cand)
        if all(abs(c - 1.0 / k) < 1e-9 for c in cand):
            uniform_dice = dice

    if uniform_dice is None:  # the exact uniform point wasn't on the grid; compute it directly
        uniform_probs = average_probs(member_stacks, uniform.tolist())
        uniform_dice = _thresholded_mean_dice(uniform_probs, gt, threshold)

    _, best_dice, best_weights = best
    return WeightFit(
        weights=best_weights,
        fit_dice=best_dice,
        uniform_dice=uniform_dice,
        n_fit_images=n,
        grid_step=step,
        n_candidates=len(candidates),
    )


def build_fit_stack(plan, read_cache: Callable[[str, str], "object"],
                    splits: tuple[str, ...] = SEEN_SPLITS
                    ) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Build the leakage-free weight-fitting stack.

    For every member in ``plan.members`` (falling back to ``plan.member_dirs``), concatenate the
    first contiguous half of each requested split's cached probabilities and ground truth.
    ``read_cache(member, split)`` returns a cache-like object with ``.probs``/``.gt`` shaped
    (N, H, W); members share one ground-truth stack by construction (see assert_aligned).

    Raises ValueError when any requested split is outside SEEN_SPLITS — the leakage guard, in
    code, not in a comment: fitting weights on unseen data would leak the primary endpoint into
    weight selection.
    """
    bad = [s for s in splits if s not in SEEN_SPLITS]
    if bad:
        raise ValueError(
            f"build_fit_stack only fits weights on {SEEN_SPLITS} (seen test data); got {bad}. "
            f"Fitting on an unseen split would leak the primary endpoint into weight selection."
        )

    members = getattr(plan, "members", None) or getattr(plan, "member_dirs")
    member_stacks: list[np.ndarray] = []
    gt_stack = None
    for member in members:
        probs_chunks: list[np.ndarray] = []
        gt_chunks: list[np.ndarray] = []
        for split in splits:
            cache = read_cache(member, split)
            fit_idx, _holdout_idx = fit_holdout_indices(cache.probs.shape[0])
            probs_chunks.append(cache.probs[fit_idx])
            gt_chunks.append(cache.gt[fit_idx])
        member_stacks.append(np.concatenate(probs_chunks, axis=0))
        gt_stack = np.concatenate(gt_chunks, axis=0)  # identical across members by construction

    meta = {"splits": list(splits), "index_rule": "first_contiguous_half",
           "n_fit_images": int(gt_stack.shape[0])}
    return member_stacks, gt_stack, meta
