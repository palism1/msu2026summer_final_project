# FILE MAP | Ensemble evaluator (docs/PLAN_ENSEMBLE.md Phase 2/3). Reads per-image probability
#   caches written by predict_cache.py, averages them (uniform or fitted weights), evaluates the
#   six metrics, and writes results/<spec>/seed<N>/metrics.json. Never imports torch or builds a
#   model — the whole computation is numpy over cached arrays. --self-check and --dry-run stay
#   fully GPU-free; the real-run write path lazily reuses src.training.reporting for run logging
#   and the Drive mirror, which is the one place torch enters (reporting.py imports it at module
#   level for its own, unrelated evaluate_all_splits — see the FILE MAP there).
"""
Usage:
  python ensemble_eval.py --spec ens_unet_samh --seed 42 [--dry-run]
  python ensemble_eval.py --self-check --member unet --seed 42
  python ensemble_eval.py --all
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.config import (
    ENSEMBLE_CHOICES,
    MODEL_CHOICES,
    build_ensemble_plan,
    build_run_plan,
    describe_ensemble_plan,
    load_run_config,
    resolve_cache_root,
    resolve_normalization,
)
from src.ensemble.cache import assert_aligned, cache_path, read_split_cache
from src.ensemble.combine import (
    average_probs,
    build_fit_stack,
    calibration_stats,
    evaluate_stack,
    fit_holdout_indices,
    fit_weights,
    merge_backbones,
    merge_normalization,
    normalize_weights,
)
from src.metrics import MetricTracker, dice_score
from src.results_summary import ALL_SPLITS, SEEN_SPLITS, build_derived_payload, write_run_metrics

TOLERANCE = 1e-3
_DEFAULT_SEEDS = (42, 43, 44)
_THRESHOLD_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ensemble evaluator (reads probability caches only).")
    p.add_argument("--config", default="configs/run.yaml")
    p.add_argument("--spec", choices=list(ENSEMBLE_CHOICES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-check", action="store_true",
                   help="one-member sanity check: cache reproduces the published row")
    p.add_argument("--member", choices=list(MODEL_CHOICES), help="model key for --self-check")
    p.add_argument("--all", action="store_true", help="run every ensemble spec over seeds 42-44")
    p.add_argument("--cache-dir", default=None)
    return p.parse_args(argv)


def _find_published_metrics(cfg: dict, model_dir: str, seed: int):
    from src.results_summary import DEFAULT_DRIVE_RESULTS

    output = cfg.get("output", {}) or {}
    local = Path(output.get("local_results_dir", "results")) / model_dir / f"seed{seed}" / "metrics.json"
    drive = (Path(output.get("drive_results_dir", DEFAULT_DRIVE_RESULTS))
            / model_dir / f"seed{seed}" / "metrics.json")
    for candidate in (local, drive):
        if candidate.is_file():
            return candidate
    return None


def _load_member_payload(cfg: dict, model_dir: str, seed: int) -> dict:
    mpath = _find_published_metrics(cfg, model_dir, seed)
    if mpath is None:
        raise FileNotFoundError(
            f"No published metrics.json for '{model_dir}' seed{seed}. Restore the Drive results "
            f"mirror into results/ first — see notebooks/08_ensemble.ipynb step 0."
        )
    return json.loads(mpath.read_text())


def _print_delta_table(table: list[dict]) -> None:
    print(f"\n{'Split':<20} {'Published':>10} {'Cached':>10} {'Delta':>10}")
    for row in table:
        pub = f"{row['published']:.6f}" if row["published"] is not None else "—"
        delta = f"{row['delta']:.6f}" if row["delta"] is not None else "—"
        print(f"{row['split']:<20} {pub:>10} {row['cached']:>10.6f} {delta:>10}")


# ---------------------------------------------------------------------------
# --self-check
# ---------------------------------------------------------------------------

def run_self_check(cfg: dict, member: str, seed: int, cache_root: str) -> int:
    plan = build_run_plan(cfg, {"model": member, "seed": seed})
    model_dir = plan.checkpoint_name
    published = _load_member_payload(cfg, model_dir, seed)

    table = []
    ok = True
    for split in ALL_SPLITS:
        cpath = cache_path(cache_root, model_dir, seed, split)
        if not cpath.is_file():
            print(f"Missing cache: {cpath}. Run predict_cache.py for '{member}' seed{seed} first.")
            return 2
        cache = read_split_cache(cpath)
        probs = average_probs([cache.probs], [1.0])
        cached = evaluate_stack(probs, cache.gt, MetricTracker)
        pub_dice = published.get("eval", {}).get(split, {}).get("dice")
        delta = None if pub_dice is None else abs(cached["dice"] - pub_dice)
        table.append({"split": split, "published": pub_dice, "cached": cached["dice"], "delta": delta})
        if delta is None or delta > TOLERANCE:
            ok = False

    _print_delta_table(table)
    if ok:
        print(f"\nSelf-check PASSED for '{member}' seed{seed}: cache reproduces the published row.")
        return 0
    print(f"\nSelf-check FAILED for '{member}' seed{seed}: cache does not reproduce the "
         f"published row within {TOLERANCE}. Re-run predict_cache.py for this member/seed.")
    return 1


# ---------------------------------------------------------------------------
# Real ensemble run
# ---------------------------------------------------------------------------

def _load_all_caches(plan, cache_root: str) -> dict[str, list]:
    """{split: [SplitCache per member, in plan.member_dirs order]}, aligned via assert_aligned."""
    caches_by_split: dict[str, list] = {}
    for split in ALL_SPLITS:
        member_caches = []
        for model_dir, seed in zip(plan.member_dirs, plan.member_seeds):
            cpath = cache_path(cache_root, model_dir, seed, split)
            if not cpath.is_file():
                raise FileNotFoundError(
                    f"Missing cache: {cpath}. Run predict_cache.py for '{model_dir}' seed{seed} "
                    f"first (all {len(plan.member_dirs)} members need every split cached)."
                )
            member_caches.append(read_split_cache(cpath))
        assert_aligned(member_caches)
        caches_by_split[split] = member_caches
    return caches_by_split


def _fit_ensemble_weights(plan, caches_by_split: dict) -> tuple[list[float], dict]:
    """Fit weights on the first contiguous half of the seen splits; refit on the second half as
    a stability check; return (weights, fit_extras_block)."""
    fit_proxy = SimpleNamespace(members=list(plan.member_dirs))

    def _read_first_half(member_dir, split):
        idx = plan.member_dirs.index(member_dir)
        return caches_by_split[split][idx]

    member_stacks_fit, gt_fit, fit_meta = build_fit_stack(fit_proxy, _read_first_half)
    fit_result = fit_weights(member_stacks_fit, gt_fit, step=plan.grid_step, threshold=plan.threshold)

    fit_image_paths: list[str] = []
    for split in SEEN_SPLITS:
        fit_idx, _holdout_idx = fit_holdout_indices(caches_by_split[split][0].probs.shape[0])
        member0 = caches_by_split[split][0]
        fit_image_paths += [member0.image_paths[i] for i in fit_idx]

    # Stability check: refit on the second (holdout) half of the same two seen splits.
    second_half_probs = {m: [] for m in plan.member_dirs}
    second_half_gt_chunks = []
    for split in SEEN_SPLITS:
        n = caches_by_split[split][0].probs.shape[0]
        _fit_idx, holdout_idx = fit_holdout_indices(n)
        for i, model_dir in enumerate(plan.member_dirs):
            second_half_probs[model_dir].append(caches_by_split[split][i].probs[holdout_idx])
        second_half_gt_chunks.append(caches_by_split[split][0].gt[holdout_idx])
    second_stacks = [np.concatenate(second_half_probs[m], axis=0) for m in plan.member_dirs]
    second_gt = np.concatenate(second_half_gt_chunks, axis=0)
    stability = fit_weights(second_stacks, second_gt, step=plan.grid_step, threshold=plan.threshold)

    uniform_weights = normalize_weights([1.0] * len(plan.member_dirs)).tolist()
    fit_paths_hash = hashlib.sha256("\n".join(fit_image_paths).encode()).hexdigest()

    threshold_sensitivity = {}
    fit_probs = average_probs(member_stacks_fit, fit_result.weights)
    for thr in _THRESHOLD_SWEEP:
        pred = (fit_probs >= thr).astype(np.float32)
        dices = [dice_score(pred[i], gt_fit[i]) for i in range(pred.shape[0])]
        threshold_sensitivity[str(thr)] = round(float(np.mean(dices)), 6)

    extras = {
        "splits": fit_meta["splits"],
        "index_rule": fit_meta["index_rule"],
        "n_fit_images": fit_meta["n_fit_images"],
        "fit_dice": round(fit_result.fit_dice, 6),
        "uniform_fit_dice": round(fit_result.uniform_dice, 6),
        "grid_step": plan.grid_step,
        "n_candidates": fit_result.n_candidates,
        "stability_second_half_weights": list(stability.weights),
        "image_paths_sha256": fit_paths_hash,
    }
    return list(fit_result.weights), extras, uniform_weights, threshold_sensitivity


def _eval_seen_holdout(plan, caches_by_split: dict, weights: list[float]) -> dict:
    out = {}
    for split in SEEN_SPLITS:
        n = caches_by_split[split][0].probs.shape[0]
        _fit_idx, holdout_idx = fit_holdout_indices(n)
        probs_h = average_probs(
            [caches_by_split[split][i].probs[holdout_idx] for i in range(len(plan.member_dirs))],
            weights,
        )
        gt_h = caches_by_split[split][0].gt[holdout_idx]
        out[split] = evaluate_stack(probs_h, gt_h, MetricTracker)
    return out


def run_ensemble(cfg: dict, spec_key: str, seed: int, cache_root: str, dry_run: bool) -> int:
    plan = build_ensemble_plan(cfg, spec_key, seed, {"cache_dir": cache_root})
    if dry_run:
        print(describe_ensemble_plan(plan))
        return 0

    caches_by_split = _load_all_caches(plan, cache_root)

    threshold_sensitivity = None
    fit_extras = None
    eval_seen_holdout = None
    if plan.weighting == "fitted":
        weights, fit_extras, uniform_weights, threshold_sensitivity = _fit_ensemble_weights(
            plan, caches_by_split)
        eval_seen_holdout = _eval_seen_holdout(plan, caches_by_split, weights)
    else:
        weights = normalize_weights([1.0] * len(plan.member_dirs)).tolist()
        uniform_weights = weights

    eval_results = {}
    for split in ALL_SPLITS:
        probs = average_probs([c.probs for c in caches_by_split[split]], weights)
        gt = caches_by_split[split][0].gt
        eval_results[split] = evaluate_stack(probs, gt, MetricTracker)
        print(f"  {split:<16} dice={eval_results[split]['dice']:.4f}")

    calibration = {}
    for i, model_dir in enumerate(plan.member_dirs):
        all_probs = np.concatenate([caches_by_split[s][i].probs for s in ALL_SPLITS], axis=0)
        calibration[model_dir] = calibration_stats(all_probs)

    member_payloads = [_load_member_payload(cfg, d, s)
                      for d, s in zip(plan.member_dirs, plan.member_seeds)]
    backbones = [build_run_plan(cfg, {"model": m, "seed": seed}).backbone for m in plan.members]
    normalizations = [resolve_normalization(m) for m in plan.members]

    ensemble_block = {
        "members": list(plan.members),
        "member_dirs": list(plan.member_dirs),
        "member_seeds": list(plan.member_seeds),
        "weighting": plan.weighting,
        "weights": weights,
        "uniform_weights": uniform_weights,
        "threshold": plan.threshold,
        "calibration": calibration,
    }
    if fit_extras is not None:
        ensemble_block["fit"] = fit_extras
    if eval_seen_holdout is not None:
        ensemble_block["eval_seen_holdout"] = eval_seen_holdout
    if threshold_sensitivity is not None:
        ensemble_block["threshold_sensitivity_fit_slice"] = threshold_sensitivity

    payload = build_derived_payload(
        plan.key, plan.seed, eval_results, member_payloads, backbones, normalizations,
        device_name=None, extras={"ensemble": ensemble_block},
    )

    # results/<spec>/seed<seed>/, plus run.log + Drive mirror. Lazy import: reporting.py imports
    # torch at module level for its own evaluate_all_splits (unrelated to this numpy-only path).
    from src.training import reporting

    with reporting.Tee(Path(plan.local_results_dir) / "run.log"):
        results_root = Path(plan.local_results_dir).parent.parent
        path = write_run_metrics(results_root, plan.key, plan.seed, payload)
        print(f"Wrote {path}")
    reporting.mirror_to_drive(plan.local_results_dir, plan.drive_results_dir)
    print(f"Done. Metrics: {Path(plan.local_results_dir) / 'metrics.json'}")
    return 0


def run_all(cfg: dict, cache_root: str) -> int:
    from src.config import ENSEMBLE_SPECS
    for spec_key in ENSEMBLE_SPECS:
        for seed in _DEFAULT_SEEDS:
            print(f"\n=== {spec_key} seed{seed} ===")
            rc = run_ensemble(cfg, spec_key, seed, cache_root, dry_run=False)
            if rc != 0:
                return rc
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = load_run_config(args.config)
    cache_root = args.cache_dir or resolve_cache_root(cfg)

    if args.self_check:
        if not args.member:
            print("ensemble_eval.py --self-check requires --member.")
            return 2
        return run_self_check(cfg, args.member, args.seed, cache_root)

    if args.all:
        return run_all(cfg, cache_root)

    if not args.spec:
        print("ensemble_eval.py requires --spec, --self-check, or --all.")
        return 2

    return run_ensemble(cfg, args.spec, args.seed, cache_root, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
