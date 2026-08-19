# FILE MAP | Box cascade evaluator (docs/PLAN_ENSEMBLE.md Phase 4). Detector row: reads the
#   U-Net probability cache (never a live U-Net forward pass), derives a box per image via
#   src.ensemble.cascade, and segments inside it with MedSAM. Ceiling row: derives the box from
#   the ground truth instead, through the exact same image loader, so the two rows are
#   path-matched. --dry-run stays torch-free; the real run lazily imports torch to build the
#   MedSAM segmenter (src.models.zeroshot.build_zeroshot_sam) and to log/mirror via
#   src.training.reporting.
#   [DO NOT TOUCH] the ceiling row's box_source="gt" / tier="oracle" pairing — build_cascade_plan
#   enforces it as a ValueError, this file must not try to route around that.
"""
Usage:
  python cascade_eval.py --spec casc_unet_medsam --seed 42 [--dry-run]
  python cascade_eval.py --spec oracle_casc_gtbox_medsam
"""

import argparse
import sys
from pathlib import Path

from src.config import (
    CASCADE_CHOICES,
    build_cascade_plan,
    build_run_plan,
    describe_cascade_plan,
    load_run_config,
    resolve_cache_root,
    resolve_normalization,
)
from src.ensemble.cache import cache_path, read_split_cache
from src.results_summary import ALL_SPLITS

_CASCADE_NOTES = (
    "MedSAM's pretraining corpus included colonoscopy polyp data, so 'unseen' here means "
    "unseen by the U-Net detector only."
)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Box cascade evaluator.")
    p.add_argument("--config", default="configs/run.yaml")
    p.add_argument("--spec", required=True, choices=list(CASCADE_CHOICES))
    p.add_argument("--seed", type=int, default=42,
                   help="ignored for the ceiling row, which always uses seed 0")
    p.add_argument("--dry-run", action="store_true")
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
    import json
    mpath = _find_published_metrics(cfg, model_dir, seed)
    if mpath is None:
        raise FileNotFoundError(
            f"No published metrics.json for '{model_dir}' seed{seed}. Restore the Drive results "
            f"mirror into results/ first — see notebooks/08_ensemble.ipynb step 0."
        )
    return json.loads(mpath.read_text())


def _load_gt_stack(mask_paths, img_size: int):
    """Resize masks to (img_size, img_size) with NEAREST and binarize — matches
    src.models.zeroshot.predict_zeroshot_prob's ground-truth handling exactly, so the ceiling
    row scores against the same GT convention the published vanilla_medsam_minmax row used."""
    import numpy as np
    from PIL import Image

    out = np.zeros((len(mask_paths), img_size, img_size), dtype=np.float32)
    for i, mp in enumerate(mask_paths):
        gt = Image.open(mp).convert("L").resize((img_size, img_size), Image.NEAREST)
        out[i] = (np.array(gt) > 127).astype(np.float32)
    return out


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = load_run_config(args.config)
    plan = build_cascade_plan(cfg, args.spec, args.seed)

    if args.dry_run:
        print(describe_cascade_plan(plan))
        return 0

    cache_root = args.cache_dir or resolve_cache_root(cfg)

    if plan.box_source == "prediction":
        probe = cache_path(cache_root, plan.detector_dir, plan.seed, ALL_SPLITS[0])
        if not probe.is_file():
            print(f"Missing detector cache: {probe}. Run predict_cache.py for '{plan.detector}' "
                 f"seed{plan.seed} first.")
            return 2

    # Heavy deps imported only for a real run.
    import torch

    from src.data import build_splits
    from src.ensemble.cascade import run_cascade_split
    from src.ensemble.combine import evaluate_stack
    from src.metrics import MetricTracker
    from src.models import build_zeroshot_sam
    from src.results_summary import build_derived_payload, write_run_metrics
    from src.training import reporting

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    with reporting.Tee(Path(plan.local_results_dir) / "run.log"):
        print(f"Cascade: {plan.key}  |  Tier: {plan.tier}  |  Box source: {plan.box_source}  |  "
             f"Device: {device} ({device_name})")
        zs = build_zeroshot_sam(
            checkpoint=plan.segmenter_checkpoint, model_type=plan.segmenter_model_type,
            device=device, prompt="box", box_padding=plan.box_padding,
            normalization=plan.segmenter_normalization,
        )

        raw_splits = build_splits(plan.data_root) if plan.box_source == "gt" else None

        eval_results = {}
        per_split_stats = {}
        total_gpu_seconds = 0.0
        total_images = 0

        for split in ALL_SPLITS:
            if plan.box_source == "prediction":
                cache = read_split_cache(cache_path(cache_root, plan.detector_dir, plan.seed, split))
                image_paths, gt, detector_probs = cache.image_paths, cache.gt, cache.probs
            else:
                if split not in raw_splits:
                    continue
                image_paths = [str(p) for p in raw_splits[split]["image_paths"]]
                gt = _load_gt_stack(raw_splits[split]["mask_paths"], plan.img_size)
                detector_probs = None

            result = run_cascade_split(
                zs, image_paths, detector_probs, gt, plan.img_size, plan.box_padding,
                plan.detector_threshold, plan.empty_policy, plan.box_source,
            )
            eval_results[split] = evaluate_stack(result.probs, gt, MetricTracker)
            per_split_stats[split] = {"n_images": int(gt.shape[0]), "n_empty": result.n_empty,
                                      "n_multiblob": result.n_multiblob}
            total_gpu_seconds += result.gpu_seconds
            total_images += gt.shape[0]
            print(f"  {split:<16} dice={eval_results[split]['dice']:.4f}  "
                 f"n_empty={result.n_empty}  n_multiblob={result.n_multiblob}")

        segmenter_payload = {
            "params": {"total": zs.total_parameters(), "trainable": 0},
            "checkpoint_size_mb": zs.checkpoint_size_mb(),
            "timing": {"total_seconds": 0},
        }
        if plan.detector is not None:
            detector_payload = _load_member_payload(cfg, plan.detector_dir, plan.seed)
            member_payloads = [detector_payload, segmenter_payload]
            backbones = [build_run_plan(cfg, {"model": plan.detector, "seed": plan.seed}).backbone,
                        plan.segmenter_model_type]
            normalizations = [resolve_normalization(plan.detector), plan.segmenter_normalization]
        else:
            member_payloads = [segmenter_payload]
            backbones = [plan.segmenter_model_type]
            normalizations = [plan.segmenter_normalization]

        cascade_block = {
            "detector": plan.detector,
            "detector_seed": plan.seed if plan.detector else None,
            "detector_source": "float16 prediction cache" if plan.detector else None,
            "segmenter": plan.segmenter,
            "box_source": plan.box_source,
            "empty_policy": plan.empty_policy,
            "box_padding": plan.box_padding,
            "detector_threshold": plan.detector_threshold,
            "per_split": per_split_stats,
            "notes": _CASCADE_NOTES,
        }

        payload = build_derived_payload(
            plan.key, plan.seed, eval_results, member_payloads, backbones, normalizations,
            device_name=device_name, extras={"cascade": cascade_block},
        )
        inference = {
            "device_name": device_name, "n_images": total_images,
            "gpu_seconds_total": round(total_gpu_seconds, 4),
            "gpu_seconds_per_image": round(total_gpu_seconds / total_images, 6) if total_images else None,
            "batch_size": 1, "source": "cascade_eval",
        }
        payload["inference"] = inference

        results_root = Path(plan.local_results_dir).parent.parent
        path = write_run_metrics(results_root, plan.key, plan.seed, payload)
        print(f"Wrote {path}")

    reporting.mirror_to_drive(plan.local_results_dir, plan.drive_results_dir)
    print(f"Done. Metrics: {Path(plan.local_results_dir) / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
