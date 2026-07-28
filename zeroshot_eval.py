# FILE MAP | Zero-shot (untrained, GT-box-prompted) oracle baseline entry point. Mirrors
#   train.py's shape: pure-config resolution first, torch imported only for a real run.
#   --dry-run prints the plan and exits 0 WITHOUT importing torch (works on any machine).
#   Runs docs/MEDSAM_INVESTIGATION.md experiments A (vanilla_medsam_minmax) and B
#   (vanilla_sam_b). The two PUBLISHED rows (vanilla_sam, vanilla_medsam) are guarded — they
#   were produced by notebooks/05_benchmark.ipynb and re-running them here from a different
#   code path would silently overwrite the published numbers.
#   [DO NOT TOUCH] the published-row guard in main().
"""
Usage:
  python zeroshot_eval.py --config configs/run.yaml --baseline vanilla_sam_b
  python zeroshot_eval.py --config configs/run.yaml --baseline vanilla_medsam_minmax
  python zeroshot_eval.py --config configs/run.yaml --baseline vanilla_sam_b --dry-run
"""

import argparse
import sys
from pathlib import Path

from src.config import ZEROSHOT_CHOICES, build_zeroshot_plan, describe_zeroshot_plan, load_run_config


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Zero-shot oracle-box baseline evaluator.")
    p.add_argument("--config", default="configs/run.yaml", help="run config (or legacy base config)")
    p.add_argument("--baseline", required=True, choices=list(ZEROSHOT_CHOICES),
                   help="which zero-shot baseline to evaluate")
    p.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit 0")
    p.add_argument("--results-dir", default=None, help="override the local results root")
    p.add_argument("--allow-published", action="store_true",
                   help="allow re-computing a PUBLISHED oracle row (vanilla_sam / vanilla_medsam)")
    return p.parse_args(argv)


def _print_row(key: str, scores: dict) -> None:
    from src.training import reporting
    label = reporting.SPLIT_LABELS.get(key, key)
    print(f"{label:<26} {scores['dice']:>7.4f} {scores['iou']:>7.4f} {scores['mae']:>7.4f} "
          f"{scores['wfm']:>7.4f} {scores['sm']:>7.4f} {scores['em']:>7.4f}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = load_run_config(args.config)
    plan = build_zeroshot_plan(cfg, args.baseline, {"results_dir": args.results_dir})

    if args.dry_run:
        print(describe_zeroshot_plan(plan))
        return 0

    if plan.published and not args.allow_published:
        print(f"'{plan.key}' is a PUBLISHED oracle row (results/{plan.key}/seed0/metrics.json), "
              f"produced by notebooks/05_benchmark.ipynb. Re-computing it here would overwrite it "
              f"from a different code path — and on a fresh Colab runtime with no local results/ "
              f"and Drive unmounted, that overwrite is silent. Re-run 05_benchmark.ipynb to "
              f"regenerate it, or pass --allow-published to override deliberately.")
        return 2

    if not Path(plan.checkpoint).exists():
        print(f"Checkpoint not found: {plan.checkpoint}. Download it first (see "
              f"notebooks/train_colab.ipynb cell 3 or docs/MEDSAM_INVESTIGATION.md).")
        return 2

    # Heavy deps imported only for a real run.
    import torch

    from src.data import build_splits
    from src.metrics import MetricTracker
    from src.models import build_zeroshot_sam, evaluate_zeroshot_all_splits
    from src.results_summary import build_zeroshot_payload, write_zeroshot_metrics
    from src.training import reporting

    if plan.published:   # implies --allow-published (checked above)
        local = Path(plan.local_results_dir) / "metrics.json"
        drive = Path(plan.drive_results_dir) / "metrics.json"
        if not local.exists() and not drive.exists():
            print(f"Refusing to compute '{plan.key}': no existing copy of the published row was "
                  f"found at {local} or {drive}. Mount Drive (or restore results/) first so the "
                  f"current published numbers can be recovered if this run differs.")
            return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    with reporting.Tee(Path(plan.local_results_dir) / "run.log"):
        print(f"Baseline: {plan.key}  |  Backbone: {plan.model_type}  |  "
              f"Normalization: {plan.normalization}  |  Device: {device} ({device_name})")
        splits = build_splits(plan.data_root, seed=plan.splits_seed)
        zs = build_zeroshot_sam(
            checkpoint=plan.checkpoint, model_type=plan.model_type, device=device,
            prompt=plan.prompt, box_padding=plan.box_padding, normalization=plan.normalization,
        )
        print(f"\n{'Split':<26} {'mDice':>7} {'mIoU':>7} {'MAE':>7} {'wFm':>7} {'Sm':>7} {'Em':>7}")
        print("-" * 75)
        # reporting.SPLIT_LABELS IS MANDATORY: without it, evaluate_zeroshot_all_splits falls
        # back to list(splits.keys()), which includes the 1450-image "train" split.
        results = evaluate_zeroshot_all_splits(
            zs, splits, plan.img_size, MetricTracker, reporting.SPLIT_LABELS, on_split=_print_row
        )
        gap = reporting.generalization_gap(results)
        if gap is not None:
            print(f"\nGeneralization gap (seen - unseen mDice): {gap:+.4f}")

        payload = build_zeroshot_payload(
            plan.key, plan.model_type, results, total_params=zs.total_parameters(),
            device_name=device_name, prompt_protocol=plan.prompt, normalization=plan.normalization,
        )
        path = write_zeroshot_metrics(plan.local_results_root, plan.key, payload)
        print(f"Wrote {path}")

    reporting.mirror_to_drive(plan.local_results_dir, plan.drive_results_dir)
    print(f"Done. Metrics: {Path(plan.local_results_dir) / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
