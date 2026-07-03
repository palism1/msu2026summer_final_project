# FILE MAP | Canonical training entry point (thin CLI).
#   Flow: parse args -> load+merge config (PURE, src/config.py) -> RunPlan.
#   --dry-run prints the plan and exits 0 WITHOUT importing torch (works on any machine).
#   A real run lazily imports src/training/{engine,reporting} to train, evaluate all 5 splits,
#   and write metrics.json + mask overlays + run.log (mirrored to Drive).
#   [DO NOT TOUCH] the argument contract / checkpoint path — evaluate.py and the notebooks
#   depend on checkpoints/<model>/seed<seed>/best.pt. [TWEAK] which model/seed via configs/run.yaml.
"""
Usage:
  python train.py --config configs/run.yaml                 # train the model set in run.yaml
  python train.py --config configs/run.yaml --dry-run       # validate + print plan, no GPU
  python train.py --config configs/run.yaml --model unet --seed 7 --epochs 50   # overrides
  python train.py --config configs/base.yaml --model unet   # legacy flat-config invocation
"""

import argparse
import sys
from pathlib import Path

from src.config import build_run_plan, describe_plan, load_run_config


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Config-driven polyp segmentation trainer.")
    p.add_argument("--config", default="configs/run.yaml", help="run config (or legacy base config)")
    p.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit 0")
    p.add_argument("--model", default=None, choices=["unet", "sam_lora", "medsam"],
                   help="override the model chosen in the config")
    p.add_argument("--seed", type=int, default=None, help="override the seed")
    p.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    p.add_argument("--output-dir", default=None, help="override the checkpoint root directory")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = load_run_config(args.config)
    plan = build_run_plan(cfg, {
        "model": args.model, "seed": args.seed,
        "epochs": args.epochs, "output_dir": args.output_dir,
    })

    if args.dry_run:
        print(describe_plan(plan))
        return 0

    # Heavy deps imported only for a real run.
    import time

    from src.training import reporting
    from src.training.engine import EngineDeps, default_tracker_factory, run_training

    deps = None
    if reporting.drive_available(plan.drive_checkpoint_dir):
        _mirror_state = {"last": None}
        _MIRROR_INTERVAL_SECONDS = 10 * 60

        def _on_best_checkpoint(checkpoint_path: str) -> None:
            now = time.time()
            last = _mirror_state["last"]
            if last is not None and (now - last) < _MIRROR_INTERVAL_SECONDS:
                return
            reporting.mirror_checkpoint_to_drive(checkpoint_path, plan.drive_checkpoint_dir)
            _mirror_state["last"] = now

        deps = EngineDeps(on_best_checkpoint=_on_best_checkpoint)

    log_path = Path(plan.local_results_dir) / "run.log"
    with reporting.Tee(log_path):
        model, result, splits = run_training(plan, cfg, deps)
        eval_results = reporting.evaluate_all_splits(
            model, splits, plan, result.device, default_tracker_factory
        )
        gap = reporting.generalization_gap(eval_results)
        if gap is not None:
            print(f"\nGeneralization gap (seen - unseen mDice): {gap:+.4f}")
        payload = reporting.build_metrics_payload(plan, result, eval_results, model)
        reporting.write_metrics_json(plan.local_results_dir, payload)
        reporting.save_mask_overlays(model, splits, plan, result.device, plan.local_results_dir)

    reporting.mirror_to_drive(plan.local_results_dir, plan.drive_results_dir)
    reporting.mirror_checkpoint_to_drive(plan.checkpoint_path, plan.drive_checkpoint_dir)
    print(f"Done. Checkpoint: {plan.checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
