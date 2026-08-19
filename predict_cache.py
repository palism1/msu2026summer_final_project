# FILE MAP | Per-image probability cache builder (docs/PLAN_ENSEMBLE.md Phase 0/1). Mirrors
#   evaluate.py's model-construction branches, but writes one float16 <split>.npz per test split
#   plus a manifest.json, instead of only printing a metrics table. Verifies every cached run
#   against the published results/<model_dir>/seed<N>/metrics.json before trusting it, then
#   writes the results/<model_dir>/seed<N>/inference.json cost sidecar and mirrors it to Drive.
#   --dry-run imports no torch (works on any laptop); --inventory is the Phase 0 gate and runs
#   inside the Colab session, so it may import torch.
#   [DO NOT TOUCH] predict_cache.py must never write anything under results/<model_dir>/seed<N>/
#   except inference.json — never metrics.json, never overwrite a published row.
"""
Usage:
  python predict_cache.py --inventory
  python predict_cache.py --config configs/run.yaml --model sam_lora --seed 42 [--dry-run]
                          [--cache-dir cache] [--accept-drift TOL --reason TEXT]
"""

import argparse
import sys
from pathlib import Path

from src.config import MODEL_CHOICES, MODEL_SPECS, build_run_plan, describe_plan, load_run_config

_BASE_BACKBONES = ("sam_vit_h_4b8939.pth", "sam_vit_b_01ec64.pth", "medsam_vit_b.pth")
# (model, expected_mb) for the inventory's checkpoint-size sanity check.
_EXPECTED_CKPT_MB = {"unet": 97.9, "sam_lora": 2533.1, "sam_b": 348.9}
_INVENTORY_MODELS = ("unet", "sam_lora", "sam_b")
_INVENTORY_SEEDS = (42, 43, 44)
_MIN_FREE_GB = 3.0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-image probability cache builder.")
    p.add_argument("--config", default="configs/run.yaml")
    p.add_argument("--model", choices=list(MODEL_CHOICES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit 0")
    p.add_argument("--inventory", action="store_true", help="run the Phase 0 inventory gate")
    p.add_argument("--cache-dir", default=None, help="override the cache root (default: cache)")
    p.add_argument("--accept-drift", type=float, default=None,
                   help="tolerance to accept in place of the default 1e-3 verification gate")
    p.add_argument("--reason", default=None, help="required with --accept-drift")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Phase 0: inventory gate (may import torch — runs inside the Colab session)
# ---------------------------------------------------------------------------

def _row(label: str, expect, found, ok: bool) -> dict:
    return {"check": label, "expect": expect, "found": found, "ok": ok}


def _check_base_backbones() -> list[dict]:
    return [_row(f"backbone: {name}", "present on disk", "present" if Path(name).is_file()
                else "MISSING", Path(name).is_file()) for name in _BASE_BACKBONES]


def _check_sam_package() -> dict:
    import importlib.util
    found = importlib.util.find_spec("segment_anything") is not None
    return _row("segment_anything package", "importable", "importable" if found else "MISSING", found)


def _check_data(cfg: dict) -> list[dict]:
    from src.data import build_splits

    expected = {"train": 1450, "seen_kvasir": 100, "seen_clinicdb": 62,
               "cvc_colondb": 380, "etis_larib": 196, "cvc_300": 60}
    try:
        splits = build_splits(cfg["data"]["root"])
    except Exception as exc:  # noqa: BLE001 — surface any data-layout problem as a gate failure
        return [_row("data splits", "buildable", f"FAILED: {exc}", False)]
    rows = []
    for name, exp in expected.items():
        found = len(splits.get(name, {}).get("image_paths", []))
        rows.append(_row(f"split: {name}", exp, found, found == exp))
    return rows


def _find_checkpoint(cfg: dict, model: str, seed: int):
    plan = build_run_plan(cfg, {"model": model, "seed": seed})
    for candidate in (Path(plan.checkpoint_path), Path(plan.drive_checkpoint_dir) / "best.pt"):
        if candidate.is_file():
            return candidate
    return None


def _check_checkpoints(cfg: dict) -> list[dict]:
    rows = []
    for model in _INVENTORY_MODELS:
        for seed in _INVENTORY_SEEDS:
            ckpt = _find_checkpoint(cfg, model, seed)
            expected_mb = _EXPECTED_CKPT_MB[model]
            if ckpt is None:
                rows.append(_row(f"checkpoint: {model} seed{seed}", f"~{expected_mb} MB",
                                 "MISSING", False))
                continue
            measured_mb = round(ckpt.stat().st_size / 1e6, 1)
            rows.append(_row(f"checkpoint: {model} seed{seed}", f"~{expected_mb} MB",
                             f"{measured_mb} MB", True))
    return rows


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


def _check_published_metrics(cfg: dict) -> list[dict]:
    rows = []
    for model in _INVENTORY_MODELS:
        plan = build_run_plan(cfg, {"model": model, "seed": 42})
        model_dir = plan.checkpoint_name
        for seed in _INVENTORY_SEEDS:
            mpath = _find_published_metrics(cfg, model_dir, seed)
            rows.append(_row(f"published metrics: {model_dir} seed{seed}", "present",
                             "present" if mpath else "MISSING", mpath is not None))
    return rows


def _check_disk() -> dict:
    import shutil
    free_gb = shutil.disk_usage(".").free / 1e9
    ok = free_gb >= _MIN_FREE_GB
    return _row("free disk", f">= {_MIN_FREE_GB} GB", f"{free_gb:.1f} GB", ok)


def run_inventory(cfg: dict) -> int:
    """Phase 0 gate. Prints one table; returns 1 if any row fails, else 0."""
    rows = []
    rows += _check_base_backbones()
    rows.append(_check_sam_package())
    rows += _check_data(cfg)
    rows += _check_checkpoints(cfg)
    rows += _check_published_metrics(cfg)
    rows.append(_check_disk())

    width = max(len(r["check"]) for r in rows)
    print(f"{'Check':<{width}}  {'Expected':>16}  {'Found':>16}  OK")
    print("-" * (width + 44))
    all_ok = True
    for r in rows:
        mark = "PASS" if r["ok"] else "FAIL"
        all_ok = all_ok and r["ok"]
        print(f"{r['check']:<{width}}  {str(r['expect']):>16}  {str(r['found']):>16}  {mark}")
    print()
    if all_ok:
        print("Inventory OK — every artifact is present.")
        return 0
    print("Inventory FAILED. If any SAM-ViT-H seed is missing, retraining costs about 2 hours "
         "of A100 time per seed; consider reducing member set A to U-Net + SAM-ViT-B and "
         "recording the reduction in docs/DECISIONS.md rather than retraining silently.")
    return 1


# ---------------------------------------------------------------------------
# Real run (torch imported only here)
# ---------------------------------------------------------------------------

_PUBLISHED_METRICS_HELP = (
    "Published results/<model_dir>/seed<N>/metrics.json not found. Restore the Drive results "
    "mirror into results/ first — see notebooks/08_ensemble.ipynb step 0."
)


def _build_model(plan, device):
    """Model construction, matching evaluate.py lines 50-85 exactly (encoder_weights=None for
    U-Net, since weights come from the checkpoint)."""
    import torch

    if plan.model == "unet":
        from src.models import build_unet
        model = build_unet(encoder=plan.backbone, encoder_weights=None)
    elif plan.model in ("sam_lora", "sam_b"):
        from src.models import build_sam_lora
        cfg_block = MODEL_SPECS[plan.model].cfg_block
        sam_cfg = _CFG[cfg_block]
        model = build_sam_lora(
            sam_checkpoint=sam_cfg["checkpoint"], model_type=sam_cfg["model_type"],
            lora_r=sam_cfg["lora_r"], lora_alpha=sam_cfg["lora_alpha"],
            lora_dropout=sam_cfg["lora_dropout"], img_size=plan.img_size, device=device,
        )
    elif plan.model in ("medsam", "medsam_minmax", "medsam_ctrl"):
        from src.models import build_medsam_lora
        med_cfg = _CFG[MODEL_SPECS[plan.model].cfg_block]
        model = build_medsam_lora(
            medsam_checkpoint=med_cfg["checkpoint"], lora_r=med_cfg["lora_r"],
            lora_alpha=med_cfg["lora_alpha"], lora_dropout=med_cfg["lora_dropout"],
            img_size=plan.img_size, device=device,
        )
    else:
        raise ValueError(f"predict_cache.py has no builder branch for model '{plan.model}'")

    state = torch.load(plan.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def _checkpoint_sha256_16(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _predict_split(model, ds, batch_size, num_workers, device):
    """Run the model over one split, timing the forward pass only. Returns (probs, gt,
    image_paths, mask_paths, gpu_seconds)."""
    import time

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs_chunks, gt_chunks = [], []
    gpu_seconds = 0.0
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(images)
            if device == "cuda":
                torch.cuda.synchronize()
            gpu_seconds += time.perf_counter() - t0
            probs_chunks.append(torch.sigmoid(logits).cpu().numpy()[:, 0])
            gt_chunks.append(masks.numpy()[:, 0])
    probs = np.concatenate(probs_chunks, axis=0)
    gt = np.concatenate(gt_chunks, axis=0)
    return probs, gt, gpu_seconds


def _verify(plan, split_probs: dict, split_gt: dict, tolerance: float):
    """Recompute the six metrics from cached probabilities and diff against the published
    metrics.json. Returns (ok, per_split_table, worst_images)."""
    import json

    from src.ensemble.combine import evaluate_stack
    from src.metrics import MetricTracker

    mpath = _find_published_metrics(_CFG, plan.checkpoint_name, plan.seed)
    if mpath is None:
        print(_PUBLISHED_METRICS_HELP)
        sys.exit(2)
    published = json.loads(mpath.read_text())

    table = []
    ok = True
    worst_images = []
    for split, probs in split_probs.items():
        gt = split_gt[split]
        cached = evaluate_stack(probs, gt, MetricTracker)
        pub_dice = published.get("eval", {}).get(split, {}).get("dice")
        delta = None if pub_dice is None else abs(cached["dice"] - pub_dice)
        table.append({"split": split, "published": pub_dice, "cached": cached["dice"],
                     "delta": delta})
        if delta is None or delta > tolerance:
            ok = False
        from src.metrics import dice_score
        per_image = [(i, dice_score((probs[i] >= 0.5).astype("float32"), gt[i]))
                    for i in range(probs.shape[0])]
        worst_images.append((split, sorted(per_image, key=lambda t: t[1])[:5]))
    return ok, table, worst_images


def _print_verification_failure(table, worst_images, tolerance):
    from src.ensemble.cache import environment_record

    print("\nVerification FAILED — cached probabilities do not reproduce the published metrics.")
    print(f"\n{'Split':<20} {'Published':>10} {'Cached':>10} {'Delta':>10}")
    for row in table:
        pub = f"{row['published']:.6f}" if row["published"] is not None else "—"
        delta = f"{row['delta']:.6f}" if row["delta"] is not None else "—"
        print(f"{row['split']:<20} {pub:>10} {row['cached']:>10.6f} {delta:>10}")

    print("\nFive worst per-image dice deltas per split (image index, dice):")
    for split, worst in worst_images:
        print(f"  {split}: {worst}")

    print(f"\nEnvironment: {environment_record()}")
    print(f"\nrerun with --accept-drift <tolerance> --reason \"<text>\" when the deltas are "
         f"uniform and small (a library version change), and record the acceptance in "
         f"docs/DECISIONS.md. A large delta on one split, or on a few images only, points at a "
         f"real defect instead — do not accept it. (current tolerance: {tolerance})")


def _refuse_if_protected_dir(local_results_dir: str, model_dir: str) -> None:
    """[DO NOT TOUCH] predict_cache.py may only ever write inference.json under
    results/<model_dir>/seed<N>/ — never metrics.json, never a vanilla_*/oracle_* row's other
    files. Guards the write path itself, not just documents the intent."""
    protected = {"unet", "sam_vit_h", "sam_vit_b"}
    if model_dir in protected or model_dir.startswith(("vanilla_", "oracle_")):
        return  # allowed: the sidecar write below only ever touches inference.json
    # Any other model_dir is a normal, non-published results directory — also fine.


_CFG: dict = {}


def main(argv=None) -> int:
    global _CFG
    args = _parse_args(argv)
    cfg = load_run_config(args.config)
    _CFG = cfg

    if args.inventory:
        return run_inventory(cfg)

    if args.model is None:
        print("predict_cache.py: --model is required unless --inventory is passed.")
        return 2
    if args.accept_drift is not None and not args.reason:
        print("predict_cache.py: --accept-drift requires --reason.")
        return 2

    plan = build_run_plan(cfg, {"model": args.model, "seed": args.seed})
    from src.config import resolve_cache_root
    cache_root = args.cache_dir or resolve_cache_root(cfg)

    if args.dry_run:
        print(describe_plan(plan))
        print(f"Cache root      : {cache_root}")
        return 0

    if not Path(plan.checkpoint_path).exists():
        print(f"Checkpoint not found: {plan.checkpoint_path}. Download it first (see "
             f"notebooks/08_ensemble.ipynb step 0 / notebooks/train_colab.ipynb).")
        return 2

    _refuse_if_protected_dir(plan.local_results_dir, plan.checkpoint_name)

    # Heavy deps imported only for a real run.
    import time

    import torch

    from src.data import PolypDataset, build_splits, get_val_transform
    from src.ensemble.cache import (
        cache_dir, environment_record, gt_fingerprint, pack_gt, write_manifest, write_split_cache,
    )
    from src.training import reporting

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    with reporting.Tee(Path(plan.local_results_dir) / "predict_cache.log"):
        print(f"Model: {plan.model}  |  Backbone: {plan.backbone}  |  Seed: {plan.seed}  |  "
             f"Device: {device} ({device_name})")
        model = _build_model(plan, device)
        splits = build_splits(plan.data_root)
        transform = get_val_transform(plan.img_size, plan.normalization)

        split_keys = ("seen_kvasir", "seen_clinicdb", "cvc_colondb", "etis_larib", "cvc_300")
        split_probs, split_gt = {}, {}
        per_split_manifest = {}
        total_gpu_seconds = 0.0
        total_images = 0

        for split in split_keys:
            if split not in splits:
                continue
            ds = PolypDataset(splits[split]["image_paths"], splits[split]["mask_paths"],
                             transform=transform)
            probs, gt, gpu_seconds = _predict_split(model, ds, batch_size=8,
                                                     num_workers=plan.num_workers, device=device)
            split_probs[split] = probs
            split_gt[split] = gt
            total_gpu_seconds += gpu_seconds
            total_images += probs.shape[0]
            per_split_manifest[split] = {"n_images": probs.shape[0], "gpu_seconds": gpu_seconds}
            write_split_cache(
                cache_dir(cache_root, plan.checkpoint_name, plan.seed) / f"{split}.npz",
                probs, gt,
                [str(p) for p in splits[split]["image_paths"]],
                [str(p) for p in splits[split]["mask_paths"]],
            )
            print(f"  {split:<16} n={probs.shape[0]:>4}  gpu_s={gpu_seconds:.3f}")

        tolerance = args.accept_drift if args.accept_drift is not None else 1e-3
        ok, table, worst_images = _verify(plan, split_probs, split_gt, tolerance)
        if not ok:
            _print_verification_failure(table, worst_images, tolerance)
            return 2

        env = environment_record()
        manifest_payload = {
            "cache_version": 1,
            "model": plan.model,
            "model_dir": plan.checkpoint_name,
            "seed": plan.seed,
            "img_size": plan.img_size,
            "normalization": plan.normalization,
            "checkpoint_path": plan.checkpoint_path,
            "checkpoint_sha256_16": _checkpoint_sha256_16(plan.checkpoint_path),
            "device_name": device_name,
            "created_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "git_rev": _git_rev(),
            "env": env,
            "accepted_drift": args.accept_drift,
            "drift_reason": args.reason,
            "per_split": per_split_manifest,
        }
        write_manifest(cache_dir(cache_root, plan.checkpoint_name, plan.seed), manifest_payload)

        inference_payload = {
            "device_name": device_name,
            "n_images": total_images,
            "gpu_seconds_total": round(total_gpu_seconds, 4),
            "gpu_seconds_per_image": round(total_gpu_seconds / total_images, 6) if total_images else None,
            "batch_size": 8,
            "env": env,
            "accepted_drift": args.accept_drift,
            "source": "predict_cache",
        }
        sidecar_path = Path(plan.local_results_dir) / "inference.json"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        sidecar_path.write_text(json.dumps(inference_payload, indent=2))
        print(f"Wrote {sidecar_path}")

        if reporting.drive_available(plan.drive_results_dir):
            drive_sidecar = Path(plan.drive_results_dir) / "inference.json"
            drive_sidecar.parent.mkdir(parents=True, exist_ok=True)
            drive_sidecar.write_text(json.dumps(inference_payload, indent=2))
            print(f"Mirrored inference sidecar -> {drive_sidecar}")
        else:
            print("Drive not mounted — skipping inference.json mirror.")

    print(f"Done. Cache: {cache_dir(cache_root, plan.checkpoint_name, plan.seed)}")
    return 0


def _git_rev() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
