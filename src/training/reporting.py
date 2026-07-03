# FILE MAP | Results I/O for a training run (kept out of engine.py so the loop stays pure-ish).
#   Produces the three deliverables the Colab run must leave behind, then mirrors them to Drive:
#     - results/<model>/seed<seed>/metrics.json   (accuracy + efficiency: params, ckpt size, timing)
#     - results/<model>/seed<seed>/overlay_*.png  (sample image / GT / prediction panels)
#     - results/<model>/seed<seed>/run.log        (full stdout tee)
#   Also mirrors the best checkpoint (checkpoints/<model>/seed<seed>/best.pt) to Drive via
#   mirror_checkpoint_to_drive, both periodically (engine.py callback) and once at the end.
#   matplotlib/PIL are imported lazily so importing this module stays cheap. [TWEAK] overlay
#   styling and the metrics payload shape are safe to extend.
"""Evaluation, metric aggregation, mask-overlay rendering, run logging, and Drive mirroring."""

from __future__ import annotations

import io
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Same split labels/keys used by evaluate.py and the benchmark notebook.
SPLIT_LABELS = {
    "seen_kvasir": "Seen — Kvasir",
    "seen_clinicdb": "Seen — CVC-ClinicDB",
    "cvc_colondb": "Unseen — CVC-ColonDB",
    "etis_larib": "Unseen — ETIS-Larib",
    "cvc_300": "Unseen — CVC-300",
}
_SEEN = ("seen_kvasir", "seen_clinicdb")
_UNSEEN = ("cvc_colondb", "etis_larib", "cvc_300")


# ---------------------------------------------------------------------------
# stdout tee -> run.log
# ---------------------------------------------------------------------------

class Tee:
    """Context manager that duplicates stdout/stderr into a log file."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: io.TextIOBase | None = None
        self._streams: list = []

    def _make_writer(self, original):
        fh = self._fh

        class _Writer:
            def write(_self, data):
                original.write(data)
                fh.write(data)
                return len(data)

            def flush(_self):
                original.flush()
                fh.flush()

            def __getattr__(_self, name):
                # Delegate isatty/encoding/fileno/etc. to the real stream.
                return getattr(original, name)

        return _Writer()

    def __enter__(self):
        self._fh = open(self.log_path, "a")
        self._fh.write(f"\n===== run started {datetime.now(timezone.utc).isoformat()} =====\n")
        self._orig_out, self._orig_err = sys.stdout, sys.stderr
        sys.stdout = self._make_writer(self._orig_out)
        sys.stderr = self._make_writer(self._orig_err)
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._orig_out, self._orig_err
        if self._fh:
            self._fh.flush()
            self._fh.close()
        return False


# ---------------------------------------------------------------------------
# Evaluation over all 5 splits
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_all_splits(model, splits, plan, device, tracker_factory, batch_size: int = 8) -> dict:
    """Evaluate the model on every available test split; returns per-split score dicts."""
    from src.data import PolypDataset, get_val_transform

    model.eval()
    transform = get_val_transform(plan.img_size)
    results: dict[str, dict] = {}
    print(f"\n{'Split':<26} {'mDice':>7} {'mIoU':>7} {'MAE':>7} {'wFm':>7} {'Sm':>7} {'Em':>7}")
    print("-" * 75)
    for key, label in SPLIT_LABELS.items():
        if key not in splits:
            continue
        ds = PolypDataset(splits[key]["image_paths"], splits[key]["mask_paths"], transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=plan.num_workers)
        tracker = tracker_factory()
        for images, masks in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            gt = masks.numpy()
            for i in range(len(probs)):
                tracker.update(probs[i, 0], gt[i, 0])
        sc = tracker.compute()
        results[key] = sc
        print(f"{label:<26} {sc['dice']:>7.4f} {sc['iou']:>7.4f} {sc['mae']:>7.4f} "
              f"{sc['wfm']:>7.4f} {sc['sm']:>7.4f} {sc['em']:>7.4f}")
    return results


def generalization_gap(results: dict) -> float | None:
    """Mean seen mDice − mean unseen mDice, or None if splits are missing."""
    seen = [results[k]["dice"] for k in _SEEN if k in results]
    unseen = [results[k]["dice"] for k in _UNSEEN if k in results]
    if not seen or not unseen:
        return None
    return sum(seen) / len(seen) - sum(unseen) / len(unseen)


# ---------------------------------------------------------------------------
# Metrics payload (accuracy + efficiency)
# ---------------------------------------------------------------------------

def build_metrics_payload(plan, result, eval_results: dict, model) -> dict:
    """Assemble the metrics.json payload: accuracy AND the efficiency dimensions."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt = Path(plan.checkpoint_path)
    ckpt_bytes = ckpt.stat().st_size if ckpt.exists() else None
    gap = generalization_gap(eval_results)

    return {
        "model": plan.model,
        "backbone": plan.backbone,
        "seed": plan.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": result.device,
        "device_name": result.device_name,
        "params": {
            "total": total,
            "trainable": trainable,
            "trainable_pct": round(100 * trainable / total, 4) if total else None,
        },
        # Download/footprint metric: lets a reader weigh accuracy vs a simpler machine.
        "checkpoint_size_bytes": ckpt_bytes,
        "checkpoint_size_mb": round(ckpt_bytes / 1e6, 2) if ckpt_bytes else None,
        "timing": {
            "epochs_run": result.epochs_run,
            "total_seconds": round(result.total_seconds, 2),
            "mean_epoch_seconds": round(
                sum(result.epoch_seconds) / len(result.epoch_seconds), 2
            ) if result.epoch_seconds else None,
            "epoch_seconds": [round(s, 2) for s in result.epoch_seconds],
            "stopped_reason": result.stopped_reason,
        },
        "best_val_dice": round(result.best_dice, 6),
        "eval": {k: {m: round(v, 6) for m, v in sc.items()} for k, sc in eval_results.items()},
        "generalization_gap_dice": round(gap, 6) if gap is not None else None,
    }


def write_metrics_json(out_dir: str | Path, payload: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Mask overlays
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_mask_overlays(model, splits, plan, device, out_dir: str | Path) -> list[Path]:
    """Save image / ground-truth / prediction panels for a few samples per overlay split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from src.data import get_val_transform
    from src.metrics import dice_score

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = get_val_transform(plan.img_size)
    model.eval()
    saved: list[Path] = []

    for split_key in plan.overlay_splits:
        if split_key not in splits:
            continue
        img_paths = splits[split_key]["image_paths"][: plan.n_overlay_samples]
        msk_paths = splits[split_key]["mask_paths"][: plan.n_overlay_samples]
        for idx, (ip, mp) in enumerate(zip(img_paths, msk_paths)):
            raw_img = np.array(Image.open(ip).convert("RGB"))
            raw_msk = (np.array(Image.open(mp).convert("L")) > 127).astype(np.float32)
            aug = transform(image=raw_img, mask=raw_msk)
            inp = aug["image"].unsqueeze(0).to(device)
            prob = torch.sigmoid(model(inp))[0, 0].cpu().numpy()
            pred = (prob >= 0.5).astype(np.float32)
            d = dice_score(pred, aug["mask"].numpy())

            fig, ax = plt.subplots(1, 3, figsize=(11, 4))
            ax[0].imshow(raw_img); ax[0].set_title("Image"); ax[0].axis("off")
            ax[1].imshow(raw_msk, cmap="gray"); ax[1].set_title("Ground truth"); ax[1].axis("off")
            ax[2].imshow(pred, cmap="gray"); ax[2].set_title(f"Pred (Dice={d:.3f})"); ax[2].axis("off")
            fig.suptitle(f"{plan.model} — {SPLIT_LABELS.get(split_key, split_key)}", fontweight="bold")
            fig.tight_layout()
            path = out_dir / f"overlay_{split_key}_{idx:02d}.png"
            fig.savefig(path, dpi=90, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    print(f"Saved {len(saved)} mask overlays to {out_dir}")
    return saved


# ---------------------------------------------------------------------------
# Drive mirror
# ---------------------------------------------------------------------------

def drive_available(drive_dir: str | Path) -> bool:
    """
    True if the Drive mount root for ``drive_dir`` is present (or no mount root applies).

    Walks ``drive_dir``'s parents looking for a directory ending in "MyDrive" or named
    "drive" (the Colab mount point). If no such parent exists in the path at all, there's
    nothing to guard against, so this returns True. If found but missing on disk, Drive
    isn't mounted (e.g. a local/offline run) and this returns False. Callers decide what,
    if anything, to print.
    """
    drive_dir = Path(drive_dir)
    mount_root = None
    for parent in drive_dir.parents:
        if str(parent).endswith("MyDrive") or parent.name == "drive":
            mount_root = parent
            break
    if mount_root is not None and not mount_root.exists():
        return False
    return True


def mirror_to_drive(local_dir: str | Path, drive_dir: str | Path) -> bool:
    """
    Copy the local results folder to the Drive results path, if Drive is mounted.

    Guarded like the existing checkpoint-backup cells: only mirrors when the Drive mount root
    (e.g. /content/drive/MyDrive) actually exists, so local/offline runs are a no-op.
    """
    drive_dir = Path(drive_dir)
    if not drive_available(drive_dir):
        mount_root = next(
            (p for p in drive_dir.parents if str(p).endswith("MyDrive") or p.name == "drive"), None
        )
        print(f"Drive not mounted ({mount_root} missing) — skipping Drive mirror.")
        return False
    drive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_dir, drive_dir, dirs_exist_ok=True)
    print(f"Mirrored results -> {drive_dir}")
    return True


def mirror_checkpoint_to_drive(checkpoint_path: str | Path, drive_checkpoint_dir: str | Path) -> bool:
    """
    Copy a single checkpoint file to the Drive checkpoint dir, if present and Drive is mounted.

    No-op (with a printed skip message) when the checkpoint doesn't exist yet or Drive isn't
    mounted — safe to call periodically during training and again at the end of a run.
    """
    checkpoint_path = Path(checkpoint_path)
    drive_checkpoint_dir = Path(drive_checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found ({checkpoint_path}) — skipping Drive checkpoint mirror.")
        return False
    if not drive_available(drive_checkpoint_dir):
        print(f"Drive not mounted — skipping Drive checkpoint mirror of {checkpoint_path}.")
        return False
    drive_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = drive_checkpoint_dir / checkpoint_path.name
    shutil.copy2(checkpoint_path, dest)
    print(f"Mirrored checkpoint -> {dest}")
    return True
