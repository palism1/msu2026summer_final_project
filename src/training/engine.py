# FILE MAP | Training loop (torch lives here, imported only on real runs).
#   Extracted from notebooks 02/03/04 so every model trains under one identical protocol.
#   Dependencies (model builder, dataloader builder, metric-tracker factory) are injected via
#   EngineDeps so the orchestration is swappable/mockable.
#   [DO NOT TOUCH] the best-only checkpoint contract in run_training(): save state_dict to
#   <plan.checkpoint_dir>/best.pt only when val Dice improves. evaluate.py and
#   notebooks/05_benchmark.ipynb depend on exactly this path + raw state_dict format.
"""Config-driven training engine with dependency injection and wall-clock timing."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import RunPlan


# ---------------------------------------------------------------------------
# Reproducibility & loss  (identical to the notebooks — [DO NOT TOUCH] semantics)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Combined Dice + BCE. Same formula used across all models for fair comparison."""
    probs = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    inter = (probs * targets).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + eps) / (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
    return bce + dice.mean()


# ---------------------------------------------------------------------------
# Default injectable dependencies
# ---------------------------------------------------------------------------

def default_build_model(plan: RunPlan, cfg: dict, device: str) -> nn.Module:
    """Dispatch to the existing model builders (reused unchanged)."""
    if plan.model == "unet":
        from src.models import build_unet
        model = build_unet(
            encoder=cfg["model"]["encoder"],
            encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
        )
    elif plan.model == "sam_lora":
        from src.models import build_sam_lora
        sam = cfg["sam"]
        model = build_sam_lora(
            sam_checkpoint=sam["checkpoint"], model_type=sam["model_type"],
            lora_r=sam["lora_r"], lora_alpha=sam["lora_alpha"], lora_dropout=sam["lora_dropout"],
            img_size=plan.img_size, device=device,
        )
    elif plan.model == "medsam":
        from src.models import build_medsam_lora
        med = cfg["medsam"]
        model = build_medsam_lora(
            medsam_checkpoint=med["checkpoint"],
            lora_r=med["lora_r"], lora_alpha=med["lora_alpha"], lora_dropout=med["lora_dropout"],
            img_size=plan.img_size, device=device,
        )
    else:  # pragma: no cover - guarded upstream by build_run_plan
        raise ValueError(f"Unknown model '{plan.model}'")
    return model.to(device)


def default_build_dataloaders(plan: RunPlan, cfg: dict):
    """Build train/val loaders and the full split dict (same protocol as the notebooks)."""
    from src.data import PolypDataset, build_splits, get_train_transform, get_val_transform

    splits = build_splits(plan.data_root, seed=plan.seed)
    train_ds = PolypDataset(
        splits["train"]["image_paths"], splits["train"]["mask_paths"],
        transform=get_train_transform(plan.img_size),
    )
    val_ds = PolypDataset(
        splits["seen_kvasir"]["image_paths"] + splits["seen_clinicdb"]["image_paths"],
        splits["seen_kvasir"]["mask_paths"] + splits["seen_clinicdb"]["mask_paths"],
        transform=get_val_transform(plan.img_size),
    )
    train_loader = DataLoader(
        train_ds, batch_size=plan.batch_size, shuffle=True,
        num_workers=plan.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=plan.batch_size, shuffle=False,
        num_workers=plan.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, splits


def default_tracker_factory():
    from src.metrics import MetricTracker
    return MetricTracker()


@dataclass
class EngineDeps:
    """Injectable dependencies; defaults use the real implementations."""
    build_model: Callable = default_build_model
    build_dataloaders: Callable = default_build_dataloaders
    tracker_factory: Callable = default_tracker_factory
    loss_fn: Callable = dice_bce_loss


# ---------------------------------------------------------------------------
# Epoch steps
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, loss_fn) -> float:
    model.train()
    total = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), masks)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, tracker_factory) -> dict:
    model.eval()
    tracker = tracker_factory()
    for images, masks in loader:
        probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
        gt = masks.numpy()
        for i in range(len(probs)):
            tracker.update(probs[i, 0], gt[i, 0])
    return tracker.compute()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    best_dice: float
    epochs_run: int
    device: str
    device_name: str
    total_seconds: float
    epoch_seconds: list = field(default_factory=list)
    history: dict = field(default_factory=dict)
    stopped_reason: str = "completed"


def run_training(plan: RunPlan, cfg: dict, deps: Optional[EngineDeps] = None):
    """
    Train one model under the resolved plan. Returns (model, TrainResult, splits).

    Best-only checkpointing and the loss/optimizer/schedule match the notebooks exactly.
    Adds per-epoch + total wall-clock timing and an optional wall-clock budget so runtime
    cost is measurable across models. ``splits`` is returned so the caller can evaluate and
    render overlays without rebuilding them.
    """
    deps = deps or EngineDeps()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    set_seed(plan.seed)
    print(f"Device: {device} ({device_name})  |  Seed: {plan.seed}  |  Model: {plan.model}")

    train_loader, val_loader, splits = deps.build_dataloaders(plan, cfg)
    model = deps.build_model(plan, cfg, device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=plan.lr, weight_decay=plan.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=plan.epochs)

    ckpt_dir = Path(plan.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_dice": [], "val_iou": []}
    epoch_seconds: list[float] = []
    best_dice, patience_count = 0.0, 0
    stopped_reason = "completed"
    run_start = time.time()

    for epoch in range(1, plan.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, deps.loss_fn)
        scores = validate(model, val_loader, device, deps.tracker_factory)
        scheduler.step()
        dt = time.time() - t0
        epoch_seconds.append(dt)

        history["train_loss"].append(train_loss)
        history["val_dice"].append(scores["dice"])
        history["val_iou"].append(scores["iou"])
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_dice={scores['dice']:.4f} | "
            f"val_iou={scores['iou']:.4f} | {dt:.0f}s"
        )

        if scores["dice"] > best_dice:
            best_dice = scores["dice"]
            patience_count = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  new best dice={best_dice:.4f} -> checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= plan.patience:
                stopped_reason = "early_stop"
                print(f"Early stopping at epoch {epoch}")
                break

        if plan.max_train_minutes is not None and (time.time() - run_start) >= plan.max_train_minutes * 60:
            stopped_reason = "time_budget"
            print(f"Reached time budget ({plan.max_train_minutes:g} min) at epoch {epoch}")
            break

    total_seconds = time.time() - run_start
    print(f"\nTraining {stopped_reason}. Best val Dice: {best_dice:.4f}  |  {total_seconds:.0f}s total")
    result = TrainResult(
        best_dice=best_dice, epochs_run=len(epoch_seconds), device=device, device_name=device_name,
        total_seconds=total_seconds, epoch_seconds=epoch_seconds, history=history,
        stopped_reason=stopped_reason,
    )
    return model, result, splits
