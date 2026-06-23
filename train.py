"""
Training entry point.

Usage:
  python train.py --config configs/base.yaml --model unet --seed 42
  python train.py --config configs/base.yaml --model sam_lora --seed 42
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import PolypDataset, build_splits, get_train_transform, get_val_transform
from src.metrics import MetricTracker


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    inter = (probs * targets).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + eps) / (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
    return bce + dice.mean()


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, leave=False, desc="train"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = dice_bce_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, device) -> dict:
    model.eval()
    tracker = MetricTracker()
    for images, masks in tqdm(loader, leave=False, desc="val"):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        gt = masks.cpu().numpy()
        for i in range(len(probs)):
            tracker.update(probs[i, 0], gt[i, 0])
    return tracker.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="unet", choices=["unet", "sam_lora"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Seed: {args.seed}")

    # --- data ---
    img_size = cfg["data"]["img_size"]
    splits = build_splits(cfg["data"]["root"], seed=args.seed)

    train_ds = PolypDataset(
        splits["train"]["image_paths"],
        splits["train"]["mask_paths"],
        transform=get_train_transform(img_size),
    )
    val_ds = PolypDataset(
        splits["seen_kvasir"]["image_paths"] + splits["seen_clinicdb"]["image_paths"],
        splits["seen_kvasir"]["mask_paths"] + splits["seen_clinicdb"]["mask_paths"],
        transform=get_val_transform(img_size),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    # --- model ---
    if args.model == "unet":
        from src.models import build_unet
        model = build_unet(
            encoder=cfg["model"]["encoder"],
            encoder_weights=cfg["model"]["encoder_weights"],
        )
    elif args.model == "sam_lora":
        from src.models import build_sam_lora
        sam_cfg = cfg["sam"]
        model = build_sam_lora(
            sam_checkpoint=sam_cfg["checkpoint"],
            model_type=sam_cfg["model_type"],
            lora_r=sam_cfg["lora_r"],
            lora_alpha=sam_cfg["lora_alpha"],
            lora_dropout=sam_cfg["lora_dropout"],
            img_size=img_size,
            device=device,
        )
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {n_total:,}  trainable: {n_train:,} "
          f"({100*n_train/n_total:.1f}%)")

    # --- optimiser / scheduler ---
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    # --- training loop ---
    output_dir = Path(args.output_dir) / args.model / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dice, patience_count = 0.0, 0
    patience = cfg["training"]["early_stop_patience"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_scores = val_epoch(model, val_loader, device)
        scheduler.step()

        dice = val_scores["dice"]
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"val_dice={dice:.4f} | val_iou={val_scores['iou']:.4f} | "
            f"{elapsed:.0f}s"
        )

        if dice > best_dice:
            best_dice = dice
            patience_count = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  ↑ New best dice={best_dice:.4f} — checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining done. Best val dice: {best_dice:.4f}")
    print(f"Checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
