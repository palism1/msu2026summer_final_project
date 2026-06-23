"""
Evaluation script — runs all five test splits and prints a results table.

Usage:
  python evaluate.py --config configs/base.yaml --model unet \
                     --checkpoint checkpoints/unet/seed42/best.pt
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import PolypDataset, build_splits, get_val_transform
from src.metrics import MetricTracker


@torch.no_grad()
def evaluate_split(model, loader, device) -> dict:
    model.eval()
    tracker = MetricTracker()
    for images, masks in tqdm(loader, leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        gt = masks.numpy()
        for i in range(len(probs)):
            tracker.update(probs[i, 0], gt[i, 0])
    return tracker.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="unet", choices=["unet", "sam_lora"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = cfg["data"]["img_size"]

    # --- load model ---
    if args.model == "unet":
        from src.models import build_unet
        model = build_unet(
            encoder=cfg["model"]["encoder"],
            encoder_weights=None,  # weights loaded from checkpoint
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
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    # --- splits ---
    splits = build_splits(cfg["data"]["root"], seed=args.seed)
    val_transform = get_val_transform(img_size)

    split_names = {
        "Seen — Kvasir": "seen_kvasir",
        "Seen — CVC-ClinicDB": "seen_clinicdb",
        "Unseen — CVC-ColonDB": "cvc_colondb",
        "Unseen — ETIS-Larib": "etis_larib",
        "Unseen — CVC-300": "cvc_300",
    }

    print(f"\n{'Split':<26} {'mDice':>7} {'mIoU':>7} {'MAE':>7} {'wFm':>7} {'Sm':>7} {'Em':>7}")
    print("-" * 75)

    results = {}
    for label, key in split_names.items():
        ds = PolypDataset(
            splits[key]["image_paths"],
            splits[key]["mask_paths"],
            transform=val_transform,
        )
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
        scores = evaluate_split(model, loader, device)
        results[key] = scores
        print(
            f"{label:<26} "
            f"{scores['dice']:>7.4f} {scores['iou']:>7.4f} "
            f"{scores['mae']:>7.4f} {scores['wfm']:>7.4f} "
            f"{scores['sm']:>7.4f} {scores['em']:>7.4f}"
        )

    # generalization gap: mean seen - mean unseen
    seen_dice = (results["seen_kvasir"]["dice"] + results["seen_clinicdb"]["dice"]) / 2
    unseen_dice = (
        results["cvc_colondb"]["dice"] + results["etis_larib"]["dice"] + results["cvc_300"]["dice"]
    ) / 3
    print(f"\nGeneralization gap (seen − unseen mDice): {seen_dice - unseen_dice:+.4f}")


if __name__ == "__main__":
    main()
