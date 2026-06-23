"""
PolypDataset and split utilities.

Expected data layout after download:
  data/polyp/
    TrainDataset/
      Kvasir/images/*.png   Kvasir/masks/*.png
      CVC-ClinicDB/images/*.png   CVC-ClinicDB/masks/*.png
    TestDataset/
      Kvasir/images/*.png   Kvasir/masks/*.png
      CVC-ClinicDB/images/*.png   CVC-ClinicDB/masks/*.png
      CVC-ColonDB/images/*.png    CVC-ColonDB/masks/*.png
      ETIS-Larib/images/*.png     ETIS-Larib/masks/*.png
      CVC-300/images/*.png        CVC-300/masks/*.png

This follows the PraNet / Polyp-PVT standard protocol exactly.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# Dataset folder names as they appear on disk after the PraNet data package
_DATASET_DIR = {
    "kvasir": "Kvasir",
    "cvc_clinicdb": "CVC-ClinicDB",
    "cvc_colondb": "CVC-ColonDB",
    "etis_larib": "ETIS-Larib",
    "cvc_300": "CVC-300",
}

# PraNet protocol: how many images go to training vs seen test
_PRANET_SPLITS = {
    "kvasir": (900, 100),
    "cvc_clinicdb": (550, 62),
}


class PolypDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        transform: Optional[Callable] = None,
    ):
        assert len(image_paths) == len(mask_paths), "image / mask count mismatch"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            # albumentations ToTensorV2 returns HWC→CHW for image, H×W for mask
            return image, mask.unsqueeze(0)

        # fallback when no transform: return numpy arrays (testing / inspection)
        return image, mask


def _collect_image_mask_pairs(folder: Path) -> Tuple[list[Path], list[Path]]:
    """Return sorted (image_paths, mask_paths) from a dataset subfolder."""
    images_dir = folder / "images"
    masks_dir = folder / "masks"
    if not images_dir.exists():
        raise FileNotFoundError(f"Expected images dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Expected masks dir not found: {masks_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    masks = sorted(p for p in masks_dir.iterdir() if p.suffix.lower() in exts)

    if len(images) != len(masks):
        raise RuntimeError(
            f"Image/mask count mismatch in {folder}: "
            f"{len(images)} images vs {len(masks)} masks"
        )
    return images, masks


def build_splits(
    data_root: str | Path,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Build the standard PraNet train/val/unseen splits.

    Returns a dict with keys:
      'train'         – 900 Kvasir + 550 CVC-ClinicDB
      'seen_kvasir'   – 100 Kvasir test images
      'seen_clinicdb' – 62 CVC-ClinicDB test images
      'cvc_colondb'   – 380 unseen
      'etis_larib'    – 196 unseen
      'cvc_300'       – 60 unseen
    Each value: {'image_paths': [...], 'mask_paths': [...]}
    """
    data_root = Path(data_root)
    rng = random.Random(seed)

    splits: dict[str, dict] = {}

    # --- training datasets (split into train / seen-test) ---
    train_imgs, train_masks = [], []
    for key, (n_train, n_test) in _PRANET_SPLITS.items():
        folder = data_root / "TrainDataset" / _DATASET_DIR[key]
        imgs, masks = _collect_image_mask_pairs(folder)

        # deterministic shuffle then slice
        paired = list(zip(imgs, masks))
        rng.shuffle(paired)
        assert len(paired) >= n_train + n_test, (
            f"{key}: need {n_train + n_test} images, found {len(paired)}"
        )

        split_name = "seen_kvasir" if key == "kvasir" else "seen_clinicdb"
        test_pairs = paired[n_train: n_train + n_test]
        splits[split_name] = {
            "image_paths": [p[0] for p in test_pairs],
            "mask_paths": [p[1] for p in test_pairs],
        }

        train_pairs = paired[:n_train]
        train_imgs.extend(p[0] for p in train_pairs)
        train_masks.extend(p[1] for p in train_pairs)

    # shuffle combined training set
    combined = list(zip(train_imgs, train_masks))
    rng.shuffle(combined)
    splits["train"] = {
        "image_paths": [p[0] for p in combined],
        "mask_paths": [p[1] for p in combined],
    }

    # --- unseen test datasets (use all images, never touched during training) ---
    for key in ("cvc_colondb", "etis_larib", "cvc_300"):
        folder = data_root / "TestDataset" / _DATASET_DIR[key]
        imgs, masks = _collect_image_mask_pairs(folder)
        splits[key] = {"image_paths": imgs, "mask_paths": masks}

    return splits


def verify_splits(splits: dict) -> None:
    """Print a table of split sizes and assert against expected counts."""
    expected = {
        "train": 1450,
        "seen_kvasir": 100,
        "seen_clinicdb": 62,
        "cvc_colondb": 380,
        "etis_larib": 196,
        "cvc_300": 60,
    }
    print(f"{'Split':<20} {'Expected':>10} {'Found':>10} {'OK':>5}")
    print("-" * 48)
    for name, exp in expected.items():
        found = len(splits.get(name, {}).get("image_paths", []))
        ok = "✓" if found == exp else "✗"
        print(f"{name:<20} {exp:>10} {found:>10} {ok:>5}")
