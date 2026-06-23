"""
PolypDataset and split utilities.

Actual PraNet data package layout on disk:
  data/polyp/
    TrainDataset/
      image/   (1450 mixed images — 900 Kvasir + 550 CVC-ClinicDB, pre-packaged flat)
      masks/   (1450 corresponding masks)
    TestDataset/
      TestDataset/           ← zip extracts with an inner folder of the same name
        Kvasir/images/       100 seen-test images
        Kvasir/masks/
        CVC-ClinicDB/images/ 62 seen-test images
        CVC-ClinicDB/masks/
        CVC-ColonDB/images/  380 unseen
        CVC-ColonDB/masks/
        ETIS-LaribPolypDB/images/ 196 unseen
        ETIS-LaribPolypDB/masks/
        CVC-300/images/      60 unseen
        CVC-300/masks/
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


_TEST_DIRS = {
    "seen_kvasir":   "Kvasir",
    "seen_clinicdb":  "CVC-ClinicDB",
    "cvc_colondb":    "CVC-ColonDB",
    "etis_larib":     "ETIS-LaribPolypDB",
    "cvc_300":        "CVC-300",
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
            return image, mask.unsqueeze(0)

        return image, mask


def _collect_image_mask_pairs(folder: Path) -> Tuple[list[Path], list[Path]]:
    """
    Return sorted (image_paths, mask_paths) from a dataset folder.
    Handles both 'images/' and 'image/' subfolder naming.
    """
    # TrainDataset uses 'image' (singular); TestDataset subfolders use 'images'
    images_dir = None
    for name in ("images", "image"):
        candidate = folder / name
        if candidate.exists():
            images_dir = candidate
            break
    if images_dir is None:
        raise FileNotFoundError(f"No images/ or image/ dir in {folder}")

    masks_dir = folder / "masks"
    if not masks_dir.exists():
        raise FileNotFoundError(f"No masks/ dir in {folder}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    masks  = sorted(p for p in masks_dir.iterdir()  if p.suffix.lower() in exts)

    if len(images) != len(masks):
        raise RuntimeError(
            f"Image/mask count mismatch in {folder}: "
            f"{len(images)} images vs {len(masks)} masks"
        )
    return images, masks


def _find_test_root(data_root: Path) -> Path:
    """
    The PraNet TestDataset zip extracts with a nested TestDataset/TestDataset/ structure.
    Try the nested path first, fall back to flat.
    """
    nested = data_root / "TestDataset" / "TestDataset"
    if nested.exists():
        return nested
    return data_root / "TestDataset"


def build_splits(
    data_root: str | Path,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Build the standard PraNet train / seen-test / unseen-test splits.

    Train:        all 1450 images from TrainDataset/image/ (pre-packaged flat)
    Seen test:    TestDataset/Kvasir/ (100) + TestDataset/CVC-ClinicDB/ (62)
    Unseen test:  TestDataset/CVC-ColonDB/ (380) + ETIS-LaribPolypDB/ (196) + CVC-300/ (60)
    """
    data_root = Path(data_root)
    rng = random.Random(seed)

    splits: dict[str, dict] = {}

    # --- training: flat folder, all 1450 images together ---
    train_imgs, train_masks = _collect_image_mask_pairs(data_root / "TrainDataset")
    combined = list(zip(train_imgs, train_masks))
    rng.shuffle(combined)
    splits["train"] = {
        "image_paths": [p[0] for p in combined],
        "mask_paths":  [p[1] for p in combined],
    }

    # --- test splits ---
    test_root = _find_test_root(data_root)
    for key, folder_name in _TEST_DIRS.items():
        folder = test_root / folder_name
        imgs, masks = _collect_image_mask_pairs(folder)
        splits[key] = {"image_paths": imgs, "mask_paths": masks}

    return splits


def verify_splits(splits: dict) -> None:
    expected = {
        "train":        1450,
        "seen_kvasir":   100,
        "seen_clinicdb":  62,
        "cvc_colondb":   380,
        "etis_larib":    196,
        "cvc_300":        60,
    }
    print(f"{'Split':<20} {'Expected':>10} {'Found':>10} {'OK':>5}")
    print("-" * 48)
    all_ok = True
    for name, exp in expected.items():
        found = len(splits.get(name, {}).get("image_paths", []))
        ok = "✓" if found == exp else "✗"
        if ok == "✗":
            all_ok = False
        print(f"{name:<20} {exp:>10} {found:>10} {ok:>5}")
    print()
    print("All splits verified!" if all_ok else "Fix needed — check paths above.")
