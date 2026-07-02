# FILE MAP | Zero-shot (un-fine-tuned) SAM / MedSAM baselines for the "with vs without
#   fine-tuning" comparison. These load the RAW foundation model and segment via a prompt
#   derived from the ground-truth mask — no LoRA, no training, no CNN decoder. They are the
#   "without fine-tuning" half of the professor's 4-way matrix (see docs/PROJECT_PLAN.md).
#
#   Prompting protocol [TWEAK, but keep SAM and MedSAM identical so they stay comparable]:
#     - "box"   (default, recommended): tightest bounding box around the GT mask, padded a few px.
#                MedSAM is trained for box prompts; standard in the medical-SAM literature.
#                NOTE: this is an ORACLE-prompted baseline — it is told roughly where the polyp is,
#                whereas the fine-tuned models get no hint. Document that caveat in any write-up.
#     - "point" : single foreground point at the GT centroid. Weaker, but no box leakage.
#
#   Pure-numpy prompt derivation (box_from_mask/point_from_mask) is GPU-free and unit-tested.
#   torch + segment_anything are imported lazily so importing this module needs neither.
"""Zero-shot vanilla SAM / MedSAM inference baselines (prompted by the GT mask)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Prompt derivation from a ground-truth mask (pure numpy, GPU-free, testable)
# ---------------------------------------------------------------------------

def box_from_mask(mask: np.ndarray, padding: int = 0) -> Optional[np.ndarray]:
    """
    Tight XYXY bounding box around the foreground of a binary mask, padded and clamped
    to the image bounds. Returns None for an empty mask (no polyp to prompt).

    mask: H×W array; foreground is anything > 0.5.
    returns: np.array([x0, y0, x1, y1], float32) in pixel coords, or None.
    """
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(w - 1, int(xs.max()) + padding)
    y1 = min(h - 1, int(ys.max()) + padding)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def point_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Single foreground point at the GT centroid. Returns None for an empty mask.

    returns: np.array([[cx, cy]], float32) (SAM expects (N, 2) XY coords), or None.
    """
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return None
    return np.array([[float(xs.mean()), float(ys.mean())]], dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Zero-shot SAM/MedSAM wrapper (holds a SamPredictor)
# ---------------------------------------------------------------------------

class ZeroShotSAM:
    """
    Raw SAM (or MedSAM) run in inference-only mode, prompted from the GT mask.

    Not an nn.Module and not trainable by design — the whole point is 0 trainable params
    and 0 training cost. Build via build_zeroshot_sam / build_zeroshot_medsam.
    """

    def __init__(self, predictor, model_type: str, prompt: str = "box",
                 box_padding: int = 5, checkpoint: Optional[str] = None):
        if prompt not in ("box", "point"):
            raise ValueError(f"prompt must be 'box' or 'point', got {prompt!r}")
        self.predictor = predictor
        self.model_type = model_type
        self.prompt = prompt
        self.box_padding = box_padding
        self.checkpoint = checkpoint

    # --- size accounting (for the benchmark param table / metrics.json) ---
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.predictor.model.parameters())

    def trainable_parameters(self) -> int:
        # Zero by construction: nothing is optimized in a zero-shot baseline.
        return sum(p.numel() for p in self.predictor.model.parameters() if p.requires_grad)

    def checkpoint_size_mb(self) -> Optional[float]:
        if self.checkpoint and Path(self.checkpoint).exists():
            return round(Path(self.checkpoint).stat().st_size / 1e6, 2)
        return None

    # --- inference ---
    def predict_prob(self, image_uint8: np.ndarray, gt_binary: np.ndarray) -> np.ndarray:
        """
        Segment `image_uint8` (H×W×3 uint8 RGB) using a prompt derived from `gt_binary`
        (H×W in {0,1}). Returns an H×W probability map in [0, 1] (sigmoid of SAM's mask
        logits), matching the [0,1] convention MetricTracker.update expects.
        """
        h, w = gt_binary.shape[:2]
        if self.prompt == "box":
            box = box_from_mask(gt_binary, padding=self.box_padding)
            point_coords = point_labels = None
        else:
            box = None
            point_coords = point_from_mask(gt_binary)
            point_labels = None if point_coords is None else np.ones(len(point_coords), dtype=np.int64)

        if box is None and point_coords is None:  # empty GT → empty prediction
            return np.zeros((h, w), dtype=np.float32)

        self.predictor.set_image(image_uint8)
        masks, _scores, _logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None if box is None else box[None, :],
            multimask_output=False,
            return_logits=True,   # raw mask logits at image resolution
        )
        return _sigmoid(masks[0]).astype(np.float32)


def is_zeroshot(obj) -> bool:
    """True if obj is a zero-shot baseline (used by the benchmark to branch its eval loop)."""
    return isinstance(obj, ZeroShotSAM)


# ---------------------------------------------------------------------------
# Builders (lazy torch / segment_anything import — keeps the module import cheap)
# ---------------------------------------------------------------------------

def build_zeroshot_sam(
    checkpoint: str,
    model_type: str = "vit_h",
    device: str = "cuda",
    prompt: str = "box",
    box_padding: int = 5,
) -> ZeroShotSAM:
    """
    Load a raw SAM checkpoint (no LoRA, no decoder) and wrap it for prompted inference.
    Requires: pip install git+https://github.com/facebookresearch/segment-anything.git
    """
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        raise ImportError(
            "Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    for p in sam.parameters():          # freeze everything: zero-shot trains nothing
        p.requires_grad_(False)
    sam.to(device).eval()
    return ZeroShotSAM(SamPredictor(sam), model_type=model_type,
                       prompt=prompt, box_padding=box_padding, checkpoint=checkpoint)


def build_zeroshot_medsam(
    checkpoint: str,
    device: str = "cuda",
    prompt: str = "box",
    box_padding: int = 5,
) -> ZeroShotSAM:
    """MedSAM shares the SAM ViT-B architecture; only the weights differ."""
    return build_zeroshot_sam(checkpoint, model_type="vit_b", device=device,
                              prompt=prompt, box_padding=box_padding)


# ---------------------------------------------------------------------------
# Evaluation over all splits (mirrors reporting.evaluate_all_splits, GT-prompted)
# ---------------------------------------------------------------------------

def predict_zeroshot_prob(zs: ZeroShotSAM, image_path, mask_path, img_size: int):
    """
    Read a raw image/mask pair, resize both to img_size (so metrics are computed at the
    same resolution as the fine-tuned models), and return (prob H×W in [0,1], gt H×W in {0,1}).
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    img_np = np.array(img, dtype=np.uint8)
    gt = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.NEAREST)
    gt_np = (np.array(gt) > 127).astype(np.float32)
    prob = zs.predict_prob(img_np, gt_np)
    return prob, gt_np


def evaluate_zeroshot_all_splits(zs: ZeroShotSAM, splits: dict, img_size: int,
                                 tracker_factory, split_labels: dict | None = None) -> dict:
    """
    Evaluate a zero-shot baseline on every available split. Returns
    {split_key: {dice, iou, mae, wfm, sm, em}} — the same shape the trained models produce,
    so results merge straight into the benchmark's `all_results`.
    """
    keys = list(split_labels.keys()) if split_labels else list(splits.keys())
    results: dict[str, dict] = {}
    for key in keys:
        if key not in splits:
            continue
        tracker = tracker_factory()
        for ip, mp in zip(splits[key]["image_paths"], splits[key]["mask_paths"]):
            prob, gt = predict_zeroshot_prob(zs, ip, mp, img_size)
            tracker.update(prob, gt)
        results[key] = tracker.compute()
    return results
