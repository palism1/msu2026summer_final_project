"""
Evaluation metrics for binary polyp segmentation.

All scalar functions operate on numpy arrays shaped (H, W) with values in [0, 1].
MetricTracker accumulates per-image scores across a dataset and reports means.

Metrics match the PraNet evaluation protocol:
  - mean Dice (mDice)
  - mean IoU  (mIoU)
  - Weighted Fβ (wFβ, β²=0.3)   — Margolin et al. 2014
  - S-measure (Sm, α=0.5)        — Fan et al. ICCV 2017
  - E-measure (Em)               — Fan et al. IJCAI 2018
  - Mean Absolute Error (MAE)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core scalar metrics (operate on single H×W numpy arrays)
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    inter = (pred * gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)


def mae_score(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean())


def weighted_f_measure(
    pred: np.ndarray,
    gt: np.ndarray,
    beta_sq: float = 0.3,
    eps: float = 1e-6,
) -> float:
    """
    Weighted Fβ measure (Margolin et al. 2014).
    β²=0.3 weights precision more than recall, following PraNet convention.
    """
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    if gt.max() == 0:
        return 1.0 if pred.max() == 0 else 0.0

    # Pixel-wise weight: distance transform on the GT mask boundary emphasises edges
    from scipy.ndimage import distance_transform_edt
    dst = distance_transform_edt(gt)
    dst_bg = distance_transform_edt(1 - gt)
    weight = 1 - (dst + dst_bg) / (dst + dst_bg).max()

    tp_w = (weight * gt * pred).sum()
    fp_w = (weight * (1 - gt) * pred).sum()
    fn_w = (weight * gt * (1 - pred)).sum()

    precision = (tp_w + eps) / (tp_w + fp_w + eps)
    recall = (tp_w + eps) / (tp_w + fn_w + eps)
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + eps)


def s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    """
    Structure measure (Fan et al. ICCV 2017).
    S = α·So + (1-α)·Sr
    """
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    y = gt.mean()
    if y == 0:
        return 1.0 - pred.mean()
    if y == 1:
        return pred.mean()
    return alpha * _s_object(pred, gt) + (1 - alpha) * _s_region(pred, gt)


def e_measure(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """
    Enhanced-alignment measure (Fan et al. IJCAI 2018).
    """
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    if gt.max() == 0:
        return (1 - pred).mean()
    if gt.min() == 1:
        return pred.mean()

    mu_p = pred.mean()
    mu_g = gt.mean()
    align_p = 2 * pred - mu_p
    align_g = 2 * gt - mu_g
    align_mat = 4 * (align_p * align_g) / (align_p ** 2 + align_g ** 2 + eps)
    return align_mat.mean()


# ---------------------------------------------------------------------------
# S-measure internals
# ---------------------------------------------------------------------------

def _ssim(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    n = x.size
    mu_x, mu_y = x.mean(), y.mean()
    sig_x = x.std()
    sig_y = y.std()
    sig_xy = ((x - mu_x) * (y - mu_y)).mean()
    return (2 * mu_x * mu_y + eps) * (2 * sig_xy + eps) / (
        (mu_x ** 2 + mu_y ** 2 + eps) * (sig_x ** 2 + sig_y ** 2 + eps)
    )


def _s_object(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    fg_pred = pred[gt == 1]
    bg_pred = pred[gt == 0]
    mu_fg = fg_pred.mean() if fg_pred.size > 0 else 0.0
    mu_bg = bg_pred.mean() if bg_pred.size > 0 else 0.0
    sigma_fg = fg_pred.std() if fg_pred.size > 0 else 0.0
    sigma_bg = bg_pred.std() if bg_pred.size > 0 else 0.0

    s_fg = (2 * mu_fg + eps) / (mu_fg ** 2 + 1.0 + sigma_fg + eps)
    s_bg = (2 * (1 - mu_bg) + eps) / ((1 - mu_bg) ** 2 + 1.0 + sigma_bg + eps)
    o = gt.mean() * s_fg + (1 - gt.mean()) * s_bg
    return o


def _s_region(pred: np.ndarray, gt: np.ndarray) -> float:
    H, W = gt.shape
    # centroid of ground-truth foreground
    total = gt.sum()
    if total == 0:
        cx, cy = W // 2, H // 2
    else:
        ys, xs = np.where(gt == 1)
        cx = int(xs.mean())
        cy = int(ys.mean())

    # four quadrants around centroid
    quads = [
        (gt[:cy, :cx], pred[:cy, :cx]),
        (gt[:cy, cx:], pred[:cy, cx:]),
        (gt[cy:, :cx], pred[cy:, :cx]),
        (gt[cy:, cx:], pred[cy:, cx:]),
    ]
    n = gt.size
    score = 0.0
    for gt_q, pred_q in quads:
        if gt_q.size == 0:
            continue
        w = gt_q.size / n
        score += w * _ssim(pred_q, gt_q)
    return score


# ---------------------------------------------------------------------------
# Accumulator for dataset-level mean metrics
# ---------------------------------------------------------------------------

class MetricTracker:
    """Accumulate per-image metrics and report dataset-level means."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._scores: dict[str, list[float]] = {
            "dice": [], "iou": [], "mae": [], "wfm": [], "sm": [], "em": []
        }

    def update(self, pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5):
        """
        pred: H×W float in [0,1] (raw sigmoid output)
        gt:   H×W float in {0,1}
        """
        pred_bin = (pred >= threshold).astype(np.float32)
        self._scores["dice"].append(dice_score(pred_bin, gt))
        self._scores["iou"].append(iou_score(pred_bin, gt))
        self._scores["mae"].append(mae_score(pred, gt))
        self._scores["wfm"].append(weighted_f_measure(pred_bin, gt))
        self._scores["sm"].append(s_measure(pred, gt))
        self._scores["em"].append(e_measure(pred, gt))

    def compute(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self._scores.items()}

    def summary(self) -> str:
        r = self.compute()
        return (
            f"mDice={r['dice']:.4f}  mIoU={r['iou']:.4f}  "
            f"MAE={r['mae']:.4f}  wFm={r['wfm']:.4f}  "
            f"Sm={r['sm']:.4f}  Em={r['em']:.4f}"
        )
