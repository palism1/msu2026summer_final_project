# FILE MAP | Box cascade: U-Net proposes a box from its cached probability map, MedSAM segments
#   inside it (docs/PLAN_ENSEMBLE.md Phase 4). Pure numpy + scipy at MODULE level — no torch, no
#   segment_anything — so tests/test_cascade.py runs GPU-free. `run_cascade_split` takes an
#   already-built zero-shot segmenter (``zs``, a src.models.zeroshot.ZeroShotSAM) and calls
#   ``zs.predict_prob_from_box``, so torch and segment_anything are imported lazily, only inside
#   cascade_eval.py's real-run path, mirroring src/models/zeroshot.py's own lazy-import shape.
#   [DO NOT TOUCH] the "zero" empty_policy default — a cascade must never invent a detection.
"""Torch-free box cascade: largest-component box derivation and the per-split runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

from src.models.zeroshot import box_from_mask

_STRUCTURE_8CONN = np.ones((3, 3), dtype=int)   # 8-connectivity for scipy.ndimage.label


def largest_component(binary: np.ndarray) -> Optional[np.ndarray]:
    """Largest 8-connected component of a binary mask, as a bool mask.

    Returns None for an all-zero input. Ties go to the lowest label index — np.argmax already
    returns the first (smallest-index) maximum, so this is deterministic across runs.
    """
    binary = np.asarray(binary).astype(bool)
    if not binary.any():
        return None
    labeled, n = ndimage.label(binary, structure=_STRUCTURE_8CONN)
    if n == 0:
        return None
    sizes = ndimage.sum(binary, labeled, index=range(1, n + 1))
    best_label = int(np.argmax(sizes)) + 1   # argmax ties -> first (lowest) label index
    return labeled == best_label


def box_from_prediction(prob: np.ndarray, threshold: float = 0.5, padding: int = 5
                        ) -> Optional[np.ndarray]:
    """Threshold a probability map, take its largest connected component, and derive a padded
    XYXY box from it via zeroshot.box_from_mask. Returns None when nothing survives thresholding
    or the largest component is empty."""
    binary = prob >= threshold
    component = largest_component(binary)
    if component is None:
        return None
    return box_from_mask(component.astype(np.float32), padding=padding)


def load_image_uint8(path, img_size: int) -> np.ndarray:
    """Read an image, resize to (img_size, img_size), return HxWx3 uint8 RGB.

    Uses PIL to open (matching the rest of the repo's image I/O) and cv2.resize with
    INTER_LINEAR, matching A.Resize's default interpolation exactly.
    """
    import cv2
    from PIL import Image

    img = np.array(Image.open(path).convert("RGB"))
    resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.uint8)


@dataclass
class CascadeSplitResult:
    probs: np.ndarray     # (N,H,W) float32
    n_empty: int
    n_multiblob: int
    gpu_seconds: float


def run_cascade_split(zs, image_paths: list, detector_probs: Optional[np.ndarray], gt: np.ndarray,
                      img_size: int, box_padding: int, detector_threshold: float,
                      empty_policy: str, box_source: str,
                      on_image: Optional[Callable] = None) -> CascadeSplitResult:
    """Run the cascade over one split.

    ``zs`` is a built src.models.zeroshot.ZeroShotSAM (segmenter). ``box_source`` selects the
    box's origin: "prediction" derives it from ``detector_probs`` (the Phase 1 float16-decoded
    detector cache) at ``detector_threshold``; "gt" derives it from ``gt`` at 0.5, for the
    ceiling row only, where ``detector_probs`` may be None. ``empty_policy`` "zero" writes an
    all-zero map when no box is found — the method invents no detection; "passthrough" copies
    ``detector_probs`` unchanged (undefined/zero when ``detector_probs`` is None).
    """
    if empty_policy not in ("zero", "passthrough"):
        raise ValueError(f"empty_policy must be 'zero' or 'passthrough', got {empty_policy!r}")
    if box_source not in ("prediction", "gt"):
        raise ValueError(f"box_source must be 'prediction' or 'gt', got {box_source!r}")

    n = gt.shape[0]
    out = np.zeros_like(gt, dtype=np.float32)
    n_empty = 0
    n_multiblob = 0
    gpu_seconds = 0.0

    for i in range(n):
        if box_source == "gt":
            binary = gt[i] >= 0.5
        else:
            binary = detector_probs[i] >= detector_threshold

        if binary.any():
            labeled, n_blobs = ndimage.label(binary, structure=_STRUCTURE_8CONN)
            if n_blobs > 1:
                n_multiblob += 1
            sizes = ndimage.sum(binary, labeled, index=range(1, n_blobs + 1))
            best_label = int(np.argmax(sizes)) + 1
            component = labeled == best_label
        else:
            component = None

        if component is None:
            n_empty += 1
            out[i] = detector_probs[i] if (empty_policy == "passthrough"
                                          and detector_probs is not None) else 0.0
            continue

        box = box_from_mask(component.astype(np.float32), padding=box_padding)
        image = load_image_uint8(image_paths[i], img_size)
        import time
        t0 = time.perf_counter()
        prob = zs.predict_prob_from_box(image, box, (gt.shape[1], gt.shape[2]))
        gpu_seconds += time.perf_counter() - t0
        out[i] = prob

        if on_image is not None:
            on_image(i)

    return CascadeSplitResult(probs=out, n_empty=n_empty, n_multiblob=n_multiblob,
                              gpu_seconds=gpu_seconds)
