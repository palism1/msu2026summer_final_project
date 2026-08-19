# FILE MAP | Per-image probability cache for the ensemble/cascade experiment (docs/PLAN_ENSEMBLE.md
#   Phase 1). numpy + stdlib ONLY — no torch, no albumentations — so predict_cache.py --dry-run and
#   tests/test_ensemble_cache.py run on a laptop with no GPU stack installed. Each (model, seed)
#   writes one <split>.npz per test split plus one manifest.json; predict_cache.py is the only
#   writer, ensemble_eval.py and cascade_eval.py only read through read_split_cache.
#   [DO NOT TOUCH] CACHE_VERSION bump discipline: bump it whenever the .npz field set changes, so
#   a stale cache fails loudly instead of silently feeding a different-shaped array downstream.
"""Torch-free per-image probability cache: pack/unpack, read/write, and alignment checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def cache_dir(cache_root: str | Path, model_dir: str, seed: int) -> Path:
    """<cache_root>/<model_dir>/seed<seed>"""
    return Path(cache_root) / model_dir / f"seed{seed}"


def cache_path(cache_root: str | Path, model_dir: str, seed: int, split: str) -> Path:
    """<cache_root>/<model_dir>/seed<seed>/<split>.npz"""
    return cache_dir(cache_root, model_dir, seed) / f"{split}.npz"


# ---------------------------------------------------------------------------
# Ground-truth packing (bit-packed, so a (N,H,W) bool array costs 1/8th on disk)
# ---------------------------------------------------------------------------

def pack_gt(gt: np.ndarray) -> np.ndarray:
    """(N,H,W) bool/float{0,1} -> packed uint8 1-D via np.packbits."""
    return np.packbits(gt.astype(bool))


def unpack_gt(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """packed uint8 1-D + original shape -> (N,H,W) float32 in {0,1}.

    np.packbits pads to a byte boundary, so the unpacked bit array must be sliced to
    prod(shape) BEFORE the reshape — a shape like (3, 5, 7) (105 bits, not a multiple of 8)
    would otherwise reshape a too-long array and raise.
    """
    n = int(np.prod(shape))
    bits = np.unpackbits(packed)[:n]
    return bits.reshape(shape).astype(np.float32)


def gt_fingerprint(packed: np.ndarray) -> str:
    """sha256 hex digest of the packed ground-truth bytes."""
    return hashlib.sha256(np.ascontiguousarray(packed).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Split cache
# ---------------------------------------------------------------------------

@dataclass
class SplitCache:
    probs: np.ndarray        # (N,H,W) float32, decoded from float16
    gt: np.ndarray           # (N,H,W) float32 in {0,1}
    image_paths: list[str]
    mask_paths: list[str]
    gt_sha256: str


def write_split_cache(path: str | Path, probs: np.ndarray, gt: np.ndarray,
                      image_paths: list[str], mask_paths: list[str]) -> Path:
    """Write one <split>.npz. Ground truth is bit-packed; probabilities are stored float16."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed_gt = pack_gt(gt)
    np.savez(
        path,
        probs_f16=probs.astype(np.float16),
        gt_packed=packed_gt,
        shape=np.array(gt.shape, dtype=np.int64),
        image_paths=np.array(image_paths, dtype=object),
        mask_paths=np.array(mask_paths, dtype=object),
        version=np.array(CACHE_VERSION, dtype=np.int64),
    )
    return path


def read_split_cache(path: str | Path) -> SplitCache:
    """Read one <split>.npz back into a SplitCache."""
    with np.load(path, allow_pickle=True) as npz:
        shape = tuple(int(x) for x in npz["shape"])
        gt = unpack_gt(npz["gt_packed"], shape)
        probs = npz["probs_f16"].astype(np.float32)
        image_paths = [str(p) for p in npz["image_paths"]]
        mask_paths = [str(p) for p in npz["mask_paths"]]
    return SplitCache(
        probs=probs, gt=gt, image_paths=image_paths, mask_paths=mask_paths,
        gt_sha256=gt_fingerprint(pack_gt(gt)),
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(dir_path: str | Path, payload: dict) -> Path:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_manifest(dir_path: str | Path) -> dict:
    path = Path(dir_path) / "manifest.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def assert_aligned(caches: list[SplitCache]) -> None:
    """Raise ValueError when any member's image_paths or gt_sha256 differ from the first.

    Converts the "every member evaluates the same grid, so probability maps are directly
    averagable" assumption into a runtime check, instead of a silent misalignment.
    """
    if not caches:
        return
    ref = caches[0]
    for c in caches[1:]:
        if c.image_paths != ref.image_paths:
            raise ValueError(
                f"image_paths mismatch: member has {len(c.image_paths)} paths, "
                f"reference has {len(ref.image_paths)} paths (or different ordering)."
            )
        if c.gt_sha256 != ref.gt_sha256:
            raise ValueError(
                f"gt_sha256 mismatch: {c.gt_sha256} != {ref.gt_sha256}. Members do not share "
                f"the same ground truth for this split."
            )


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------

_ENV_PACKAGES = ("numpy", "scipy", "albumentations", "segment_anything")


def environment_record() -> dict:
    """Best-effort version snapshot for the manifest / inference.json sidecar.

    Every field is a string. torch / torchvision / cuda / cudnn are read only when torch is
    already imported (`sys.modules`), so this stays importable without touching torch. A
    missing package falls back to "unknown" rather than raising.
    """
    import importlib.metadata as _md
    import sys

    def _version(pkg: str) -> str:
        try:
            return _md.version(pkg)
        except Exception:
            return "unknown"

    record = {"python": sys.version.split()[0]}
    for pkg in _ENV_PACKAGES:
        record[pkg] = _version(pkg)

    torch = sys.modules.get("torch")
    if torch is not None:
        record["torch"] = getattr(torch, "__version__", "unknown")
        try:
            record["torchvision"] = _version("torchvision")
        except Exception:
            record["torchvision"] = "unknown"
        try:
            record["cuda"] = torch.version.cuda or "unknown"
        except Exception:
            record["cuda"] = "unknown"
        try:
            record["cudnn"] = str(torch.backends.cudnn.version() or "unknown")
        except Exception:
            record["cudnn"] = "unknown"
    else:
        record["torch"] = "unknown"
        record["torchvision"] = "unknown"
        record["cuda"] = "unknown"
        record["cudnn"] = "unknown"

    return record
