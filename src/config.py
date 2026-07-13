# FILE MAP | Pure config layer for the training runner.
#   Loads/merges configs/run.yaml (+ its base_config), resolves a RunPlan, and renders a
#   human-readable dry-run summary. PURE by design: stdlib + PyYAML only, NO torch/numpy,
#   so `train.py --dry-run` and the config unit tests run on any machine without a GPU.
#   [DO NOT TOUCH] the checkpoint-path scheme in build_run_plan — evaluate.py and
#   notebooks/05_benchmark.ipynb load models by exactly that path.
"""Configuration loading, merging, and run-plan resolution (I/O-free logic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Pipeline stages, named once here so --dry-run and CLAUDE.md stay in sync. [DO NOT TOUCH]
PIPELINE_STAGES = ("data pipeline", "model build", "train", "evaluate", "benchmark")

MODEL_CHOICES = ("unet", "sam_lora", "medsam", "sam_b")

_DEFAULT_DRIVE_RESULTS = "/content/drive/MyDrive/msu2026_checkpoints/results"
_DEFAULT_DRIVE_CHECKPOINTS = "/content/drive/MyDrive/msu2026_checkpoints"
_DEFAULT_OVERLAY_SPLITS = ("seen_kvasir", "cvc_colondb")


# ---------------------------------------------------------------------------
# Loading & merging
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto a copy of ``base`` (override wins)."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_run_config(path: str | Path) -> dict:
    """
    Load a config file into a merged dict.

    A run config may point at a shared base via a top-level ``base_config:`` key; the base
    is loaded first and the run file merged on top. A plain base config (no ``base_config``,
    no ``run`` block) is returned as-is — this keeps the legacy
    ``train.py --config configs/base.yaml --model unet`` invocation working.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base_ref = cfg.pop("base_config", None)
    if base_ref is not None:
        base_path = _resolve_base_ref(base_ref, path.parent)
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def _resolve_base_ref(base_ref: str, sibling_dir: Path) -> Path:
    """Resolve base_config as absolute, sibling-relative, or cwd-relative (first that exists)."""
    ref = Path(base_ref)
    if ref.is_absolute():
        return ref
    for candidate in (sibling_dir / ref, ref):
        if candidate.exists():
            return candidate
    return sibling_dir / ref  # let open() raise a clear error on the sibling path


# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------

@dataclass
class RunPlan:
    """Fully resolved, ready-to-execute description of a single training run."""

    model: str
    backbone: str
    checkpoint_name: str
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    img_size: int
    num_workers: int
    data_root: str
    max_train_minutes: Optional[float]
    checkpoint_dir: str
    local_results_dir: str
    drive_results_dir: str
    drive_checkpoint_dir: str
    n_overlay_samples: int
    overlay_splits: list[str] = field(default_factory=list)

    @property
    def checkpoint_path(self) -> str:
        return str(Path(self.checkpoint_dir) / "best.pt")


def _resolve_backbone(model: str, cfg: dict) -> str:
    if model == "unet":
        return cfg.get("model", {}).get("encoder", "resnet34")
    if model == "sam_lora":
        return cfg.get("sam", {}).get("model_type", "vit_h")
    if model == "sam_b":
        return cfg.get("sam_b", {}).get("model_type", "vit_b")
    if model == "medsam":
        return cfg.get("medsam", {}).get("model_type", "vit_b")
    raise ValueError(f"Unknown model '{model}'. Choices: {', '.join(MODEL_CHOICES)}")


def _checkpoint_name(model: str, backbone: str) -> str:
    # Matches the folder scheme used by evaluate.py and notebooks 02-05:
    #   unet -> "unet",  sam_lora/sam_b -> "sam_<backbone>" (sam_vit_h / sam_vit_b),  medsam -> "medsam".
    if model in ("sam_lora", "sam_b"):
        return f"sam_{backbone}"
    return model


def build_run_plan(cfg: dict, overrides: Optional[dict] = None) -> RunPlan:
    """
    Resolve a merged config dict (+ optional CLI overrides) into a concrete RunPlan.

    ``overrides`` keys (any may be None/absent): model, seed, epochs, output_dir.
    Raises ValueError on an unknown model or non-positive epochs.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    run = cfg.get("run", {}) or {}
    training = cfg.get("training", {}) or {}
    data = cfg.get("data", {}) or {}
    output = cfg.get("output", {}) or {}

    model = overrides.get("model") or run.get("model") or cfg.get("model", {}).get("name") or "unet"
    if model not in MODEL_CHOICES:
        raise ValueError(f"Unknown model '{model}'. Choices: {', '.join(MODEL_CHOICES)}")

    backbone = _resolve_backbone(model, cfg)
    ckpt_name = _checkpoint_name(model, backbone)

    seed = int(overrides.get("seed", run.get("seed", training.get("seed", 42))))
    epochs = int(overrides.get("epochs", run.get("epochs", training.get("epochs", 100))))
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    batch_size = int(run.get("batch_size", training.get("batch_size", 16)))
    lr = float(run.get("lr", training.get("lr", 1e-4)))
    weight_decay = float(run.get("weight_decay", training.get("weight_decay", 1e-4)))
    patience = int(run.get("patience", training.get("early_stop_patience", 10)))
    max_minutes = run.get("max_train_minutes", None)
    max_minutes = float(max_minutes) if max_minutes is not None else None

    img_size = int(data.get("img_size", 352))
    num_workers = int(data.get("num_workers", 4))
    data_root = str(data.get("root", "data/polyp"))

    seed_leaf = f"{ckpt_name}/seed{seed}"
    ckpt_base = overrides.get("output_dir") or output.get("checkpoint_dir", "checkpoints")
    local_base = output.get("local_results_dir", "results")
    drive_base = output.get("drive_results_dir", _DEFAULT_DRIVE_RESULTS)
    drive_ckpt_base = output.get("drive_checkpoint_dir", _DEFAULT_DRIVE_CHECKPOINTS)

    overlay_splits = list(output.get("overlay_splits", _DEFAULT_OVERLAY_SPLITS))

    return RunPlan(
        model=model,
        backbone=backbone,
        checkpoint_name=ckpt_name,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        img_size=img_size,
        num_workers=num_workers,
        data_root=data_root,
        max_train_minutes=max_minutes,
        checkpoint_dir=str(Path(ckpt_base) / seed_leaf),
        local_results_dir=str(Path(local_base) / seed_leaf),
        drive_results_dir=str(Path(drive_base) / seed_leaf),
        drive_checkpoint_dir=str(Path(drive_ckpt_base) / seed_leaf),
        n_overlay_samples=int(output.get("n_overlay_samples", 8)),
        overlay_splits=overlay_splits,
    )


def describe_plan(plan: RunPlan) -> str:
    """Render a RunPlan as a human-readable dry-run summary."""
    budget = (
        f"{plan.max_train_minutes:g} min wall-clock cap"
        if plan.max_train_minutes is not None
        else "none (epoch/patience only)"
    )
    stages = " -> ".join(PIPELINE_STAGES)
    return "\n".join([
        "Run plan (dry run — nothing executed)",
        "=" * 52,
        f"Pipeline stages : {stages}",
        f"Model           : {plan.model}  (backbone: {plan.backbone})",
        f"Seed            : {plan.seed}",
        f"Epochs          : {plan.epochs}   batch: {plan.batch_size}   patience: {plan.patience}",
        f"LR / wd         : {plan.lr:g} / {plan.weight_decay:g}",
        f"Image size      : {plan.img_size}",
        f"Time budget     : {budget}",
        f"Data root       : {plan.data_root}",
        f"Checkpoint      : {plan.checkpoint_path}",
        f"Local results   : {plan.local_results_dir}",
        f"Drive results   : {plan.drive_results_dir}",
        f"Drive checkpoint: {plan.drive_checkpoint_dir}",
        f"Overlays        : {plan.n_overlay_samples} samples from {', '.join(plan.overlay_splits)}",
        "=" * 52,
    ])
