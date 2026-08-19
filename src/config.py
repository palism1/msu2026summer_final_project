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

# Pipeline stages, named once here so --dry-run and the README's repo map stay in sync.
# [DO NOT TOUCH]
PIPELINE_STAGES = ("data pipeline", "model build", "train", "evaluate", "benchmark")

NORMALIZATION_CHOICES = ("imagenet", "minmax")   # kept in sync with src/normalization.py

# Photometric augmentation policies. STANDARD is what every published run used. [DO NOT TOUCH]
STANDARD_COLOR_JITTER = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2,
                         "hue": 0.1, "p": 0.5}
# Brightness/contrast disabled. Exists ONLY for medsam_ctrl: min_max_normalize is exactly
# invariant to A.ColorJitter's brightness (a*x) and contrast (f*x + mean*(1-f)), so a min-max
# model never sees those two augmentations. medsam_ctrl is the ImageNet-normalized arm with
# them removed, making the medsam_ctrl -> medsam_minmax comparison a normalization ablation
# rather than a normalization+augmentation one. See docs/MEDSAM_INVESTIGATION.md.
NO_BC_COLOR_JITTER = {**STANDARD_COLOR_JITTER, "brightness": 0.0, "contrast": 0.0}


@dataclass(frozen=True)
class ModelSpec:
    cfg_block: str          # configs/base.yaml block holding DEPLOYMENT settings (weights, LoRA)
    normalization: str      # PROTOCOL — bound to the model key here, never read from YAML
    color_jitter: dict      # PROTOCOL — ditto
    default_backbone: str


# [DO NOT TOUCH] The single source of model-keyed dispatch. Adding a model = adding one row.
# Protocol (normalization, color_jitter) lives here and NOT in YAML on purpose: checkpoint and
# results paths derive from the model key alone (checkpoints/<name>/seed<N>/best.pt), so a
# config-settable protocol would let two different input pipelines write to one results path
# and silently invalidate published numbers.
# The three MedSAM arms deliberately share ONE cfg_block, so their weights/LoRA settings
# cannot drift apart — they differ only in the protocol columns.
MODEL_SPECS: dict[str, ModelSpec] = {
    "unet":          ModelSpec("model",  "imagenet", STANDARD_COLOR_JITTER, "resnet34"),
    "sam_lora":      ModelSpec("sam",    "imagenet", STANDARD_COLOR_JITTER, "vit_h"),
    "medsam":        ModelSpec("medsam", "imagenet", STANDARD_COLOR_JITTER, "vit_b"),
    "sam_b":         ModelSpec("sam_b",  "imagenet", STANDARD_COLOR_JITTER, "vit_b"),
    # --- corrected / control MedSAM arms (docs/MEDSAM_INVESTIGATION.md, experiment C) ---
    "medsam_minmax": ModelSpec("medsam", "minmax",   STANDARD_COLOR_JITTER, "vit_b"),
    "medsam_ctrl":   ModelSpec("medsam", "imagenet", NO_BC_COLOR_JITTER,    "vit_b"),
}
MODEL_CHOICES = tuple(MODEL_SPECS)

_PROTOCOL_KEYS = ("normalization", "color_jitter")

_DEFAULT_DRIVE_RESULTS = "/content/drive/MyDrive/msu2026_checkpoints/results"
_DEFAULT_DRIVE_CHECKPOINTS = "/content/drive/MyDrive/msu2026_checkpoints"
_DEFAULT_OVERLAY_SPLITS = ("seen_kvasir", "cvc_colondb")
_DEFAULT_CACHE_ROOT = "cache"


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
    normalization: str = "imagenet"
    color_jitter: dict = field(default_factory=lambda: dict(STANDARD_COLOR_JITTER))

    @property
    def checkpoint_path(self) -> str:
        return str(Path(self.checkpoint_dir) / "best.pt")


def _spec(model: str) -> ModelSpec:
    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{model}'. Choices: {', '.join(MODEL_CHOICES)}")
    return MODEL_SPECS[model]


def resolve_normalization(model: str) -> str:
    """Input normalization for `model`. A pure function of the model key by design."""
    return _spec(model).normalization


def resolve_color_jitter(model: str) -> dict:
    return dict(_spec(model).color_jitter)   # copy: callers must not mutate the registry


def resolve_cache_root(cfg: dict) -> str:
    """Per-image probability cache root for predict_cache.py / ensemble_eval.py / cascade_eval.py.

    Reads ``cfg["ensemble"]["cache_dir"]``; defaults to ``"cache"`` (gitignored, see
    docs/PLAN_ENSEMBLE.md Phase 1)."""
    return str(cfg.get("ensemble", {}).get("cache_dir", _DEFAULT_CACHE_ROOT))


def reject_protocol_overrides(cfg: dict) -> None:
    """Raise if any model config block tries to set the training protocol.

    Protocol is bound to the model key (MODEL_SPECS). Honouring a YAML override would let
    `normalization: minmax` under `medsam:` retrain into checkpoints/medsam/seed42/best.pt and
    results/medsam/seed42/ under different preprocessing, overwriting published numbers with
    silently incomparable ones. Rejecting is the runtime invariant that a unit test on the
    current YAML cannot provide.
    """
    for block_name in sorted({s.cfg_block for s in MODEL_SPECS.values()}):
        block = cfg.get(block_name, {}) or {}
        found = [k for k in _PROTOCOL_KEYS if k in block]
        if found:
            raise ValueError(
                f"configs set {found} under the '{block_name}:' block, but training protocol "
                f"(input normalization, photometric augmentation) is bound to the model key in "
                f"src/config.MODEL_SPECS and is deliberately not configurable — results and "
                f"checkpoint paths derive from the model key alone. Remove those keys. To train "
                f"MedSAM under [0,1] min-max use --model medsam_minmax; for the augmentation "
                f"control use --model medsam_ctrl."
            )


def _resolve_backbone(model: str, cfg: dict) -> str:
    spec = _spec(model)
    if model == "unet":                        # U-Net names its backbone `encoder`, not `model_type`
        return cfg.get("model", {}).get("encoder", spec.default_backbone)
    return cfg.get(spec.cfg_block, {}).get("model_type", spec.default_backbone)


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
    reject_protocol_overrides(cfg)

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
        normalization=resolve_normalization(model),
        color_jitter=resolve_color_jitter(model),
    )


def describe_plan(plan: RunPlan) -> str:
    """Render a RunPlan as a human-readable dry-run summary."""
    budget = (
        f"{plan.max_train_minutes:g} min wall-clock cap"
        if plan.max_train_minutes is not None
        else "none (epoch/patience only)"
    )
    stages = " -> ".join(PIPELINE_STAGES)
    cj = plan.color_jitter
    return "\n".join([
        "Run plan (dry run — nothing executed)",
        "=" * 52,
        f"Pipeline stages : {stages}",
        f"Model           : {plan.model}  (backbone: {plan.backbone})",
        f"Seed            : {plan.seed}",
        f"Epochs          : {plan.epochs}   batch: {plan.batch_size}   patience: {plan.patience}",
        f"LR / wd         : {plan.lr:g} / {plan.weight_decay:g}",
        f"Image size      : {plan.img_size}",
        f"Normalization   : {plan.normalization}",
        f"Color jitter    : brightness={cj['brightness']} contrast={cj['contrast']} "
        f"saturation={cj['saturation']} hue={cj['hue']} p={cj['p']}",
        f"Time budget     : {budget}",
        f"Data root       : {plan.data_root}",
        f"Checkpoint      : {plan.checkpoint_path}",
        f"Local results   : {plan.local_results_dir}",
        f"Drive results   : {plan.drive_results_dir}",
        f"Drive checkpoint: {plan.drive_checkpoint_dir}",
        f"Overlays        : {plan.n_overlay_samples} samples from {', '.join(plan.overlay_splits)}",
        "=" * 52,
    ])


# ---------------------------------------------------------------------------
# Zero-shot (oracle-box) baseline registry — zeroshot_eval.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZeroShotSpec:
    model_type: str
    normalization: str
    published: bool = False   # produced by 05_benchmark.ipynb; guarded in zeroshot_eval.py


# [DO NOT TOUCH] Oracle-tier protocol, in code for the same reason as MODEL_SPECS.
# YAML supplies only the checkpoint filename (deployment) under zeroshot.checkpoints.
ZEROSHOT_SPECS: dict[str, ZeroShotSpec] = {
    "vanilla_sam":           ZeroShotSpec("vit_h", "imagenet", published=True),
    "vanilla_medsam":        ZeroShotSpec("vit_b", "imagenet", published=True),
    "vanilla_sam_b":         ZeroShotSpec("vit_b", "imagenet"),   # experiment B
    "vanilla_medsam_minmax": ZeroShotSpec("vit_b", "minmax"),     # experiment A
}
ZEROSHOT_CHOICES = tuple(ZEROSHOT_SPECS)


@dataclass
class ZeroShotPlan:
    """Fully resolved, ready-to-execute description of one zero-shot oracle-box baseline run."""

    key: str
    model_type: str
    checkpoint: str
    normalization: str
    prompt: str
    box_padding: int
    img_size: int
    data_root: str
    local_results_root: str
    drive_results_root: str
    seed: int = 0
    # build_splits' seed only shuffles the train list, which zero-shot never touches — hence
    # the fixed seed0 results folder regardless of splits_seed.
    splits_seed: int = 42
    published: bool = False

    @property
    def local_results_dir(self) -> str:
        return str(Path(self.local_results_root) / self.key / f"seed{self.seed}")

    @property
    def drive_results_dir(self) -> str:
        return str(Path(self.drive_results_root) / self.key / f"seed{self.seed}")


def build_zeroshot_plan(cfg: dict, baseline: str, overrides: Optional[dict] = None) -> ZeroShotPlan:
    """Resolve a merged config dict + baseline name into a concrete ZeroShotPlan."""
    if baseline not in ZEROSHOT_SPECS:
        raise ValueError(f"Unknown baseline '{baseline}'. Choices: {', '.join(ZEROSHOT_CHOICES)}")
    spec = ZEROSHOT_SPECS[baseline]
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    zs_cfg = cfg.get("zeroshot", {}) or {}
    checkpoints = zs_cfg.get("checkpoints", {}) or {}
    try:
        checkpoint = checkpoints[baseline]
    except KeyError:
        raise ValueError(
            f"No checkpoint filename for baseline '{baseline}' under configs/base.yaml -> "
            f"zeroshot.checkpoints. Add one there."
        )

    prompt = zs_cfg.get("prompt", "box")
    if prompt not in ("box", "point"):
        raise ValueError(f"zeroshot.prompt must be 'box' or 'point', got {prompt!r}")
    box_padding = int(zs_cfg.get("box_padding", 5))

    data = cfg.get("data", {}) or {}
    img_size = int(data.get("img_size", 352))
    data_root = str(data.get("root", "data/polyp"))

    output = cfg.get("output", {}) or {}
    local_results_root = overrides.get("results_dir") or output.get("local_results_dir", "results")
    drive_results_root = output.get("drive_results_dir", _DEFAULT_DRIVE_RESULTS)

    return ZeroShotPlan(
        key=baseline,
        model_type=spec.model_type,
        checkpoint=checkpoint,
        normalization=spec.normalization,
        prompt=prompt,
        box_padding=box_padding,
        img_size=img_size,
        data_root=data_root,
        local_results_root=str(local_results_root),
        drive_results_root=str(drive_results_root),
        published=spec.published,
    )


def describe_zeroshot_plan(plan: ZeroShotPlan) -> str:
    """Render a ZeroShotPlan as a human-readable dry-run summary."""
    lines = [
        "Zero-shot plan (dry run — nothing executed)",
        "=" * 52,
        f"Baseline        : {plan.key}  (tier: oracle)",
        f"Backbone        : {plan.model_type}",
        f"Checkpoint      : {plan.checkpoint}",
        f"Normalization   : {plan.normalization}",
        f"Prompt          : {plan.prompt}  (padding: {plan.box_padding})",
        f"Image size      : {plan.img_size}",
        f"Data root       : {plan.data_root}",
        f"Local results   : {plan.local_results_dir}",
        f"Drive results   : {plan.drive_results_dir}",
    ]
    if plan.published:
        lines.append("PUBLISHED — re-running requires --allow-published")
    lines.append("=" * 52)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ensemble registry — ensemble_eval.py (docs/PLAN_ENSEMBLE.md Phase 2/3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnsembleSpec:
    members: tuple[str, ...]   # model keys from MODEL_SPECS
    weighting: str             # "uniform" | "fitted"
    display: str


# [DO NOT TOUCH] Pre-registered member sets (docs/DECISIONS.md). Member set A is the largest
# architectural distance in the study (U-Net vs SAM-ViT-H); member set B is the top three rows
# of the unseen leaderboard. Each set gets a uniform arm and a fitted arm.
ENSEMBLE_SPECS: dict[str, EnsembleSpec] = {
    "ens_unet_samh":        EnsembleSpec(("unet", "sam_lora"), "uniform",
                                         "Ensemble: U-Net + SAM-ViT-H (uniform)"),
    "ens_top3":             EnsembleSpec(("unet", "sam_lora", "sam_b"), "uniform",
                                         "Ensemble: top-3 (uniform)"),
    "ens_unet_samh_fitted": EnsembleSpec(("unet", "sam_lora"), "fitted",
                                         "Ensemble: U-Net + SAM-ViT-H (fitted)"),
    "ens_top3_fitted":      EnsembleSpec(("unet", "sam_lora", "sam_b"), "fitted",
                                         "Ensemble: top-3 (fitted)"),
}
ENSEMBLE_CHOICES = tuple(ENSEMBLE_SPECS)


@dataclass
class EnsemblePlan:
    """Fully resolved, ready-to-execute description of one ensemble evaluation run."""

    key: str
    members: tuple[str, ...]
    member_dirs: tuple[str, ...]
    member_seeds: tuple[int, ...]
    weighting: str
    seed: int
    threshold: float
    grid_step: float
    img_size: int
    data_root: str
    cache_root: str
    local_results_dir: str
    drive_results_dir: str
    display: str


def build_ensemble_plan(cfg: dict, key: str, seed: int, overrides: Optional[dict] = None) -> EnsemblePlan:
    """Resolve a merged config dict + ensemble spec key + seed into a concrete EnsemblePlan.

    Ensembles are seed-matched (docs/PLAN_ENSEMBLE.md interpretation 4): every member is read at
    the same seed as the ensemble itself. ``member_dirs`` resolves through the same
    ``_checkpoint_name(model, _resolve_backbone(model, cfg))`` pair build_run_plan uses, so
    ``sam_lora`` maps to the ``sam_vit_h`` results/cache directory.
    """
    if key not in ENSEMBLE_SPECS:
        raise ValueError(f"Unknown ensemble spec '{key}'. Choices: {', '.join(ENSEMBLE_CHOICES)}")
    spec = ENSEMBLE_SPECS[key]
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    member_dirs = tuple(_checkpoint_name(m, _resolve_backbone(m, cfg)) for m in spec.members)

    data = cfg.get("data", {}) or {}
    img_size = int(data.get("img_size", 352))
    data_root = str(data.get("root", "data/polyp"))

    ens_cfg = cfg.get("ensemble", {}) or {}
    cache_root = overrides.get("cache_dir") or resolve_cache_root(cfg)
    grid_step = float(ens_cfg.get("weight_grid_step", 0.05))

    output = cfg.get("output", {}) or {}
    local_base = output.get("local_results_dir", "results")
    drive_base = output.get("drive_results_dir", _DEFAULT_DRIVE_RESULTS)
    seed_leaf = f"{key}/seed{seed}"

    return EnsemblePlan(
        key=key,
        members=spec.members,
        member_dirs=member_dirs,
        member_seeds=tuple(seed for _ in spec.members),
        weighting=spec.weighting,
        seed=int(seed),
        threshold=0.5,
        grid_step=grid_step,
        img_size=img_size,
        data_root=data_root,
        cache_root=str(cache_root),
        local_results_dir=str(Path(local_base) / seed_leaf),
        drive_results_dir=str(Path(drive_base) / seed_leaf),
        display=spec.display,
    )


def describe_ensemble_plan(plan: EnsemblePlan) -> str:
    """Render an EnsemblePlan as a human-readable dry-run summary."""
    return "\n".join([
        "Ensemble plan (dry run — nothing executed)",
        "=" * 52,
        f"Spec            : {plan.key}  ({plan.display})",
        f"Members         : {', '.join(plan.members)}",
        f"Member dirs     : {', '.join(plan.member_dirs)}",
        f"Seed            : {plan.seed}  (seed-matched across members)",
        f"Weighting       : {plan.weighting}",
        f"Threshold       : {plan.threshold}",
        f"Grid step       : {plan.grid_step}",
        f"Image size      : {plan.img_size}",
        f"Data root       : {plan.data_root}",
        f"Cache root      : {plan.cache_root}",
        f"Local results   : {plan.local_results_dir}",
        f"Drive results   : {plan.drive_results_dir}",
        "=" * 52,
    ])


# ---------------------------------------------------------------------------
# Cascade registry — cascade_eval.py (docs/PLAN_ENSEMBLE.md Phase 4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CascadeSpec:
    detector: Optional[str]    # model key, or None for the GT-box ceiling row
    segmenter: str             # a ZEROSHOT_SPECS key
    box_source: str            # "prediction" | "gt"
    empty_policy: str          # "zero" | "passthrough"
    tier: str                  # "derived" | "oracle"
    display: str


def _check_cascade_invariants(key: str, spec: "CascadeSpec") -> None:
    """The fair-versus-ceiling boundary as a code invariant (docs/PLAN_ENSEMBLE.md Phase 4),
    in the style reject_protocol_overrides already uses for training protocol."""
    if spec.box_source == "gt" and spec.tier != "oracle":
        raise ValueError(
            f"cascade spec '{key}': box_source='gt' requires tier='oracle', got tier={spec.tier!r}. "
            f"A GT-derived box is an oracle prompt; it cannot be labeled 'derived' (no ground "
            f"truth at inference)."
        )
    if spec.box_source == "prediction":
        if spec.detector is None:
            raise ValueError(
                f"cascade spec '{key}': box_source='prediction' requires a detector model key."
            )
        if spec.tier != "derived":
            raise ValueError(
                f"cascade spec '{key}': box_source='prediction' requires tier='derived', got "
                f"tier={spec.tier!r}."
            )


# [DO NOT TOUCH] The segmenter weights/normalization come from ZEROSHOT_SPECS["vanilla_medsam_minmax"]
# and cfg["zeroshot"]["checkpoints"] — no second source of MedSAM configuration.
CASCADE_SPECS: dict[str, CascadeSpec] = {
    "casc_unet_medsam":         CascadeSpec("unet", "vanilla_medsam_minmax", "prediction", "zero",
                                            "derived", "Cascade: U-Net box -> MedSAM min-max"),
    "oracle_casc_gtbox_medsam": CascadeSpec(None, "vanilla_medsam_minmax", "gt", "zero",
                                            "oracle",
                                            "Cascade: GT box -> MedSAM min-max (path-matched ceiling)"),
}
for _cascade_key, _cascade_spec in CASCADE_SPECS.items():
    _check_cascade_invariants(_cascade_key, _cascade_spec)
del _cascade_key, _cascade_spec

CASCADE_CHOICES = tuple(CASCADE_SPECS)


@dataclass
class CascadePlan:
    """Fully resolved, ready-to-execute description of one cascade evaluation run."""

    key: str
    detector: Optional[str]
    detector_dir: Optional[str]
    segmenter: str
    segmenter_checkpoint: str
    segmenter_model_type: str
    segmenter_normalization: str
    box_source: str
    empty_policy: str
    tier: str
    seed: int
    box_padding: int
    detector_threshold: float
    img_size: int
    data_root: str
    cache_root: str
    local_results_dir: str
    drive_results_dir: str
    display: str


def build_cascade_plan(cfg: dict, key: str, seed: Optional[int] = None,
                       overrides: Optional[dict] = None) -> CascadePlan:
    """Resolve a merged config dict + cascade spec key into a concrete CascadePlan.

    The ceiling row (``tier == "oracle"``) always uses seed 0, matching every other oracle row
    (see ZeroShotPlan). A fair, detector-driven row uses the caller's seed (default 42).
    """
    if key not in CASCADE_SPECS:
        raise ValueError(f"Unknown cascade spec '{key}'. Choices: {', '.join(CASCADE_CHOICES)}")
    spec = CASCADE_SPECS[key]
    _check_cascade_invariants(key, spec)
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    resolved_seed = 0 if spec.tier == "oracle" else int(seed if seed is not None else overrides.get("seed", 42))

    detector_dir = None
    if spec.detector is not None:
        detector_dir = _checkpoint_name(spec.detector, _resolve_backbone(spec.detector, cfg))

    if spec.segmenter not in ZEROSHOT_SPECS:
        raise ValueError(f"Unknown segmenter '{spec.segmenter}' for cascade spec '{key}'.")
    zs_spec = ZEROSHOT_SPECS[spec.segmenter]
    zs_cfg = cfg.get("zeroshot", {}) or {}
    checkpoints = zs_cfg.get("checkpoints", {}) or {}
    try:
        segmenter_checkpoint = checkpoints[spec.segmenter]
    except KeyError:
        raise ValueError(
            f"No checkpoint filename for segmenter '{spec.segmenter}' under configs/base.yaml -> "
            f"zeroshot.checkpoints. Add one there."
        )

    cascade_cfg = cfg.get("cascade", {}) or {}
    box_padding = int(cascade_cfg.get("box_padding", zs_cfg.get("box_padding", 5)))
    detector_threshold = float(cascade_cfg.get("detector_threshold", 0.5))

    data = cfg.get("data", {}) or {}
    img_size = int(data.get("img_size", 352))
    data_root = str(data.get("root", "data/polyp"))

    cache_root = overrides.get("cache_dir") or resolve_cache_root(cfg)

    output = cfg.get("output", {}) or {}
    local_base = output.get("local_results_dir", "results")
    drive_base = output.get("drive_results_dir", _DEFAULT_DRIVE_RESULTS)
    seed_leaf = f"{key}/seed{resolved_seed}"

    return CascadePlan(
        key=key,
        detector=spec.detector,
        detector_dir=detector_dir,
        segmenter=spec.segmenter,
        segmenter_checkpoint=segmenter_checkpoint,
        segmenter_model_type=zs_spec.model_type,
        segmenter_normalization=zs_spec.normalization,
        box_source=spec.box_source,
        empty_policy=spec.empty_policy,
        tier=spec.tier,
        seed=resolved_seed,
        box_padding=box_padding,
        detector_threshold=detector_threshold,
        img_size=img_size,
        data_root=data_root,
        cache_root=str(cache_root),
        local_results_dir=str(Path(local_base) / seed_leaf),
        drive_results_dir=str(Path(drive_base) / seed_leaf),
        display=spec.display,
    )


def describe_cascade_plan(plan: CascadePlan) -> str:
    """Render a CascadePlan as a human-readable dry-run summary."""
    lines = [
        "Cascade plan (dry run — nothing executed)",
        "=" * 52,
        f"Spec            : {plan.key}  ({plan.display})",
        f"Tier            : {plan.tier}",
        f"Detector        : {plan.detector or '(none — GT box)'}",
        f"Segmenter       : {plan.segmenter}  (backbone: {plan.segmenter_model_type}, "
        f"normalization: {plan.segmenter_normalization})",
        f"Box source      : {plan.box_source}",
        f"Empty policy    : {plan.empty_policy}",
        f"Box padding     : {plan.box_padding}",
        f"Detector thresh : {plan.detector_threshold}",
        f"Seed            : {plan.seed}",
        f"Image size      : {plan.img_size}",
        f"Data root       : {plan.data_root}",
        f"Cache root      : {plan.cache_root}",
        f"Local results   : {plan.local_results_dir}",
        f"Drive results   : {plan.drive_results_dir}",
        "=" * 52,
    ]
    return "\n".join(lines)
