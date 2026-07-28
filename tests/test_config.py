# FILE MAP | Unit tests for the PURE config layer in src/config.py.
#   No torch/numpy needed. Covers run.yaml load+merge, plan/path resolution, the legacy
#   flat-config path, and validation. [DO NOT TOUCH] the checkpoint-path assertions — they
#   encode the contract evaluate.py + notebooks/05_benchmark.ipynb rely on.
"""GPU-free tests for config loading and run-plan resolution."""

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config import (
    MODEL_CHOICES,
    MODEL_SPECS,
    NO_BC_COLOR_JITTER,
    NORMALIZATION_CHOICES,
    PIPELINE_STAGES,
    STANDARD_COLOR_JITTER,
    ZEROSHOT_CHOICES,
    ZEROSHOT_SPECS,
    build_run_plan,
    build_zeroshot_plan,
    describe_plan,
    describe_zeroshot_plan,
    load_run_config,
    resolve_color_jitter,
)

REPO = Path(__file__).resolve().parents[1]


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


@pytest.fixture
def base_cfg(tmp_path):
    _write(tmp_path, "base.yaml", """
        data: {root: data/polyp, img_size: 352, num_workers: 4}
        training: {batch_size: 16, epochs: 100, lr: 1.0e-4, weight_decay: 1.0e-4, early_stop_patience: 10, seed: 42}
        model: {name: unet, encoder: resnet34, encoder_weights: imagenet}
        sam: {model_type: vit_h, checkpoint: sam_vit_h_4b8939.pth, lora_r: 4, lora_alpha: 8, lora_dropout: 0.1}
        sam_b: {model_type: vit_b, checkpoint: sam_vit_b_01ec64.pth, lora_r: 4, lora_alpha: 8, lora_dropout: 0.1}
        medsam: {model_type: vit_b, checkpoint: medsam_vit_b.pth, lora_r: 4, lora_alpha: 8, lora_dropout: 0.1}
        zeroshot:
          prompt: box
          box_padding: 5
          checkpoints:
            vanilla_sam: sam_vit_h_4b8939.pth
            vanilla_sam_b: sam_vit_b_01ec64.pth
            vanilla_medsam: medsam_vit_b.pth
            vanilla_medsam_minmax: medsam_vit_b.pth
    """)
    return tmp_path


def test_run_config_merges_base(base_cfg):
    _write(base_cfg, "run.yaml", """
        base_config: base.yaml
        run: {model: medsam, seed: 7, epochs: 5, batch_size: 4, lr: 5.0e-5, patience: 3}
        output: {checkpoint_dir: checkpoints, local_results_dir: results,
                 drive_results_dir: /drive/results, n_overlay_samples: 2,
                 overlay_splits: [seen_kvasir]}
    """)
    cfg = load_run_config(base_cfg / "run.yaml")
    plan = build_run_plan(cfg)

    assert plan.model == "medsam"
    assert plan.backbone == "vit_b"
    assert plan.seed == 7 and plan.epochs == 5 and plan.batch_size == 4
    assert plan.img_size == 352  # inherited from base
    assert plan.checkpoint_dir == "checkpoints/medsam/seed7"
    assert plan.checkpoint_path == "checkpoints/medsam/seed7/best.pt"
    assert plan.local_results_dir == "results/medsam/seed7"
    assert plan.drive_results_dir == "/drive/results/medsam/seed7"
    assert plan.overlay_splits == ["seen_kvasir"]
    # No drive_checkpoint_dir override in this config -> falls back to the default root.
    assert plan.drive_checkpoint_dir == "/content/drive/MyDrive/msu2026_checkpoints/medsam/seed7"


def test_drive_checkpoint_dir_override(base_cfg):
    _write(base_cfg, "run.yaml", """
        base_config: base.yaml
        run: {model: medsam, seed: 7}
        output: {drive_checkpoint_dir: /drive/my_checkpoints}
    """)
    cfg = load_run_config(base_cfg / "run.yaml")
    plan = build_run_plan(cfg)
    assert plan.drive_checkpoint_dir == "/drive/my_checkpoints/medsam/seed7"


def test_sam_checkpoint_name_encodes_backbone(base_cfg):
    _write(base_cfg, "run.yaml", "base_config: base.yaml\nrun: {model: sam_lora, seed: 42}\n")
    plan = build_run_plan(load_run_config(base_cfg / "run.yaml"))
    # notebooks/05_benchmark.ipynb looks for checkpoints/sam_vit_h/seed42
    assert plan.checkpoint_name == "sam_vit_h"
    assert plan.checkpoint_dir == "checkpoints/sam_vit_h/seed42"


def test_sam_b_isolates_backbone_from_medsam(base_cfg):
    # SAM ViT-B + LoRA: same ViT-B backbone as MedSAM but generic SAM weights, so it lands
    # at its own checkpoints/sam_vit_b path, distinct from medsam/ and sam_vit_h/.
    _write(base_cfg, "run.yaml", "base_config: base.yaml\nrun: {model: sam_b, seed: 42}\n")
    plan = build_run_plan(load_run_config(base_cfg / "run.yaml"))
    assert plan.backbone == "vit_b"
    assert plan.checkpoint_name == "sam_vit_b"
    assert plan.checkpoint_dir == "checkpoints/sam_vit_b/seed42"


def test_cli_overrides_win(base_cfg):
    _write(base_cfg, "run.yaml", "base_config: base.yaml\nrun: {model: medsam, seed: 1, epochs: 5}\n")
    cfg = load_run_config(base_cfg / "run.yaml")
    plan = build_run_plan(cfg, {"model": "unet", "seed": 99, "epochs": 3, "output_dir": "/ck"})
    assert plan.model == "unet" and plan.backbone == "resnet34"
    assert plan.seed == 99 and plan.epochs == 3
    assert plan.checkpoint_dir == "/ck/unet/seed99"


def test_legacy_flat_config_uses_cli_model(base_cfg):
    # A plain base config (no run:/base_config:) still resolves via --model, as before.
    cfg = load_run_config(base_cfg / "base.yaml")
    plan = build_run_plan(cfg, {"model": "unet"})
    assert plan.model == "unet"
    assert plan.epochs == 100 and plan.seed == 42  # from training block
    assert plan.checkpoint_dir == "checkpoints/unet/seed42"


def test_invalid_model_raises(base_cfg):
    cfg = load_run_config(base_cfg / "base.yaml")
    with pytest.raises(ValueError):
        build_run_plan(cfg, {"model": "nope"})


def test_non_positive_epochs_raises(base_cfg):
    cfg = load_run_config(base_cfg / "base.yaml")
    with pytest.raises(ValueError):
        build_run_plan(cfg, {"model": "unet", "epochs": 0})


def test_describe_plan_names_all_stages(base_cfg):
    plan = build_run_plan(load_run_config(base_cfg / "base.yaml"), {"model": "unet"})
    text = describe_plan(plan)
    for stage in PIPELINE_STAGES:
        assert stage in text


def test_model_choices_constant():
    assert set(MODEL_CHOICES) == {
        "unet", "sam_lora", "medsam", "sam_b", "medsam_minmax", "medsam_ctrl",
    }


# ---------------------------------------------------------------------------
# MODEL_SPECS registry — the single dispatch table
# ---------------------------------------------------------------------------

def test_model_specs_is_the_only_dispatch_table():
    assert set(MODEL_CHOICES) == set(MODEL_SPECS)
    shipped = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    for spec in MODEL_SPECS.values():
        assert spec.cfg_block in shipped
        assert spec.normalization in NORMALIZATION_CHOICES
        assert set(spec.color_jitter) == {"brightness", "contrast", "saturation", "hue", "p"}


def test_all_medsam_arms_share_one_config_block():
    assert MODEL_SPECS["medsam"].cfg_block == "medsam"
    assert MODEL_SPECS["medsam_minmax"].cfg_block == "medsam"
    assert MODEL_SPECS["medsam_ctrl"].cfg_block == "medsam"


def test_medsam_keeps_imagenet_and_standard_jitter():
    spec = MODEL_SPECS["medsam"]
    assert spec.normalization == "imagenet"
    assert spec.color_jitter == STANDARD_COLOR_JITTER


def test_medsam_minmax_uses_minmax_with_standard_jitter(base_cfg):
    spec = MODEL_SPECS["medsam_minmax"]
    assert spec.normalization == "minmax"
    assert spec.color_jitter == STANDARD_COLOR_JITTER

    _write(base_cfg, "run.yaml", "base_config: base.yaml\nrun: {model: medsam_minmax, seed: 42}\n")
    plan = build_run_plan(load_run_config(base_cfg / "run.yaml"))
    assert plan.checkpoint_dir == "checkpoints/medsam_minmax/seed42"
    assert plan.checkpoint_dir != "checkpoints/medsam/seed42"


def test_medsam_ctrl_is_imagenet_with_brightness_and_contrast_off():
    spec = MODEL_SPECS["medsam_ctrl"]
    assert spec.normalization == "imagenet"
    assert spec.color_jitter == NO_BC_COLOR_JITTER
    assert spec.color_jitter["brightness"] == 0.0
    assert spec.color_jitter["contrast"] == 0.0
    assert spec.color_jitter["saturation"] == STANDARD_COLOR_JITTER["saturation"]


def test_resolve_color_jitter_returns_a_copy():
    cj = resolve_color_jitter("medsam")
    cj["brightness"] = 999
    assert MODEL_SPECS["medsam"].color_jitter["brightness"] == STANDARD_COLOR_JITTER["brightness"]


def test_config_normalization_override_is_rejected(base_cfg):
    _write(base_cfg, "run.yaml", """
        base_config: base.yaml
        medsam: {normalization: minmax}
        run: {model: medsam, seed: 42}
    """)
    cfg = load_run_config(base_cfg / "run.yaml")
    with pytest.raises(ValueError, match="bound to the model key"):
        build_run_plan(cfg)

    # Checked for every model, not just the one selected by run.model.
    with pytest.raises(ValueError, match="bound to the model key"):
        build_run_plan(cfg, {"model": "unet"})


def test_config_color_jitter_override_is_rejected(base_cfg):
    _write(base_cfg, "run.yaml", """
        base_config: base.yaml
        medsam: {color_jitter: {brightness: 0.0}}
        run: {model: medsam, seed: 42}
    """)
    cfg = load_run_config(base_cfg / "run.yaml")
    with pytest.raises(ValueError, match="bound to the model key"):
        build_run_plan(cfg)


def test_describe_plan_shows_protocol(base_cfg):
    plan = build_run_plan(load_run_config(base_cfg / "base.yaml"), {"model": "medsam_ctrl"})
    text = describe_plan(plan)
    assert "Normalization" in text and "imagenet" in text
    assert "Color jitter" in text and "brightness=0.0" in text


# ---------------------------------------------------------------------------
# Zero-shot registry
# ---------------------------------------------------------------------------

def test_build_zeroshot_plan_resolves_protocol_from_code_not_yaml(base_cfg):
    cfg = load_run_config(base_cfg / "base.yaml")
    plan = build_zeroshot_plan(cfg, "vanilla_medsam_minmax")
    assert plan.model_type == "vit_b"
    assert plan.normalization == "minmax"
    assert plan.checkpoint == "medsam_vit_b.pth"
    assert plan.local_results_dir == "results/vanilla_medsam_minmax/seed0"
    text = describe_zeroshot_plan(plan)
    assert "vanilla_medsam_minmax" in text and "minmax" in text

    published = build_zeroshot_plan(cfg, "vanilla_sam")
    assert published.published is True
    assert "PUBLISHED" in describe_zeroshot_plan(published)


def test_build_zeroshot_plan_unknown_baseline_lists_choices(base_cfg):
    cfg = load_run_config(base_cfg / "base.yaml")
    with pytest.raises(ValueError, match="vanilla_sam"):
        build_zeroshot_plan(cfg, "nope")


def test_build_zeroshot_plan_missing_checkpoint_entry_names_the_yaml_path(base_cfg):
    _write(base_cfg, "base2.yaml", """
        data: {root: data/polyp, img_size: 352, num_workers: 4}
        zeroshot: {prompt: box, box_padding: 5, checkpoints: {}}
    """)
    cfg = load_run_config(base_cfg / "base2.yaml")
    with pytest.raises(ValueError, match="zeroshot.checkpoints"):
        build_zeroshot_plan(cfg, "vanilla_sam")


# ---------------------------------------------------------------------------
# Shipped-config tests
# ---------------------------------------------------------------------------

def test_shipped_config_builds_every_model():
    cfg = load_run_config(REPO / "configs" / "run.yaml")
    for model in MODEL_CHOICES:
        plan = build_run_plan(cfg, {"model": model})
        assert plan.model == model


def test_shipped_config_declares_no_protocol_keys():
    cfg = load_run_config(REPO / "configs" / "run.yaml")
    for spec in MODEL_SPECS.values():
        build_run_plan(cfg, {"model": "unet"})  # any model triggers the full-cfg scan; must not raise
    for block_name in {s.cfg_block for s in MODEL_SPECS.values()}:
        block = cfg.get(block_name, {}) or {}
        assert "normalization" not in block
        assert "color_jitter" not in block


def test_shipped_zeroshot_checkpoints_cover_every_spec():
    cfg = load_run_config(REPO / "configs" / "run.yaml")
    checkpoints = cfg["zeroshot"]["checkpoints"]
    for baseline in ZEROSHOT_CHOICES:
        assert baseline in checkpoints
        plan = build_zeroshot_plan(cfg, baseline)
        assert plan.checkpoint == checkpoints[baseline]
