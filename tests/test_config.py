# FILE MAP | Unit tests for the PURE config layer in src/config.py.
#   No torch/numpy needed. Covers run.yaml load+merge, plan/path resolution, the legacy
#   flat-config path, and validation. [DO NOT TOUCH] the checkpoint-path assertions — they
#   encode the contract evaluate.py + notebooks/05_benchmark.ipynb rely on.
"""GPU-free tests for config loading and run-plan resolution."""

import textwrap

import pytest

from src.config import (
    MODEL_CHOICES,
    PIPELINE_STAGES,
    build_run_plan,
    describe_plan,
    load_run_config,
)


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
        medsam: {model_type: vit_b, checkpoint: medsam_vit_b.pth, lora_r: 4, lora_alpha: 8, lora_dropout: 0.1}
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


def test_sam_checkpoint_name_encodes_backbone(base_cfg):
    _write(base_cfg, "run.yaml", "base_config: base.yaml\nrun: {model: sam_lora, seed: 42}\n")
    plan = build_run_plan(load_run_config(base_cfg / "run.yaml"))
    # notebooks/05_benchmark.ipynb looks for checkpoints/sam_vit_h/seed42
    assert plan.checkpoint_name == "sam_vit_h"
    assert plan.checkpoint_dir == "checkpoints/sam_vit_h/seed42"


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
    assert set(MODEL_CHOICES) == {"unet", "sam_lora", "medsam"}
