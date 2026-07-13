# FILE MAP | Unit tests for the PURE results-consolidation layer (src/results_summary.py).
#   No torch/numpy/pandas. Builds fake metrics.json trees, then checks discovery, flattening,
#   multi-seed mean/std aggregation, local-over-Drive precedence, and the written summary files.
"""GPU-free tests for results consolidation."""

import json
from pathlib import Path

from src.results_summary import (
    aggregate_by_model,
    build_summary,
    discover_metrics,
    flatten,
)


def _payload(model, backbone, seed, seen, unseen, trainable=830_000, total=641_000_000):
    """A metrics.json payload matching src/training/reporting.build_metrics_payload's schema."""
    ev = {
        "seen_kvasir": {"dice": seen}, "seen_clinicdb": {"dice": seen},
        "cvc_colondb": {"dice": unseen}, "etis_larib": {"dice": unseen},
        "cvc_300": {"dice": unseen},
    }
    return {
        "model": model, "backbone": backbone, "seed": seed,
        "device_name": "Tesla T4",
        "params": {"total": total, "trainable": trainable,
                   "trainable_pct": round(100 * trainable / total, 4)},
        "checkpoint_size_mb": 12.3,
        "timing": {"epochs_run": 40, "total_seconds": 1200.0},
        "best_val_dice": seen,
        "eval": {k: {"dice": v["dice"]} for k, v in ev.items()},
        "generalization_gap_dice": round(seen - unseen, 6),
    }


def _write_metrics(root: Path, model_dir: str, seed: int, payload: dict) -> None:
    d = root / model_dir / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(payload))


def test_flatten_computes_seen_unseen_means():
    row = flatten("sam_vit_h", _payload("sam_lora", "vit_h", 42, seen=0.90, unseen=0.79))
    assert row["model"] == "SAM-ViT-H + LoRA"
    assert row["mean_seen_dice"] == 0.90
    assert row["mean_unseen_dice"] == 0.79
    assert row["seed"] == 42
    assert row["trainable_params"] == 830_000


def test_aggregate_mean_std_over_seeds():
    rows = [
        flatten("unet", _payload("unet", "resnet34", 42, seen=0.90, unseen=0.75)),
        flatten("unet", _payload("unet", "resnet34", 43, seen=0.92, unseen=0.77)),
    ]
    agg = aggregate_by_model(rows)
    assert len(agg) == 1
    e = agg[0]
    assert e["n_seeds"] == 2 and e["seeds"] == [42, 43]
    assert e["mean_unseen_dice_mean"] == 0.76      # (0.75 + 0.77) / 2
    assert e["mean_unseen_dice_std"] == 0.01       # population std of {0.75, 0.77}


def test_single_seed_std_is_zero_not_nan():
    rows = [flatten("medsam", _payload("medsam", "vit_b", 42, seen=0.88, unseen=0.66))]
    e = aggregate_by_model(rows)[0]
    assert e["mean_unseen_dice_std"] == 0.0


def test_discover_prefers_local_over_drive(tmp_path):
    local = tmp_path / "results"
    drive = tmp_path / "drive_results"
    # same model/seed in both roots, different unseen score
    _write_metrics(local, "unet", 42, _payload("unet", "resnet34", 42, 0.9, 0.75))
    _write_metrics(drive, "unet", 42, _payload("unet", "resnet34", 42, 0.9, 0.60))
    found = discover_metrics([local, drive])  # local first
    (payload, src) = found[("unet", 42)]
    assert payload["eval"]["cvc_colondb"]["dice"] == 0.75  # local won
    assert str(local) in src


def test_build_summary_writes_all_files(tmp_path):
    local = tmp_path / "results"
    _write_metrics(local, "sam_vit_h", 42, _payload("sam_lora", "vit_h", 42, 0.90, 0.79))
    _write_metrics(local, "sam_vit_b", 42, _payload("sam_b", "vit_b", 42, 0.87, 0.70))
    out = tmp_path / "summary"
    report = build_summary([local, tmp_path / "missing_drive"], out)

    assert report["n_runs"] == 2
    assert (out / "SUMMARY.md").exists()
    assert (out / "summary_flat.csv").exists()
    assert (out / "summary_by_model.csv").exists()
    assert (out / "summary.json").exists()
    md = (out / "SUMMARY.md").read_text()
    assert "SAM-ViT-H + LoRA" in md and "SAM-ViT-B + LoRA" in md
    payload = json.loads((out / "summary.json").read_text())
    assert len(payload["runs"]) == 2 and len(payload["by_model"]) == 2
