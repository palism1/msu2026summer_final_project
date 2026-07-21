# FILE MAP | Unit tests for the PURE results-consolidation layer (src/results_summary.py).
#   No torch/numpy/pandas. Builds fake metrics.json trees, then checks discovery, flattening,
#   multi-seed mean/std aggregation, local-over-Drive precedence, and the written summary files.
"""GPU-free tests for results consolidation."""

import json
from pathlib import Path

from src.results_summary import (
    aggregate_by_model,
    build_summary,
    build_zeroshot_payload,
    discover_metrics,
    flatten,
    render_markdown,
    split_by_tier,
    tier_of,
    write_zeroshot_metrics,
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
    assert e["train_minutes_mean"] == 20.0         # 1200 s per run in the fixture payload
    assert e["train_minutes_std"] == 0.0


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


# ---------------------------------------------------------------------------
# Oracle tier: build_zeroshot_payload, the write->discover->flatten round-trip, and the
# honesty assertion that the two tiers never end up in the same table.
# ---------------------------------------------------------------------------

def _zeroshot_eval(seen, unseen):
    return {
        "seen_kvasir": {"dice": seen}, "seen_clinicdb": {"dice": seen},
        "cvc_colondb": {"dice": unseen}, "etis_larib": {"dice": unseen},
        "cvc_300": {"dice": unseen},
    }


def test_build_zeroshot_payload_is_untrained_oracle_tier():
    payload = build_zeroshot_payload(
        "vanilla_sam", "vit_h", _zeroshot_eval(seen=0.86, unseen=0.90),
        total_params=641_000_000, device_name="Tesla T4", prompt_protocol="box",
    )
    assert payload["params"]["trainable"] == 0
    assert payload["params"]["total"] == 641_000_000
    assert payload["seed"] == 0
    assert payload["vanilla"] is True
    assert payload["prompt_protocol"] == "box"
    assert payload["generalization_gap_dice"] == round(0.86 - 0.90, 6)

    row = flatten("vanilla_sam", payload)
    assert row["tier"] == "oracle"
    assert "(vanilla, oracle-box)" in row["model"]
    assert row["trainable_params"] == 0


def test_write_zeroshot_metrics_round_trips_through_discover_and_flatten(tmp_path):
    results_root = tmp_path / "results"
    payload = build_zeroshot_payload(
        "vanilla_medsam", "vit_b", _zeroshot_eval(seen=0.80, unseen=0.84),
        total_params=93_000_000, device_name="Tesla T4", prompt_protocol="box",
    )
    written = write_zeroshot_metrics(results_root, "vanilla_medsam", payload)
    assert written == results_root / "vanilla_medsam" / "seed0" / "metrics.json"

    found = discover_metrics([results_root])
    assert ("vanilla_medsam", 0) in found
    loaded_payload, _src = found[("vanilla_medsam", 0)]
    row = flatten("vanilla_medsam", loaded_payload)
    assert row["tier"] == "oracle"
    assert row["trainable_params"] == 0
    assert "(vanilla, oracle-box)" in row["model"]


def test_oracle_tier_never_merges_into_the_prompt_free_table(tmp_path):
    """HONESTY: an oracle row that outscores every trained model must still land in its own
    table, never ranked into the prompt-free comparison."""
    local = tmp_path / "results"
    _write_metrics(local, "sam_vit_h", 42, _payload("sam_lora", "vit_h", 42, seen=0.90, unseen=0.79))
    oracle_payload = build_zeroshot_payload(
        "vanilla_sam", "vit_h", _zeroshot_eval(seen=0.86, unseen=0.95),  # beats every trained row
        total_params=641_000_000, device_name="Tesla T4", prompt_protocol="box",
    )
    write_zeroshot_metrics(local, "vanilla_sam", oracle_payload)

    found = discover_metrics([local])
    rows = [flatten(m, payload) for (m, _s), (payload, _src) in sorted(found.items())]
    agg = aggregate_by_model(rows)

    assert tier_of("vanilla_sam") == "oracle"
    assert tier_of("sam_vit_h") == "prompt-free"

    tiers = split_by_tier(agg)
    pf_dirs = {e["model_dir"] for e in tiers["prompt-free"]}
    oracle_dirs = {e["model_dir"] for e in tiers["oracle"]}
    assert pf_dirs == {"sam_vit_h"}
    assert oracle_dirs == {"vanilla_sam"}

    md = render_markdown(rows, agg)
    pf_header = "## Prompt-free (trained, mean ± std over seeds)"
    oracle_header = "## Oracle-box baselines"
    assert pf_header in md and oracle_header in md

    pf_section = md.split(pf_header, 1)[1].split(oracle_header, 1)[0]
    oracle_section = md.split(oracle_header, 1)[1]
    assert "SAM ViT-H (vanilla, oracle-box)" not in pf_section
    assert "SAM ViT-H (vanilla, oracle-box)" in oracle_section
    assert "SAM-ViT-H + LoRA" in pf_section
