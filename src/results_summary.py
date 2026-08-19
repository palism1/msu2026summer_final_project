# FILE MAP | Pure results-consolidation layer. Discovers every per-run metrics.json written by
#   train.py (src/training/reporting.py:build_metrics_payload) under one or more result roots
#   (local results/ and/or the Drive mirror), flattens them into one table, aggregates multi-seed
#   runs into mean +/- std per model, and renders SUMMARY.md / CSVs / JSON.
#   PURE by design: stdlib only (no torch/numpy/pandas), so aggregate_results.py runs on any machine
#   without a GPU. [DO NOT TOUCH] the split groupings — they must match reporting.generalization_gap.
#   Two tiers, never merged: "prompt-free" (trained models) and "oracle" (untrained, GT-box-
#   prompted vanilla SAM/MedSAM baselines — see TIER/tier_of/split_by_tier). The oracle tier is
#   also written here, not just read: build_zeroshot_payload + write_zeroshot_metrics let
#   05_benchmark.ipynb persist its zero-shot rows as metrics.json for this same module to pick up.
"""Consolidate per-run metrics.json files into one human- and machine-readable summary."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Iterable

# Split groupings — mirror src/training/reporting.py (_SEEN / _UNSEEN). [DO NOT TOUCH]
SEEN_SPLITS = ("seen_kvasir", "seen_clinicdb")
UNSEEN_SPLITS = ("cvc_colondb", "etis_larib", "cvc_300")
ALL_SPLITS = SEEN_SPLITS + UNSEEN_SPLITS

# Result-folder name (== plan.checkpoint_name) -> display label for the summary.
MODEL_DISPLAY = {
    "unet": "U-Net (ResNet-34)",
    "sam_vit_h": "SAM-ViT-H + LoRA",
    "sam_vit_b": "SAM-ViT-B + LoRA",
    "medsam": "MedSAM-ViT-B + LoRA",
    "vanilla_sam": "SAM ViT-H (vanilla, oracle-box)",
    "vanilla_medsam": "MedSAM ViT-B (vanilla, oracle-box)",
    "medsam_minmax": "MedSAM-ViT-B + LoRA (min-max norm)",
    "medsam_ctrl": "MedSAM-ViT-B + LoRA (ImageNet norm, jitter control)",
    "vanilla_sam_b": "SAM ViT-B (vanilla, oracle-box)",
    "vanilla_medsam_minmax": "MedSAM ViT-B (vanilla, oracle-box, min-max norm)",
    "ens_unet_samh": "Ensemble: U-Net + SAM-ViT-H (uniform)",
    "ens_unet_samh_fitted": "Ensemble: U-Net + SAM-ViT-H (fitted)",
    "ens_top3": "Ensemble: top-3 (uniform)",
    "ens_top3_fitted": "Ensemble: top-3 (fitted)",
    "casc_unet_medsam": "Cascade: U-Net box -> MedSAM min-max",
    "oracle_casc_gtbox_medsam": "Cascade: GT box -> MedSAM min-max (path-matched ceiling)",
}

# Default Drive mirror layout written by train.py (src/config.py _DEFAULT_DRIVE_RESULTS).
DEFAULT_DRIVE_RESULTS = "/content/drive/MyDrive/msu2026_checkpoints/results"

# Tier split: "oracle" = untrained, GT-box-prompted zero-shot baselines (upper bound, not a fair
# peer); "derived" = ensembles and the cascade (combinations of the models above; no ground truth
# at inference, but not a peer of a single trained model either — see render_markdown /
# split_by_tier); everything else defaults to "prompt-free" (trained, no prompt hint at eval time).
# Never merge these three into one ranked table.
TIER = {
    "vanilla_sam": "oracle",
    "vanilla_medsam": "oracle",
    "vanilla_sam_b": "oracle",
    "vanilla_medsam_minmax": "oracle",
    "ens_unet_samh": "derived",
    "ens_unet_samh_fitted": "derived",
    "ens_top3": "derived",
    "ens_top3_fitted": "derived",
    "casc_unet_medsam": "derived",
    "oracle_casc_gtbox_medsam": "oracle",
}


def tier_of(model_dir: str) -> str:
    """Oracle = untrained, GT-prompted baseline. Derived = an ensemble or cascade of the models
    above. Any results dir named vanilla_*/oracle_* is oracle, and any named ens_*/casc_* is
    derived, by convention — so a newly added baseline or combination can never silently land in
    the fair prompt-free table."""
    if model_dir in TIER:
        return TIER[model_dir]
    if model_dir.startswith(("vanilla", "oracle")):
        return "oracle"
    if model_dir.startswith(("ens_", "casc_")):
        return "derived"
    return "prompt-free"


def _display_name(model_dir: str) -> str:
    return MODEL_DISPLAY.get(model_dir, model_dir)


def _merge_inference_sidecar(payload: dict, mpath: Path) -> dict:
    """Merge a sibling ``inference.json`` into ``payload["inference"]``, but only when the payload
    has no inline ``inference`` key already. Reads the sidecar from ``mpath.parent`` — the same
    directory that supplied this row — so a Drive-sourced row never picks up a local sidecar for
    a different run. Published rows (no sidecar on disk) pass through untouched."""
    if "inference" in payload:
        return payload
    sidecar = mpath.parent / "inference.json"
    if not sidecar.is_file():
        return payload
    try:
        inference = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return payload
    return {**payload, "inference": inference}


def discover_metrics(roots: Iterable[str | Path]) -> dict[tuple[str, int], tuple[dict, str]]:
    """
    Find every ``<root>/<model_dir>/seed<N>/metrics.json`` across the given roots.

    Roots are tried in order; the FIRST root that has a given ``(model_dir, seed)`` wins, so pass
    local ``results/`` before the Drive mirror to prefer local copies. Returns a dict keyed by
    ``(model_dir, seed)`` -> (parsed payload, source path str). Missing roots are skipped silently.
    A sibling ``inference.json`` next to the chosen ``metrics.json`` is merged into
    ``payload["inference"]`` (see ``_merge_inference_sidecar``).
    """
    found: dict[tuple[str, int], tuple[dict, str]] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for mpath in sorted(root.glob("*/seed*/metrics.json")):
            seed_dir = mpath.parent.name          # "seed42"
            model_dir = mpath.parent.parent.name  # "sam_vit_h"
            try:
                seed = int(seed_dir.replace("seed", ""))
            except ValueError:
                continue
            key = (model_dir, seed)
            if key in found:  # earlier root already supplied this run
                continue
            try:
                payload = json.loads(mpath.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            payload = _merge_inference_sidecar(payload, mpath)
            found[key] = (payload, str(mpath))
    return found


def _mean_over(eval_block: dict, splits: Iterable[str]) -> float | None:
    vals = [eval_block[s]["dice"] for s in splits if s in eval_block]
    return sum(vals) / len(vals) if vals else None


def flatten(model_dir: str, payload: dict) -> dict:
    """One flat row (accuracy + efficiency) for a single run's metrics.json payload."""
    ev = payload.get("eval", {})
    params = payload.get("params", {})
    timing = payload.get("timing", {})
    total_s = timing.get("total_seconds")
    row = {
        "model_dir": model_dir,
        "model": _display_name(model_dir),
        "tier": tier_of(model_dir),
        "backbone": payload.get("backbone"),
        "seed": payload.get("seed"),
        "normalization": payload.get("normalization"),
    }
    for s in ALL_SPLITS:
        row[s] = round(ev[s]["dice"], 6) if s in ev else None
    mean_seen = _mean_over(ev, SEEN_SPLITS)
    mean_unseen = _mean_over(ev, UNSEEN_SPLITS)
    row["mean_seen_dice"] = round(mean_seen, 6) if mean_seen is not None else None
    row["mean_unseen_dice"] = round(mean_unseen, 6) if mean_unseen is not None else None
    row["generalization_gap_dice"] = payload.get("generalization_gap_dice")
    row["best_val_dice"] = payload.get("best_val_dice")
    row["trainable_params"] = params.get("trainable")
    row["total_params"] = params.get("total")
    row["trainable_pct"] = params.get("trainable_pct")
    row["checkpoint_size_mb"] = payload.get("checkpoint_size_mb")
    row["epochs_run"] = timing.get("epochs_run")
    row["train_minutes"] = round(total_s / 60, 2) if total_s is not None else None
    row["device_name"] = payload.get("device_name")

    # Always set, defaulting to None — load-bearing. _write_csv derives its field names from
    # rows[0].keys(); a key present only on a later row is dropped by DictWriter with no error,
    # silently losing the column from summary_flat.csv / summary.json (see module FILE MAP).
    inference = payload.get("inference") or {}
    row["infer_gpu_seconds_per_image"] = inference.get("gpu_seconds_per_image")
    row["infer_n_images"] = inference.get("n_images")
    if row["infer_gpu_seconds_per_image"] is not None and row["mean_unseen_dice"] is not None:
        gps = row["infer_gpu_seconds_per_image"]
        row["unseen_dice_per_gpu_second"] = (
            round(row["mean_unseen_dice"] / gps, 6) if gps else None
        )
    else:
        row["unseen_dice_per_gpu_second"] = None
    row["accepted_drift"] = inference.get("accepted_drift") or payload.get("accepted_drift")
    return row


# Numeric columns that get mean +/- std when a model has multiple seeds.
_AGG_METRICS = (
    *ALL_SPLITS,
    "mean_seen_dice",
    "mean_unseen_dice",
    "generalization_gap_dice",
    "train_minutes",
    "infer_gpu_seconds_per_image",
    "unseen_dice_per_gpu_second",
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Population mean & std; std is 0.0 for a single value (never NaN)."""
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return round(mean, 6), round(std, 6)


def aggregate_by_model(rows: list[dict]) -> list[dict]:
    """Group flat rows by model and compute mean +/- std of the accuracy metrics across seeds."""
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model_dir"], []).append(r)

    agg: list[dict] = []
    for model_dir, group in by_model.items():
        seeds = sorted(r["seed"] for r in group if r["seed"] is not None)
        entry = {
            "model_dir": model_dir,
            "model": _display_name(model_dir),
            "n_seeds": len(group),
            "seeds": seeds,
            "trainable_params": group[0].get("trainable_params"),
            "accepted_drift": any(r.get("accepted_drift") for r in group),
        }
        for metric in _AGG_METRICS:
            vals = [r[metric] for r in group if r.get(metric) is not None]
            if vals:
                mean, std = _mean_std(vals)
                entry[f"{metric}_mean"] = mean
                entry[f"{metric}_std"] = std
            else:
                entry[f"{metric}_mean"] = None
                entry[f"{metric}_std"] = None
        agg.append(entry)
    agg.sort(key=lambda e: (e.get("mean_unseen_dice_mean") is None,
                            -(e.get("mean_unseen_dice_mean") or 0)))
    return agg


def split_by_tier(agg: list[dict]) -> dict[str, list[dict]]:
    """Split aggregated rows into {"prompt-free": [...], "derived": [...], "oracle": [...]},
    order preserved.

    ``aggregate_by_model`` already sorts by mean unseen Dice descending, so each tier's slice
    comes out sorted too — no re-sort needed. Never combine these three lists into one table:
    that's the whole point of the tier split (see module docstring / TIER)."""
    out: dict[str, list[dict]] = {"prompt-free": [], "derived": [], "oracle": []}
    for e in agg:
        out.setdefault(tier_of(e["model_dir"]), []).append(e)
    return out


def build_zeroshot_payload(model_key: str, backbone: str, eval_results: dict,
                           total_params: int, device_name: str | None,
                           prompt_protocol: str, normalization: str | None = None) -> dict:
    """
    Build a metrics.json-shaped payload for an untrained (zero-shot) vanilla SAM/MedSAM oracle
    baseline, so it round-trips through discover_metrics/flatten exactly like a trained run's
    payload: 0 trainable params, no training time, and the oracle-box prompt protocol recorded
    for provenance. ``eval_results`` is ``{split_key: {"dice": ..., ...}}``, same shape
    MetricTracker.compute() returns and src/training/reporting.build_metrics_payload uses.

    ``normalization`` is included only when not None, so 05_benchmark.ipynb's existing call
    (which doesn't pass it) keeps producing the same payload shape as before.
    """
    mean_seen = _mean_over(eval_results, SEEN_SPLITS)
    mean_unseen = _mean_over(eval_results, UNSEEN_SPLITS)
    gap = (mean_seen - mean_unseen) if (mean_seen is not None and mean_unseen is not None) else None
    payload = {
        "model": model_key,
        "backbone": backbone,
        "seed": 0,
        "device_name": device_name,
        "params": {
            "total": total_params,
            "trainable": 0,
            "trainable_pct": 0.0 if total_params else None,
        },
        "checkpoint_size_mb": None,
        "timing": {"epochs_run": 0, "total_seconds": 0},
        "best_val_dice": None,
        "eval": {k: {m: round(v, 6) for m, v in sc.items()} for k, sc in eval_results.items()},
        "generalization_gap_dice": round(gap, 6) if gap is not None else None,
        "prompt_protocol": prompt_protocol,
        "vanilla": True,
    }
    if normalization is not None:
        payload["normalization"] = normalization
    return payload


def write_run_metrics(results_root: str | Path, model_key: str, seed: int, payload: dict) -> Path:
    """Write any run's payload to ``<results_root>/<model_key>/seed<seed>/metrics.json``.

    The one writer every caller (train.py, zeroshot_eval.py, ensemble_eval.py, cascade_eval.py)
    goes through, so the path scheme lives in exactly one place."""
    out_dir = Path(results_root) / model_key / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_zeroshot_metrics(results_root: str | Path, model_key: str, payload: dict) -> Path:
    """Write a zero-shot baseline payload to ``<results_root>/<model_key>/seed0/metrics.json``.

    Delegates to write_run_metrics with seed=0, so 05_benchmark.ipynb keeps working unchanged."""
    return write_run_metrics(results_root, model_key, 0, payload)


def build_derived_payload(model_key: str, seed: int, eval_results: dict,
                          member_payloads: list[dict], backbones: list[str],
                          normalizations: list[str], device_name: str | None = None,
                          extras: dict | None = None) -> dict:
    """
    Build a metrics.json-shaped payload for a derived method (ensemble or cascade) — an
    accuracy-fair, more-expensive combination of the trained models above. Field rules
    (docs/PLAN_ENSEMBLE.md Phase 2):

    - params.trainable / params.total: summed over members (the method costs all of them).
    - checkpoint_size_mb: summed over members (download footprint).
    - timing.total_seconds: summed over member training times (renders as a correct "Train min").
    - timing.epochs_run: 0 (no training happened here).
    - best_val_dice: None (no validation loop).
    - backbone / normalization: derived via merge_backbones / merge_normalization — the single
      source of truth for those strings; never hardcode "mixed" or a backbone string elsewhere.

    ``member_payloads`` is the list of each member's own metrics.json payload (for params /
    checkpoint size / timing); ``backbones`` and ``normalizations`` are the per-member protocol
    values (e.g. from src.config.MODEL_SPECS). ``extras`` (e.g. the ``ensemble`` or ``cascade``
    block) is merged into the payload verbatim.
    """
    from src.ensemble.combine import merge_backbones, merge_normalization  # lazy: numpy import

    total_trainable = sum(p.get("params", {}).get("trainable") or 0 for p in member_payloads)
    total_params = sum(p.get("params", {}).get("total") or 0 for p in member_payloads)
    total_ckpt_mb = sum(p.get("checkpoint_size_mb") or 0 for p in member_payloads)
    total_seconds = sum(p.get("timing", {}).get("total_seconds") or 0 for p in member_payloads)

    mean_seen = _mean_over(eval_results, SEEN_SPLITS)
    mean_unseen = _mean_over(eval_results, UNSEEN_SPLITS)
    gap = (mean_seen - mean_unseen) if (mean_seen is not None and mean_unseen is not None) else None

    payload = {
        "model": model_key,
        "backbone": merge_backbones(backbones),
        "seed": seed,
        "normalization": merge_normalization(normalizations),
        "device_name": device_name,
        "params": {
            "total": total_params,
            "trainable": total_trainable,
            "trainable_pct": round(100 * total_trainable / total_params, 4) if total_params else None,
        },
        "checkpoint_size_mb": round(total_ckpt_mb, 2) if total_ckpt_mb else None,
        "timing": {"epochs_run": 0, "total_seconds": round(total_seconds, 2)},
        "best_val_dice": None,
        "eval": {k: {m: round(v, 6) for m, v in sc.items()} for k, sc in eval_results.items()},
        "generalization_gap_dice": round(gap, 6) if gap is not None else None,
    }
    if extras:
        payload.update(extras)
    return payload


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


# Cost-table reference model: every "delta" and "cost multiple" column is relative to this
# model's GPU-s per image. SAM-ViT-H is the strongest single trained model, so a cheaper method
# reads as a savings and a more expensive one (the cascade) reads as a multiple of the reference.
COST_REFERENCE_MODEL_DIR = "sam_vit_h"


def _drift_marker(e: dict) -> str:
    return " †" if e.get("accepted_drift") else ""


def _agg_table_rows(entries: list[dict]) -> list[str]:
    """Header + body lines for one tier's "mean ± std over seeds" table."""
    out = ["| Model | Seeds | Mean seen mDice | Mean unseen mDice | Gap (seen−unseen) | "
           "Trainable params | Train min |",
           "|---|---|---|---|---|---|---|"]
    for e in entries:
        seen = f"{_fmt(e['mean_seen_dice_mean'])} ± {_fmt(e['mean_seen_dice_std'])}"
        unseen = f"{_fmt(e['mean_unseen_dice_mean'])} ± {_fmt(e['mean_unseen_dice_std'])}"
        gap = f"{_fmt(e['generalization_gap_dice_mean'])} ± {_fmt(e['generalization_gap_dice_std'])}"
        tp = f"{e['trainable_params']:,}" if e.get("trainable_params") is not None else "—"
        tm = _fmt(e.get("train_minutes_mean"), 1)
        n = f"{e['n_seeds']} ({', '.join(str(s) for s in e['seeds'])})" if e["seeds"] else "0"
        out.append(f"| {e['model']}{_drift_marker(e)} | {n} | {seen} | {unseen} | {gap} | {tp} | {tm} |")
    return out


def _cost_table_rows(agg: list[dict]) -> list[str]:
    """Header + body lines for the inference-cost table. Rows without a recorded
    infer_gpu_seconds_per_image are omitted — this table only, not the tier tables."""
    priced = [e for e in agg if e.get("infer_gpu_seconds_per_image_mean") is not None]
    if not priced:
        return []
    ref = next((e for e in priced if e["model_dir"] == COST_REFERENCE_MODEL_DIR), None)
    ref_cost = ref["infer_gpu_seconds_per_image_mean"] if ref else None

    out = ["| Model | Mean unseen mDice | GPU-s per image | Delta GPU-s vs reference | "
           "Cost multiple | Unseen Dice per GPU-s |",
           "|---|---|---|---|---|---|"]
    for e in priced:
        cost = e["infer_gpu_seconds_per_image_mean"]
        delta = f"{cost - ref_cost:+.4f}" if ref_cost is not None else "—"
        multiple = f"{cost / ref_cost:.2f}x" if ref_cost else "—"
        dps = _fmt(e.get("unseen_dice_per_gpu_second_mean"), 2)
        out.append(f"| {e['model']}{_drift_marker(e)} | {_fmt(e.get('mean_unseen_dice_mean'))} | "
                   f"{_fmt(cost, 4)} | {delta} | {multiple} | {dps} |")
    return out


def render_markdown(rows: list[dict], agg: list[dict]) -> str:
    """Human-readable SUMMARY.md: two tier-separated per-model tables, plus a per-run detail
    table. The two tiers are never merged into one ranked table — an untrained oracle-box
    baseline outscoring a trained model on unseen data is an artifact of the prompt it was
    given, not evidence it generalizes better; keeping it in its own table (with its own
    caption) is the whole point of the tier split."""
    lines: list[str] = ["# Results Summary", ""]
    lines.append(
        "_Consolidated from per-run `metrics.json` by `aggregate_results.py`. No notebooks "
        "were re-run. Two tables follow: models trained and evaluated with no prompt at all "
        "(the fair comparison), and untrained oracle-box baselines evaluated with a "
        "ground-truth-derived box prompt the trained models never see (an upper bound, not a "
        "fair peer)._"
    )
    lines.append("")

    tiers = split_by_tier(agg)

    lines.append("## Prompt-free (trained, mean ± std over seeds)")
    lines.append("")
    lines += _agg_table_rows(tiers.get("prompt-free", []))
    lines.append("")

    best_single = max(
        (e for e in tiers.get("prompt-free", []) if e.get("mean_unseen_dice_mean") is not None),
        key=lambda e: e["mean_unseen_dice_mean"], default=None,
    )
    best_note = (f" The best single-model unseen mDice is {best_single['model']} at "
                f"{_fmt(best_single['mean_unseen_dice_mean'])}."
                if best_single is not None else "")
    lines.append(
        "## Derived methods (combinations of the models above; no ground truth at inference)"
    )
    lines.append("")
    lines.append(
        "_Ensembles and the box cascade read no ground truth at inference, so they are fair on "
        "accuracy — but each one costs more compute than any single member, and they are "
        "combinations of the models above, not a new architecture, so they get their own table "
        "rather than a rank inside the prompt-free comparison." + best_note + "_"
    )
    lines.append("")
    derived_entries = tiers.get("derived", [])
    if derived_entries:
        lines += _agg_table_rows(derived_entries)
    else:
        lines.append("_None recorded yet. Run `ensemble_eval.py` / `cascade_eval.py` to "
                     "populate `results/ens_*` / `results/casc_*`._")
    lines.append("")

    lines.append("## Oracle-box baselines (not trained — GT-derived box prompt; "
                 "upper bound, not a fair peer)")
    lines.append("")
    oracle_entries = tiers.get("oracle", [])
    if oracle_entries:
        lines += _agg_table_rows(oracle_entries)
    else:
        lines.append("_None recorded yet. Run `05_benchmark.ipynb`'s zero-shot cell to "
                     "populate `results/vanilla_sam` / `results/vanilla_medsam`._")
    lines.append("")

    cost_rows = _cost_table_rows(agg)
    lines.append("## Inference cost (rows that recorded it)")
    lines.append("")
    if cost_rows:
        lines += cost_rows
        lines.append("")
        lines.append(f"_Reference model: `{COST_REFERENCE_MODEL_DIR}`. † marks a row with "
                     f"accepted verification drift (see docs/DECISIONS.md)._")
    else:
        lines.append("_None recorded yet. Rows populate once `predict_cache.py` / "
                     "`ensemble_eval.py` / `cascade_eval.py` write an `inference` block._")
    lines.append("")

    lines.append("## Per run (each model × seed)")
    lines.append("")
    header = (["Model", "Seed", "Tier"] + [s for s in ALL_SPLITS]
              + ["Mean seen", "Mean unseen", "Gap", "Params", "Ckpt MB", "Epochs", "Train min",
                 "Device"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in sorted(rows, key=lambda x: (x["model"], x["seed"] if x["seed"] is not None else -1)):
        cells = [r["model"], _fmt(r["seed"]), r.get("tier", "—")]
        cells += [_fmt(r[s]) for s in ALL_SPLITS]
        cells += [
            _fmt(r["mean_seen_dice"]), _fmt(r["mean_unseen_dice"]),
            _fmt(r["generalization_gap_dice"]),
            f"{r['trainable_params']:,}" if r.get("trainable_params") is not None else "—",
            _fmt(r["checkpoint_size_mb"], 1), _fmt(r["epochs_run"]),
            _fmt(r["train_minutes"], 1), r.get("device_name") or "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def write_summary(rows: list[dict], agg: list[dict], out_dir: str | Path,
                  sources: dict[tuple[str, int], str] | None = None) -> dict[str, str]:
    """Write SUMMARY.md, summary_flat.csv, summary_by_model.csv, summary.json into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "SUMMARY.md"
    md_path.write_text(render_markdown(rows, agg))

    flat_path = out / "summary_flat.csv"
    _write_csv(flat_path, [{k: v for k, v in r.items() if k != "model_dir"} for r in rows])

    by_model_path = out / "summary_by_model.csv"
    _write_csv(by_model_path, [{k: (v if k != "seeds" else " ".join(map(str, v)))
                                for k, v in e.items() if k != "model_dir"} for e in agg])

    json_path = out / "summary.json"
    json_path.write_text(json.dumps(
        {"runs": rows, "by_model": agg,
         "sources": {f"{m}/seed{s}": p for (m, s), p in (sources or {}).items()}},
        indent=2))

    return {
        "summary_md": str(md_path),
        "summary_flat_csv": str(flat_path),
        "summary_by_model_csv": str(by_model_path),
        "summary_json": str(json_path),
    }


def build_summary(roots: Iterable[str | Path], out_dir: str | Path) -> dict:
    """End-to-end: discover -> flatten -> aggregate -> write. Returns a small coverage report."""
    discovered = discover_metrics(roots)
    rows = [flatten(model_dir, payload) for (model_dir, _seed), (payload, _src) in
            sorted(discovered.items())]
    agg = aggregate_by_model(rows)
    sources = {key: src for key, (_payload, src) in discovered.items()}
    written = write_summary(rows, agg, out_dir, sources)
    return {
        "n_runs": len(rows),
        "runs": sorted(f"{m}/seed{s}" for (m, s) in discovered),
        "written": written,
    }
