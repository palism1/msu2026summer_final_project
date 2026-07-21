<!-- FILE MAP | Repo orientation map for humans & agents. A MAP, not a manual.
     Tags used across new files: [TWEAK] safe to change  ·  [CHANGE ME] must set before use
     ·  [DO NOT TOUCH] load-bearing contract, changing it breaks something else. -->

# CLAUDE.md — Repo Map

**Purpose:** Cross-dataset polyp-segmentation study — can LoRA-adapted SAM (and MedSAM)
match specialist U-Net on the PraNet benchmark while generalizing better to unseen datasets?

## Pipeline stages (in order)
1. **Data pipeline** — `src/data/` — PraNet splits (1450 train, 5 test), `PolypDataset`, transforms.
2. **Model build** — `src/models/` — `build_unet`, `build_sam_lora` (ViT-H = `sam_lora`, ViT-B = `sam_b`), `build_medsam_lora`.
3. **Train** — `train.py` (driven by `configs/run.yaml`) → `src/config.py` + `src/training/`.
4. **Evaluate** — all 5 splits (`evaluate.py`; also inside `train.py` via `src/training/reporting.py`).
5. **Benchmark** — `notebooks/05_benchmark.ipynb` — compares all trained models side by side;
   `notebooks/06_findings.ipynb` / `notebooks/07_report.ipynb` read out the results (see
   `docs/FINDINGS.md`).

## Run order
1. `notebooks/01_data_pipeline.ipynb` — download data, verify splits (run once).
2. Train: `notebooks/train_colab.ipynb` (Colab; pick models/seeds/epochs in **cell 1** — trains all
   selected model×seed pairs in one session, skips already-trained ones, mirrors checkpoints +
   results to Drive) or locally: `python train.py --config configs/run.yaml [--model M]`
   where `M` ∈ `unet | sam_lora | medsam | sam_b`. Offline check: `... --dry-run`.
3. `notebooks/05_benchmark.ipynb` — compare results across models; `notebooks/06_findings.ipynb` /
   `notebooks/07_report.ipynb` — runnable readouts of the write-up in `docs/FINDINGS.md`.
4. `python aggregate_results.py` — consolidate every run's `metrics.json` (local or Drive mirror)
   into `results/summary/` (SUMMARY.md + CSVs + JSON). No GPU, no notebook re-run.

## Where things live
- `configs/base.yaml` — shared hyperparameters. `configs/run.yaml` — per-run overrides + outputs.
- Checkpoints: `checkpoints/<model>/seed<seed>/best.pt` (best-val-Dice only; **[DO NOT TOUCH]** path
  contract); auto-mirrored to Drive under `msu2026_checkpoints/<model>/seed<seed>/` during training.
- Results: `results/<model>/seed<seed>/` (metrics.json, mask overlays, run.log); mirrored to Drive.
- Consolidated summary: `results/summary/` (SUMMARY.md + CSVs + JSON), produced by
  `aggregate_results.py` + `src/results_summary.py` (both torch-free).
- `notebooks/02–04` — per-model exploration; superseded by `train.py` and preserved on the `backup/per-model-notebooks` branch (not on `main`).

## Doc status (see docs/DECISIONS.md, 2026-07-13)
Committed markdown declares a class in YAML frontmatter: **living** (edit freely, bump
`last_updated`), **immutable** (frozen; supersede rather than edit), **append-only** (`DECISIONS.md`).
`README.md` and this `CLAUDE.md` are exempt (tooling-rendered; git history is their timestamp).

## Pointers
- **Plan & where to pick up (start here when resuming):** `docs/PROJECT_PLAN.md`
- **Design decisions & rationale:** `docs/DECISIONS.md`  ·  **Remaining-work delegation:** `docs/DELEGATION_PLAN.md`
- **Results write-up (source of truth for all numbers):** `docs/FINDINGS.md`
- **Project overview, datasets, CLI:** `README.md`
