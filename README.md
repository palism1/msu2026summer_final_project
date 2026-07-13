# Can LoRA-Adapted SAM Match Specialist Polyp Networks?
### A Cross-Dataset Generalization Study in Colonoscopy

**MSU Summer 2026 — Masters Final Project**

---

## Project Summary

Can a segmentation foundation model, adapted with only a small number of extra parameters, match specialist polyp networks on standard benchmarks while degrading less on unseen datasets?

We compare:
| Method | Type | Trained here? | Role |
|---|---|---|---|
| U-Net (ResNet-34) | CNN | Yes | Specialist baseline |
| PraNet | CNN + attention | Reference | Specialist SOTA |
| Polyp-PVT | Transformer | Reference | Specialist SOTA |
| SAM (zero-shot) | Foundation | No | Foundation baseline |
| SAM-LoRA | Foundation + PEFT | Yes (adapters) | Main contribution |

> **Backbone-confound ablation:** `sam_b` (SAM ViT-B + LoRA) has the same backbone size as MedSAM but
> generic SAM weights, so comparing it to MedSAM ViT-B separates *backbone capacity* from *medical
> pretraining*. Train it with `--model sam_b`; it appears as an optional row in the benchmark.

---

## Repository Layout

```
.
├── configs/
│   ├── base.yaml              # Shared hyperparameters
│   └── run.yaml               # Per-run overrides: selects model / seed / outputs
├── src/
│   ├── config.py             # Config load/merge + run-plan resolution (torch-free)
│   ├── results_summary.py    # Consolidate per-run metrics.json -> results/summary (torch-free)
│   ├── data/
│   │   ├── dataset.py         # PolypDataset + PraNet split builder
│   │   └── transforms.py      # Albumentations train/val pipelines
│   ├── metrics/
│   │   └── segmentation.py    # mDice, mIoU, MAE, wFm, Sm, Em
│   ├── models/
│   │   ├── unet.py            # U-Net via segmentation-models-pytorch
│   │   └── sam_adapter.py     # SAM + LoRA + lightweight decoder
│   └── training/
│       ├── engine.py          # Config-driven training loop (all models)
│       └── reporting.py       # Writes metrics.json, mask overlays, run.log
├── notebooks/
│   ├── 01_data_pipeline.ipynb     # Download data, verify splits (run once)
│   ├── train_colab.ipynb          # Colab wrapper: pick models/seeds in cell 1, runs train.py
│   ├── 05_benchmark.ipynb         # Compare all trained models side by side
│   └── 06_findings.ipynb          # Illustrate the two-tier (prompt-free vs oracle) result
├── train.py                   # CLI training entry point
├── evaluate.py                # CLI evaluation (all 5 splits)
├── aggregate_results.py       # Consolidate results -> results/summary (no GPU, no re-run)
└── requirements.txt
```

> The per-model training notebooks (`02_unet_baseline`, `03_sam_lora`, `04_medsam_lora`) were
> superseded by the config-driven runner (`train.py`) and are preserved on the
> `backup/per-model-notebooks` branch rather than kept on `main`.

---

## Datasets (PraNet Protocol)

| Dataset | Role | Size |
|---|---|---|
| Kvasir-SEG | Train + seen test | 1,000 |
| CVC-ClinicDB | Train + seen test | 612 |
| CVC-ColonDB | Unseen test | 380 |
| ETIS-Larib | Unseen test | 196 |
| CVC-300 (EndoScene) | Unseen test | 60 |

Training split: 900 Kvasir + 550 CVC-ClinicDB = 1,450 images

---

## Quick Start (Google Colab)

1. Open `notebooks/01_data_pipeline.ipynb` in Colab. It installs deps, clones the repo, downloads the datasets, and verifies the splits end-to-end (run once).
2. Choose what to train by editing `run.model` in `configs/run.yaml` (`unet` | `sam_lora` | `medsam`), then commit and push.
3. Open `notebooks/train_colab.ipynb` and run its three cells: it fetches the data plus the one checkpoint your config needs and runs `train.py` under a single shared protocol. Results land in `results/<model>/seed<seed>/` and mirror to Drive.
4. Open `notebooks/05_benchmark.ipynb` to compare all trained models side by side.
5. Run `python aggregate_results.py` to collect every run's metrics into `results/summary/SUMMARY.md` — no GPU, and no re-running notebooks.

---

## Evaluation Metrics

All metrics match the PraNet evaluation protocol:
- mDice, mIoU (primary)
- Weighted F-beta (beta^2=0.3)
- S-measure (Sm) -- Fan et al. ICCV 2017
- Enhanced-alignment measure (Em) -- Fan et al. IJCAI 2018
- MAE

---

## CLI Usage

```bash
# Train the model selected in configs/run.yaml (unet | sam_lora | medsam | sam_b)
python train.py --config configs/run.yaml

# Offline sanity check — no GPU or torch required
python train.py --config configs/run.yaml --dry-run

# Evaluate all 5 splits
python evaluate.py --config configs/base.yaml --model unet \
                   --checkpoint checkpoints/unet/seed42/best.pt

# Consolidate every run's metrics.json into results/summary/ (no GPU, no notebook re-run)
python aggregate_results.py
```

---

## Work Plan

| Phase | Weeks | Deliverable |
|---|---|---|
| 1 | 1-2 | Data pipeline, splits, metrics (done) |
| 2 | 3-5 | U-Net baseline + evaluation harness |
| 3 | 6-8 | SAM-LoRA adaptation |
| 4 | 9-10 | Full benchmark, ablations, efficiency |
| 5 | 11-12 | Web demo, final report |
