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

---

## Repository Layout

```
.
├── configs/
│   └── base.yaml              # All hyperparameters
├── src/
│   ├── data/
│   │   ├── dataset.py         # PolypDataset + PraNet split builder
│   │   └── transforms.py      # Albumentations train/val pipelines
│   ├── metrics/
│   │   └── segmentation.py    # mDice, mIoU, MAE, wFm, Sm, Em
│   └── models/
│       ├── unet.py            # U-Net via segmentation-models-pytorch
│       └── sam_adapter.py     # SAM + LoRA + lightweight decoder
├── notebooks/
│   ├── 01_data_pipeline.ipynb     <- Phase 1 (start here)
│   ├── 02_unet_baseline.ipynb     <- Phase 2
│   └── 03_sam_lora.ipynb          <- Phase 3
├── train.py                   # CLI training entry point
├── evaluate.py                # CLI evaluation (all 5 splits)
└── requirements.txt
```

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

1. Open `notebooks/01_data_pipeline.ipynb` in Colab. It installs deps, clones the repo, downloads Kvasir-SEG, and verifies the pipeline end-to-end.
2. Set `GDRIVE_FILE_ID` in that notebook (from PraNet's GitHub README) to fetch all five datasets.
3. Open `notebooks/02_unet_baseline.ipynb` to train the U-Net baseline (free T4 GPU).
4. Open `notebooks/03_sam_lora.ipynb` to train SAM-LoRA (A100/L4 recommended for ViT-H).

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
# Train U-Net baseline
python train.py --config configs/base.yaml --model unet --seed 42

# Evaluate all 5 splits
python evaluate.py --config configs/base.yaml --model unet \
                   --checkpoint checkpoints/unet/seed42/best.pt
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
