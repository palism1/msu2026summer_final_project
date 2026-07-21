---
status: living
last_updated: 2026-07-21
---

<!-- Accuracy-vs-cost findings for the cross-dataset polyp-segmentation study. Doc class: living
     (draft) — it reports the three trained prompt-free models over three seeds. It becomes final
     once sam_b and the vanilla zero-shot baselines run; mark immutable then. Numbers here are read
     from results/summary/ (aggregate_results.py); the runnable version is notebooks/07_report.ipynb. -->

# Findings — Cross-Dataset Polyp Segmentation

**Question:** can a LoRA-adapted SAM match a specialist U-Net on the PraNet benchmark while
generalizing better to datasets neither model trained on, and at what training cost?

**Answer:** SAM-ViT-H + LoRA generalizes best. It leads on every unseen split, holds a mean
unseen Dice of 0.806 against U-Net's 0.755, and its seen-to-unseen drop is roughly half of
U-Net's — while training 0.83M parameters against a frozen backbone, about 3% of the 24.4M
U-Net trains from scratch. U-Net still wins on the two splits both models trained on. MedSAM +
LoRA trails both on every split.

All numbers below are means over three seeds (42, 43, 44), one benchmark family (PraNet), one
A100. Seen splits: Kvasir, ClinicDB. Unseen splits: CVC-ColonDB, ETIS-LaribDB, CVC-300.

## Accuracy and cost by model

| Model | Seen mDice | Unseen mDice | Gap (seen−unseen) | Trainable params | Ckpt MB | Train min |
|---|---|---|---|---|---|---|
| SAM-ViT-H + LoRA | 0.887 ± 0.008 | **0.806 ± 0.011** | **0.082 ± 0.004** | 830,177 | 2533 | 120 |
| U-Net (ResNet-34) | **0.900 ± 0.008** | 0.755 ± 0.014 | 0.145 ± 0.014 | 24,436,369 | 98 | 10 |
| MedSAM-ViT-B + LoRA | 0.819 ± 0.006 | 0.661 ± 0.008 | 0.158 ± 0.011 | 322,273 | 349 | 31 |

## Per-split behavior

| Model | Kvasir (seen) | ClinicDB (seen) | CVC-ColonDB | ETIS-Larib | CVC-300 |
|---|---|---|---|---|---|
| SAM-ViT-H + LoRA | 0.906 | 0.869 | 0.785 | 0.745 | 0.887 |
| U-Net (ResNet-34) | 0.903 | 0.897 | 0.729 | 0.690 | 0.847 |
| MedSAM-ViT-B + LoRA | 0.852 | 0.786 | 0.656 | 0.500 | 0.827 |

The three models sit close together on the seen splits and spread apart on the unseen ones. The
gap is widest on ETIS-LaribDB, the hardest unseen split: SAM-ViT-H holds 0.745 there while U-Net
drops to 0.690 and MedSAM to 0.500. Generalization is where the models separate, and it tracks
the seen-to-unseen gap column above.

## What the cost buys

U-Net trains fastest (10 min) and produces the smallest checkpoint (98 MB), and it fits the seen
distribution tightest — the highest seen ClinicDB score of any model, 0.897. That fit is also why
it falls furthest on unseen data. SAM-ViT-H + LoRA costs more per run (120 min, a 2.5 GB
checkpoint dominated by the frozen backbone) and updates only 0.83M parameters, yet returns the
best unseen accuracy and the smallest gap. The accuracy-per-trainable-parameter comparison favors
the LoRA adapter; the accuracy-per-minute comparison favors U-Net. Which one matters depends on
whether the target is the benchmark or the unseen clinic.

## Limitations

- **One benchmark family.** Every split comes from PraNet's own five datasets. These numbers do
  not speak to colonoscopy data collected under different equipment or protocols.
- **Three seeds.** The std columns show run-to-run spread is real; the sample is small.
- **The MedSAM result mixes two variables.** MedSAM uses a ViT-B backbone, smaller than
  SAM-ViT-H's ViT-H, so its weaker scores reflect either the smaller backbone or the medical
  pretraining, and cannot be attributed to one. Separating them
  needs SAM-ViT-B + LoRA (`sam_b`): same backbone size and LoRA recipe as MedSAM, generic SAM
  weights. That run is designed but not yet executed.
- **No without-fine-tuning numbers here.** The vanilla zero-shot SAM/MedSAM baselines (the
  with-vs-without-fine-tuning comparison) run in `06_findings.ipynb` under an oracle box prompt
  and are pending; this document covers the trained prompt-free models only.

## Status

Draft over the three trained prompt-free models. It finalizes once `sam_b` and the vanilla
zero-shot baselines run and `aggregate_results.py` reabsorbs them; mark immutable at that point.
The runnable, chart-bearing version of this report is `notebooks/07_report.ipynb`.
