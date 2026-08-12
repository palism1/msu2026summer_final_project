---
status: living
last_updated: 2026-08-12
---

<!-- Accuracy-vs-cost findings for the cross-dataset polyp-segmentation study. Doc class: living.
     Covers six trained prompt-free models over three seeds plus four oracle-box baselines. Numbers
     are read from results/summary/ (aggregate_results.py); the runnable version is
     notebooks/07_report.ipynb. -->

# Findings — Cross-Dataset Polyp Segmentation

**Question:** can a LoRA-adapted SAM match a specialist U-Net on the PraNet benchmark while
generalizing better to datasets neither model trained on, and at what training cost?

**Answer:** SAM-ViT-H + LoRA generalizes best. It leads on every unseen split, holds a mean
unseen Dice of 0.806 against U-Net's 0.755, and its seen-to-unseen drop is roughly half of
U-Net's — while training 0.83M parameters against a frozen backbone, about 3% of the 24.4M
U-Net trains from scratch. U-Net still wins on the two splits both models trained on. Two further
results sharpen the picture: even the small SAM-ViT-B + LoRA (0.761 unseen) edges past U-Net at
1.3% of its trainable parameters, and MedSAM's apparent collapse was mostly a preprocessing bug
in our pipeline. Retrained under its own min-max normalization, MedSAM recovers +0.070 unseen
Dice (0.655 → 0.725, jitter-controlled); the residual −0.036 against generic SAM at the same
backbone is the true pretraining effect, and the oracle tier shows why: MedSAM was fine-tuned to
be told where to look, and with a box prompt plus correct preprocessing it posts the best unseen
number in the study (0.925).

All numbers below are means over three seeds (42, 43, 44), one benchmark family (PraNet), one
A100. Seen splits: Kvasir, ClinicDB. Unseen splits: CVC-ColonDB, ETIS-LaribDB, CVC-300.

## Accuracy and cost by model

| Model | Seen mDice | Unseen mDice | Gap (seen−unseen) | Trainable params | Ckpt MB | Train min |
|---|---|---|---|---|---|---|
| SAM-ViT-H + LoRA | 0.887 ± 0.008 | **0.806 ± 0.011** | **0.082 ± 0.004** | 830,177 | 2533 | 120 |
| SAM-ViT-B + LoRA | 0.865 ± 0.005 | 0.761 ± 0.005 | 0.104 ± 0.007 | 322,273 | 349 | 29 |
| U-Net (ResNet-34) | **0.900 ± 0.008** | 0.755 ± 0.014 | 0.145 ± 0.014 | 24,436,369 | 98 | 10 |
| MedSAM-ViT-B + LoRA (min-max norm) | 0.834 ± 0.001 | 0.725 ± 0.004 | 0.110 ± 0.004 | 322,273 | 349 | 30 |
| MedSAM-ViT-B + LoRA (ImageNet norm, as published) | 0.819 ± 0.006 | 0.661 ± 0.008 | 0.158 ± 0.011 | 322,273 | 349 | 31 |
| MedSAM-ViT-B + LoRA (ImageNet norm, jitter control) | 0.815 ± 0.001 | 0.655 ± 0.002 | 0.160 ± 0.002 | 322,273 | 349 | 30 |

## Per-split behavior

| Model | Kvasir (seen) | ClinicDB (seen) | CVC-ColonDB | ETIS-Larib | CVC-300 |
|---|---|---|---|---|---|
| SAM-ViT-H + LoRA | 0.906 | 0.869 | 0.785 | 0.745 | 0.887 |
| SAM-ViT-B + LoRA | 0.899 | 0.831 | 0.731 | 0.692 | 0.859 |
| U-Net (ResNet-34) | 0.903 | 0.897 | 0.729 | 0.690 | 0.847 |
| MedSAM-ViT-B + LoRA (min-max norm) | 0.883 | 0.785 | 0.732 | 0.584 | 0.859 |
| MedSAM-ViT-B + LoRA (ImageNet norm, as published) | 0.852 | 0.786 | 0.656 | 0.500 | 0.827 |
| MedSAM-ViT-B + LoRA (ImageNet norm, jitter control) | 0.853 | 0.777 | 0.650 | 0.497 | 0.819 |

The models sit close together on the seen splits and spread apart on the unseen ones. The gap is
widest on ETIS-LaribDB, the hardest unseen split: SAM-ViT-H holds 0.745 there, the two ViT-B SAM
variants land near 0.69, and MedSAM drops to 0.500 under the wrong normalization (0.584 under its
own). Generalization is where the models separate, and it tracks the seen-to-unseen gap column
above.

## Backbone size vs. pretraining source vs. preprocessing

MedSAM scored weakest, but its deficit bundled three causes: a smaller ViT-B backbone, medical
pretraining instead of SAM's generic weights, and — found in review on 2026-07-28 — a real bug:
the pipeline standardized every model's input with ImageNet statistics, which is correct for SAM
by construction (`Sam.preprocess` uses the same constants) but wrong for MedSAM, whose frozen
encoder was fine-tuned on per-image `[0,1]` min-max inputs. Three controlled comparisons now
separate the three causes (see `docs/MEDSAM_INVESTIGATION.md` for the design):

| Comparison | Change | Unseen mDice |
|---|---|---|
| SAM-ViT-H → SAM-ViT-B | backbone shrinks (weights fixed = SAM) | 0.806 → 0.761 (**−0.045**) |
| medsam → medsam_ctrl | jitter off (norm fixed = ImageNet) | 0.661 → 0.655 (**−0.006**) |
| medsam_ctrl → medsam_minmax | normalization corrected (jitter-controlled) | 0.655 → 0.725 (**+0.070**) |
| SAM-ViT-B → MedSAM-minmax | weights change (backbone + preprocessing fair) | 0.761 → 0.725 (**−0.036**) |

The `medsam_ctrl` arm exists because min-max normalization is exactly invariant to
brightness/contrast jitter; comparing `medsam` to `medsam_minmax` directly would also change
effective augmentation strength. The −0.006 shows that confound was negligible; the +0.070 is
the normalization bug's true cost.

**Resolution (2026-08-12):** the earlier "medical pretraining is a net drag and accounts for most
of MedSAM's deficit" reading is retracted. Of the original −0.100 weights-swap penalty, +0.070
was our preprocessing bug. The corrected pretraining effect is −0.036 at equal backbone size —
real, but smaller than the backbone effect (−0.045), not larger. The remaining deficit is
consistent with prompt dependence: MedSAM's fine-tuning always supplied a box prompt, so its
encoder was never trained to localize. The oracle tier quantifies this — a ground-truth box lifts
SAM-ViT-H by +0.099 unseen (0.806 → 0.905) but lifts corrected MedSAM by +0.200 (0.725 → 0.925).

## What the cost buys

U-Net trains fastest (10 min) and produces the smallest checkpoint (98 MB), and it fits the seen
distribution tightest — the highest seen ClinicDB score of any model, 0.897. That fit is also why
it falls furthest on unseen data. SAM-ViT-H + LoRA costs more per run (120 min, a 2.5 GB
checkpoint dominated by the frozen backbone) and updates only 0.83M parameters, yet returns the
best unseen accuracy and the smallest gap. SAM-ViT-B + LoRA sits between them on cost — 29 min,
0.32M trainable parameters, a 349 MB checkpoint — and still generalizes better than U-Net, which
makes it the efficiency pick when ViT-H's checkpoint or runtime is too heavy. Accuracy-per-
trainable-parameter favors the LoRA adapters; accuracy-per-minute favors U-Net. Which one matters
depends on whether the target is the benchmark or the unseen clinic.

## Without fine-tuning (oracle-box baselines)

Vanilla SAM and MedSAM, run with no LoRA and no training at all, prompted with a box derived from
the ground-truth mask. Same backbones as the rows above, zero trainable parameters, a prompt the
trained models never get.

| Model | Seen mDice | Unseen mDice | Gap (seen−unseen) | Trainable params |
|---|---|---|---|---|
| MedSAM ViT-B (vanilla, oracle-box, min-max norm) | 0.912 | **0.925** | −0.012 | 0 |
| SAM ViT-H (vanilla, oracle-box) | 0.860 | 0.905 | −0.046 | 0 |
| SAM ViT-B (vanilla, oracle-box) | 0.821 | 0.880 | −0.060 | 0 |
| MedSAM ViT-B (vanilla, oracle-box, ImageNet norm) | 0.800 | 0.845 | −0.045 | 0 |

Two reads on this tier. First, the preprocessing bug replicates without any training: the same
MedSAM weights score 0.845 unseen under ImageNet normalization and 0.925 under their own min-max
(+0.080). Second, with both of its contracts honored — a box prompt and min-max inputs — vanilla
MedSAM is the strongest model in the study, above vanilla SAM-ViT-H at a fraction of the size.
MedSAM is not a weak model; it is a specialized one, and it pays for that specialization exactly
when the box is taken away.

These stay in a separate tier — never ranked against the trained models. A GT-derived box hands
the model the polyp's location, so the high unseen scores (0.925, 0.905) are an upper bound on how
hard the pixels are to segment once you know where to look, not evidence a zero-shot model beats a
trained one. The **negative seen-to-unseen gap is the tell**: both baselines score higher on
unseen splits than seen ones, which a genuinely generalizing model does not do. It happens because
the oracle box is equally informative on every split, so the numbers track split difficulty rather
than anything the model learned. Read this section as a ceiling check, not a fourth entry in the
accuracy-and-cost table.

## Limitations

- **One benchmark family.** Every split comes from PraNet's own five datasets. These numbers do
  not speak to colonoscopy data collected under different equipment or protocols.
- **Three seeds.** The std columns show run-to-run spread is real; the sample is small.
- **The oracle-box baselines are an upper bound, not a peer.** They see the ground-truth box at
  eval time; the trained models see nothing. Treat the "Without fine-tuning" section as a ceiling
  check, kept in its own tier for that reason.

## Status

Complete over the six trained prompt-free models (including the `medsam_minmax` and `medsam_ctrl`
arms, run 2026-08-12) and the four oracle-box baselines, three seeds each (oracle baselines are
seed-invariant, one run). The runnable, chart-bearing version is `notebooks/07_report.ipynb`,
which does not yet include the two new arms. Left `living` pending professor sign-off on the
oracle-box prompt protocol; mark immutable after that.
