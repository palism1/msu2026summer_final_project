---
status: living
last_updated: 2026-07-21
---

<!-- Accuracy-vs-cost findings for the cross-dataset polyp-segmentation study. Doc class: living.
     Covers four trained prompt-free models over three seeds plus two oracle-box baselines. Numbers
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
1.3% of its trainable parameters, and MedSAM's weakness traces mostly to its medical pretraining
rather than its smaller backbone.

All numbers below are means over three seeds (42, 43, 44), one benchmark family (PraNet), one
A100. Seen splits: Kvasir, ClinicDB. Unseen splits: CVC-ColonDB, ETIS-LaribDB, CVC-300.

## Accuracy and cost by model

| Model | Seen mDice | Unseen mDice | Gap (seen−unseen) | Trainable params | Ckpt MB | Train min |
|---|---|---|---|---|---|---|
| SAM-ViT-H + LoRA | 0.887 ± 0.008 | **0.806 ± 0.011** | **0.082 ± 0.004** | 830,177 | 2533 | 120 |
| SAM-ViT-B + LoRA | 0.865 ± 0.005 | 0.761 ± 0.005 | 0.104 ± 0.007 | 322,273 | 349 | 28 |
| U-Net (ResNet-34) | **0.900 ± 0.008** | 0.755 ± 0.014 | 0.145 ± 0.014 | 24,436,369 | 98 | 10 |
| MedSAM-ViT-B + LoRA | 0.819 ± 0.006 | 0.661 ± 0.008 | 0.158 ± 0.011 | 322,273 | 349 | 31 |

## Per-split behavior

| Model | Kvasir (seen) | ClinicDB (seen) | CVC-ColonDB | ETIS-Larib | CVC-300 |
|---|---|---|---|---|---|
| SAM-ViT-H + LoRA | 0.906 | 0.869 | 0.785 | 0.745 | 0.887 |
| SAM-ViT-B + LoRA | 0.899 | 0.831 | 0.731 | 0.692 | 0.859 |
| U-Net (ResNet-34) | 0.903 | 0.897 | 0.729 | 0.690 | 0.847 |
| MedSAM-ViT-B + LoRA | 0.852 | 0.786 | 0.656 | 0.500 | 0.827 |

The models sit close together on the seen splits and spread apart on the unseen ones. The gap is
widest on ETIS-LaribDB, the hardest unseen split: SAM-ViT-H holds 0.745 there, the two ViT-B SAM
variants land near 0.69, and MedSAM drops to 0.500. Generalization is where the models separate,
and it tracks the seen-to-unseen gap column above.

## Backbone size vs. pretraining source

SAM-ViT-B + LoRA settles the MedSAM confound. MedSAM scored weakest, but it carries two
disadvantages against SAM-ViT-H at once: a smaller ViT-B backbone and medical-image pretraining
instead of SAM's generic weights. SAM-ViT-B + LoRA holds one fixed and varies the other — same
ViT-B backbone and LoRA recipe as MedSAM, but SAM's generic weights — so the two comparisons
below each move a single variable:

| Comparison | Change | Unseen mDice |
|---|---|---|
| SAM-ViT-H → SAM-ViT-B | backbone shrinks (weights fixed = SAM) | 0.806 → 0.761 (**−0.045**) |
| SAM-ViT-B → MedSAM-ViT-B | weights change (backbone fixed = ViT-B) | 0.761 → 0.661 (**−0.100**) |

Shrinking the backbone from ViT-H to ViT-B costs 0.045 unseen mDice. Swapping SAM's generic
weights for MedSAM's medical weights at the same backbone size costs 0.100 — more than twice as
much. On this polyp benchmark under LoRA, MedSAM's medical pretraining is a net drag relative to
generic SAM weights, and it accounts for most of MedSAM's deficit. Backbone capacity matters, but
the pretraining source matters more.

## What the cost buys

U-Net trains fastest (10 min) and produces the smallest checkpoint (98 MB), and it fits the seen
distribution tightest — the highest seen ClinicDB score of any model, 0.897. That fit is also why
it falls furthest on unseen data. SAM-ViT-H + LoRA costs more per run (120 min, a 2.5 GB
checkpoint dominated by the frozen backbone) and updates only 0.83M parameters, yet returns the
best unseen accuracy and the smallest gap. SAM-ViT-B + LoRA sits between them on cost — 28 min,
0.32M trainable parameters, a 349 MB checkpoint — and still generalizes better than U-Net, which
makes it the efficiency pick when ViT-H's checkpoint or runtime is too heavy. Accuracy-per-
trainable-parameter favors the LoRA adapters; accuracy-per-minute favors U-Net. Which one matters
depends on whether the target is the benchmark or the unseen clinic.

## Without fine-tuning (oracle-box baselines)

Vanilla SAM-ViT-H and vanilla MedSAM-ViT-B, run with no LoRA and no training at all, prompted with
a box derived from the ground-truth mask. Same backbones as the rows above, zero trainable
parameters, a prompt the trained models never get.

| Model | Seen mDice | Unseen mDice | Gap (seen−unseen) | Trainable params |
|---|---|---|---|---|
| SAM ViT-H (vanilla, oracle-box) | 0.860 | 0.905 | −0.046 | 0 |
| MedSAM ViT-B (vanilla, oracle-box) | 0.800 | 0.845 | −0.045 | 0 |

These stay in a separate tier — never ranked against the trained models. A GT-derived box hands
the model the polyp's location, so the high unseen scores (0.905, 0.845) are an upper bound on how
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

Complete over the four trained prompt-free models and the two oracle-box baselines, three seeds
each (oracle baselines are seed-invariant, one run). The runnable, chart-bearing version is
`notebooks/07_report.ipynb`. Left `living` pending professor sign-off on the oracle-box prompt
protocol; mark immutable after that.
