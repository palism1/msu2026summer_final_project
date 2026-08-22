---
status: frozen
date: 2026-07-14
---

# Project Interim Report

**Research Method in Computing, MSU Summer 2026**

**LoRA-Adapted Foundation Models vs. Specialist U-Net for Cross-Dataset Polyp Segmentation**

Mikko Palis, 2026-07-14

## 1. Problem Formulation Refinement

The proposal asked whether a foundation segmentation model could match a specialist polyp
network. Three weeks of reading and the first training runs narrowed that into one measurable
question: can the Segment Anything Model (SAM), adapted with Low-Rank Adaptation (LoRA) while
its backbone stays frozen, match a specialist U-Net on the seen PraNet test splits, and lose
less accuracy on the three datasets neither model trained on?

Three adjustments came out of the literature review.

**The metric moved from benchmark accuracy to the seen-to-unseen drop.** PraNet (Fan et al.,
MICCAI 2020) defined the five-split protocol the field uses: train on 1,450 images from
Kvasir-SEG and CVC-ClinicDB, test on held-out portions of the same two (seen) and on
CVC-ColonDB, ETIS-LaribPolypDB, and CVC-300 (unseen). PraNet's own tables show U-Net falling
from 0.818 Dice on Kvasir to 0.512 on ColonDB. That drop is the deployment problem a clinic
faces when the endoscope, lighting, or patient population changes, so the drop is now the
primary number, and raw benchmark accuracy is secondary.

**MedSAM entered as a third arm, with a condition attached.** MedSAM (Ma et al., Nature
Communications 2024) fine-tuned SAM ViT-B on 1.57M medical image-mask pairs and reports
accuracy above specialist U-Net and DeepLabV3+ baselines, endoscopy included. Reading the
training procedure showed that every training step supplied a bounding-box prompt, and the
specialist baselines received the box as well. The refined problem therefore separates two
contracts: prompt-free inference, which is what a deployed model without a localizer gets,
and box-prompted inference, which is an upper bound. The study runs both, in separate tiers.

**A backbone-size control was added.** SAM ViT-H has 632M encoder parameters; MedSAM is
ViT-B. A direct SAM-ViT-H against MedSAM comparison would confound pretraining source with
backbone size. A SAM-ViT-B + LoRA arm now sits between them so each pair changes one factor.

The research questions as they stand:

- RQ1. Does LoRA-adapted SAM match U-Net on the seen splits?
- RQ2. Does it degrade less on the unseen splits, and by how much?
- RQ3. What does each option cost in trainable parameters, checkpoint size, and training time?
- RQ4. Does medical pretraining (MedSAM) help or hurt under prompt-free adaptation at matched
   backbone size?
- RQ5. How much of each model's deficit is prompt dependence, measured against a
   ground-truth-box upper bound?

## 2. Methodology Implementation

**What is built.** The pipeline is config-driven and runs on Google Colab A100 instances.
`configs/base.yaml` holds the defaults and the per-model blocks; `configs/run.yaml` selects
the model, seed, and output paths. One `train.py` trains any model key under the same
protocol: 352 × 352 inputs, batch size 4, AdamW at learning rate 5e-5, cosine schedule,
Dice + BCE loss, up to 50 epochs with early stopping at patience 8, checkpoint on best
validation Dice. `evaluate.py` scores all five splits with the PraNet metric suite (mDice,
mIoU, weighted F-beta, S-measure, E-measure, MAE). `zeroshot_eval.py` runs the untrained
box-prompted baselines. `aggregate_results.py` consolidates every run's `metrics.json` into
one summary without a GPU.

Four model keys train today: U-Net with a ResNet-34 ImageNet encoder (24.4M trainable
parameters), SAM-ViT-H + LoRA (830K), SAM-ViT-B + LoRA (322K), and MedSAM-ViT-B + LoRA
(322K). The LoRA models follow the SAMed recipe: frozen ViT encoder, rank-4 LoRA with alpha 8
and dropout 0.1 on the encoder attention projections, and a small CNN decoder that reads a
mask with no prompt at inference. Two untrained oracle baselines, vanilla SAM and vanilla
MedSAM prompted with a ground-truth box padded by 5 px, run inference only.

**Challenges and what I did about them.**

*Colab sessions die.* Sessions are preemptible and quota-bound, and a ViT-H run takes two
hours. The trainer now restores finished runs from Google Drive and skips them, so a lost
session costs only the run in flight. Checkpoints follow one path contract,
`checkpoints/<model>/seed<seed>/best.pt`, so the aggregator finds every run mechanically.

*SAM's positional embedding assumes 1024 px input.* At 352 px the encoder crashed. The fix
interpolates `pos_embed` to the training resolution at model init and is recorded as a
do-not-touch decision in `docs/DECISIONS.md`.

*The MedSAM checkpoint download.* The Hugging Face mirror served a truncated file. I switched
to the authors' Zenodo release and added an MD5 check, so a bad checkpoint fails loudly.

*A prompt for the untrained tier.* A zero-shot SAM needs a prompt, and a ground-truth box
hands the model the location. I confirmed the box protocol with my advisor on 2026-06-24 and
keep that tier in a separate table so it never ranks against trained models.

*Testing without a GPU.* The config layer, the metric code, and the prompt-derivation math
are torch-free and covered by unit tests (`pytest tests/`), so the logic that defines the
experiment is checked on my laptop before GPU time is spent.

## 3. Preliminary Results and Analysis

Numbers below are mean Dice over three seeds (42, 43, 44) on one A100, from
`results/summary/`. The four trained arms are prompt-free. The two oracle rows are untrained
and box-prompted, and sit in a separate tier.

| Model | Seen mDice | Unseen mDice | Drop | Trainable params | Train min |
|---|---|---|---|---|---|
| SAM-ViT-H + LoRA | 0.887 | **0.806** | **0.082** | 830,177 | 120 |
| SAM-ViT-B + LoRA | 0.865 | 0.761 | 0.104 | 322,273 | 29 |
| U-Net (ResNet-34) | **0.900** | 0.755 | 0.145 | 24,436,369 | 10 |
| MedSAM-ViT-B + LoRA | 0.819 | 0.661 | 0.158 | 322,273 | 31 |
| Oracle: vanilla SAM ViT-H, GT box | 0.860 | 0.905 | −0.046 | 0 | 0 |
| Oracle: vanilla MedSAM, GT box | 0.800 | 0.845 | −0.045 | 0 | 0 |

**RQ1 and RQ2.** U-Net wins at home (0.900 seen) and SAM-ViT-H + LoRA wins away (0.806
unseen against 0.755). The LoRA drop is about half of U-Net's (0.082 against 0.145). That is
the main hypothesis holding in the first full pass, with 3% of U-Net's trainable parameters.
SAM-ViT-B + LoRA lands level with U-Net on the unseen mean but with a smaller drop, so the
efficiency arm also supports the direction.

**RQ3.** The cost trade is asymmetric. U-Net trains in 10 minutes to a 98 MB checkpoint;
SAM-ViT-H takes 120 minutes and 2.5 GB, almost all of it the frozen backbone. SAM-ViT-B at 29
minutes and 349 MB is the practical middle.

**RQ4, open.** MedSAM + LoRA lands last on every split, 0.661 unseen and 0.500 on ETIS. My
first reading is that medical pretraining hurts under prompt-free adaptation. I do not trust
that reading yet. The deficit against SAM-ViT-B at the same backbone size is 0.100, which is
large for a change in starting weights alone, and the oracle row shows the same MedSAM
weights scoring 0.845 unseen with a box. Something in the pipeline or the protocol may be
wrong for MedSAM specifically, and I plan a root-cause pass before any claim goes in the
final report.

**RQ5.** A ground-truth box lifts vanilla SAM ViT-H to 0.905 unseen, above every trained
prompt-free model. The negative drops in the oracle tier (unseen above seen) say that tier
measures split difficulty given a location hint, not generalization, which is why it stays
separate.

**Significance so far.** The direction of the main result is the one the proposal predicted,
and it is consistent across three seeds. The MedSAM result is the interesting one precisely
because it is suspicious. Either medical pretraining is a net drag without a prompt, which
would be a finding, or my pipeline violates one of MedSAM's input assumptions, which would be
a methods lesson. Both are worth the remaining time.

## 4. Next Steps and Milestones

| Week | Dates | Task | Deliverable |
|---|---|---|---|
| 4 | Jul 14 to Jul 20 | Root-cause pass on MedSAM: rank hypotheses (input normalization, augmentation, checkpoint, decoder) and test the cheapest first | `docs/MEDSAM_INVESTIGATION.md` with ranked hypotheses |
| 5 | Jul 21 to Jul 27 | Run whatever controlled arm the investigation calls for, three seeds; keep the published arm untouched | New model keys and results under separate paths |
| 6 | Jul 28 to Aug 3 | Vanilla MedSAM oracle under its own preprocessing, to bound the ceiling; finish per-split and cost tables | Updated `results/summary/` |
| 7 | Aug 4 to Aug 10 | Findings write-up, figures, decision log frozen | `docs/FINDINGS.md`, `docs/figures/` |
| 8 | Aug 11 to Aug 17 | Final report draft, presentation outline | `docs/PROJECT_REPORT.md` draft |
| 9 | Aug 18 to Aug 22 | Report revision, defense script, submission | Final PDF, slides, speaker script |

**Meeting schedule and communication.** Weekly meetings with the instructor on Wednesdays,
set after the 2026-06-24 meeting where the box-prompt protocol was agreed. Each meeting gets
a one-page status: what ran, what the numbers say, what is blocked, and the one decision I
need. Decisions land in `docs/DECISIONS.md` the same day with a date and a do-not-touch or
tweak tag, so the reasoning survives the session. Between meetings, blockers go by email the
day they appear rather than waiting for the slot.

**Risks.** Colab quota is the main one; the Drive restore-and-skip logic caps the loss from
any one dead session to a single run. The MedSAM investigation could add up to six training
runs (two arms, three seeds), about three A100-hours, which fits the remaining budget.
