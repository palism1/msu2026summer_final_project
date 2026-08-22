---
status: living
last_updated: 2026-08-22
---

# LoRA-Adapted Foundation Models vs. Specialist U-Net for Cross-Dataset Polyp Segmentation

**Project Report, Research Method in Computing**
**MSU Summer 2026, Master's Final Project**
Mikko Palis

---

## 1. Introduction

I wanted to answer one question with this project: can the Segment Anything Model (SAM),
adapted with Low-Rank Adaptation (LoRA) while its backbone stays frozen, match a specialist
U-Net on the standard PraNet benchmark, and lose less accuracy on datasets neither model
trained on?

The question matters because clinical deployment is a cross-dataset problem. A segmentation
model trained at one clinic meets different endoscope hardware, lighting, and patient
populations at the next clinic. Accuracy on the training benchmark does not measure this.
The seen-to-unseen drop does, so that drop is the number I built the study around.

I trained six model variants under one shared protocol, three random seeds each, on one
NVIDIA A100 GPU, and evaluated four untrained oracle-box baselines, for 22 runs in total.
Four findings came out of it:

1. **SAM-ViT-H + LoRA generalizes best.** It reaches 0.806 mean Dice on the three unseen
   datasets against U-Net's 0.755, and its seen-to-unseen drop (0.082) is about half of
   U-Net's (0.145). It trains 830,177 parameters, about 3% of the 24.4M that U-Net trains
   from scratch.
2. **The specialist still wins at home.** U-Net posts the best seen-data score (0.900 mean
   Dice) and trains twelve times faster.
3. **A measured diagnosis of MedSAM's deficit.** MedSAM's apparent collapse (0.661
   unseen) bundled three causes. A controlled three-arm experiment separated them: a
   preprocessing bug in my pipeline cost 0.069 Dice; with backbone and preprocessing
   matched, the MedSAM checkpoint still trails generic SAM by 0.036, comparable in size
   to the 0.045 cost of the smaller backbone. My first reading was that medical
   pretraining is a net drag. I retracted that after the controlled runs.
4. **Prompt dependence quantified.** With its two contracts honored (a box prompt and
   min-max inputs), untrained MedSAM posts the best unseen number in the study (0.925).
   A ground-truth box lifts SAM-ViT-H by +0.099 but lifts corrected MedSAM by +0.200.
   MedSAM is a specialized model, and it pays for the specialization exactly when the
   box is taken away.

The contributions are a controlled comparison of four model families under one protocol,
two matched pairings that separate the pretraining source from the backbone size, a
measured preprocessing effect, a documented root-cause investigation of a preprocessing
confound, and a reproducible config-driven pipeline with all results consolidated in one
summary.

## 2. Literature Review

**Specialist polyp segmentation networks.** U-Net [1] established the encoder-decoder
architecture with skip connections that dominates medical image segmentation. PraNet [2]
defined the evaluation protocol this study adopts: train on 1,450 images from Kvasir-SEG and
CVC-ClinicDB, test on held-out portions of the same two datasets (seen) and on three datasets
the model never saw (CVC-ColonDB, ETIS-LaribPolypDB, CVC-300). PraNet reported that specialist
CNNs lose a lot of accuracy on the unseen splits, which made cross-dataset generalization a
standard metric for this task. Polyp-PVT [3] pushed specialist accuracy further with a
pyramid vision transformer backbone. Both are strong on the benchmark; both are trained
end-to-end on a few thousand polyp images, which bounds how much visual diversity they can
absorb.

**Segmentation foundation models.** SAM [4] trained a ViT image encoder and a promptable mask
decoder on SA-1B, a corpus of 11M images and 1.1B masks. Zero-shot SAM segments natural images
well but underperforms on medical imagery, where object boundaries follow tissue contrast
rather than natural-image edges. MedSAM [5] fine-tuned SAM ViT-B on about 1.57M medical
image-mask pairs across ten modalities and reported specialist-level accuracy on endoscopy,
with one structural condition: every training step supplied a bounding-box prompt. The model
was never asked to localize an object on its own.

**Parameter-efficient fine-tuning.** LoRA [6] freezes a pretrained weight matrix and learns a
low-rank update, which cuts trainable parameters by orders of magnitude while it keeps the
pretrained representation intact. SAMed [7] applied LoRA to a frozen SAM encoder for medical
segmentation and is the method this project adapts: LoRA on the attention projections, a
lightweight convolutional decoder, no prompts at inference.

**The unanswered question.** The surveyed work reports benchmark accuracy. None of it says
whether LoRA adaptation of a foundation model buys *generalization*, measured on the same
seen/unseen protocol the specialists were built on, under one training protocol, with the
cost of each option recorded. Two published claims also sit in tension: MedSAM reports
beating specialists on endoscopy, and the PraNet protocol is exactly the setting where that
claim can be tested without a prompt oracle. Nobody had run that test. This study does.

## 3. Problem Formulation

**Research problem.** Specialist polyp networks fit their training distribution tightly and
lose accuracy on data from other clinics. Foundation models carry broad visual priors but are
expensive to fine-tune fully and, zero-shot, underperform on medical imagery. The problem is
to find out whether parameter-efficient adaptation keeps the foundation model's
generalization at a cost a specialist would accept, and to trace any difference to its
actual cause rather than to a confound.

**Research questions.**

- **RQ1.** Does LoRA-adapted SAM match a specialist U-Net on the seen PraNet test splits?
- **RQ2.** Does it degrade less on the three unseen splits, and by how much?
- **RQ3.** What does each option cost in trainable parameters, checkpoint size, and training
  time on identical hardware?
- **RQ4.** Does medical pretraining (MedSAM) help or hurt under prompt-free adaptation, once
  backbone size and preprocessing are controlled?
- **RQ5.** How much of each model's deficit is prompt dependence, measured against an
  oracle-box upper bound?

**Hypotheses.** H-main: the frozen SAM backbone's broad pretraining transfers, so the LoRA
variant shows a smaller seen-to-unseen drop than U-Net. H-medical: medical pretraining gives
MedSAM an advantage over generic SAM at equal backbone size. The data did not support the
second hypothesis. Section 6 reports the measured comparisons, including the part of the
deficit my own pipeline caused.

**Significance.** If a frozen foundation backbone plus a sub-1M-parameter adapter generalizes
better than a specialist, then sites with modest data and compute can prefer adapters over
from-scratch training. The MedSAM analysis also shows how a preprocessing contract violation can look like a
scientific finding. That is a methods lesson that reaches past this task.

## 4. Methodology

### 4.1 Datasets and protocol

The study follows the PraNet five-split protocol exactly.

| Dataset | Role | Images |
|---|---|---|
| Kvasir-SEG | 900 train / 100 seen test | 1,000 |
| CVC-ClinicDB | 550 train / 62 seen test | 612 |
| CVC-ColonDB | unseen test | 380 |
| ETIS-LaribPolypDB | unseen test | 196 |
| CVC-300 (EndoScene) | unseen test | 60 |

Training uses 1,450 images. The two seen test splits measure learning ability; the three
unseen splits measure generalization. The primary derived metric is the gap: mean seen mDice
minus mean unseen mDice.

### 4.2 Models

| Model | Backbone | Trainable params | Role |
|---|---|---|---|
| U-Net | ResNet-34 encoder (ImageNet init) | 24,436,369 | specialist baseline |
| SAM-ViT-H + LoRA | frozen SAM ViT-H | 830,177 | main contribution |
| SAM-ViT-B + LoRA | frozen SAM ViT-B | 322,273 | backbone-size control |
| MedSAM-ViT-B + LoRA | frozen MedSAM ViT-B | 322,273 | pretraining-source arm |
| MedSAM min-max / ctrl | frozen MedSAM ViT-B | 322,273 | normalization experiment arms |
| Vanilla SAM / MedSAM (4 variants) | frozen, untrained | 0 | oracle-box upper bound |

The LoRA models follow SAMed: the ViT encoder is frozen, rank-4 LoRA (alpha 8, dropout 0.1)
is injected into the attention query and value projections, and a lightweight CNN decoder
reads a mask from the encoder features with no prompt at inference. SAM-ViT-B + LoRA shares
MedSAM's backbone size but keeps generic SAM weights, so the pair separates backbone capacity
from pretraining source. Two caveats bound these pairings, and I want them stated before the results rather
than after. The LoRA parameter count scales with the backbone (830K on ViT-H against 322K on
ViT-B), so the size comparison moves adapter capacity together with backbone capacity. And
MedSAM's starting weights bundle two changes its authors made together, medical training data
and a box prompt supplied on every training example, so the pretraining pair measures the
checkpoint as delivered, not the medical data alone.

### 4.3 Training protocol

All trained models share one protocol: 352×352 inputs, batch size 4, AdamW (learning rate
5e-5, weight decay 1e-4), cosine schedule, combined Dice + BCE loss, up to 50 epochs with
early stopping at patience 8, checkpoint selection by best validation Dice, three seeds
(42, 43, 44), one NVIDIA A100-SXM4-40GB. Augmentation: flips, rotation, and color jitter via
Albumentations. Input normalization is bound to the model key: ImageNet standardization for
SAM and U-Net (this reproduces SAM's native `preprocess`), per-image min-max [0,1] for the
corrected MedSAM arms (MedSAM's native contract).

### 4.4 Evaluation

All five splits are scored with the PraNet metric suite: mDice and mIoU (primary), weighted
F-beta, S-measure, E-measure, and MAE. Reported numbers are means with standard deviation
over the three seeds. Cost metrics (trainable parameters, checkpoint size, wall-clock training
time, device name) are recorded in each run's `metrics.json`.

### 4.5 Oracle-box baselines

Four untrained baselines run inference only, prompted with a bounding box derived from the
ground-truth mask (5 px padding): vanilla SAM ViT-H, vanilla SAM ViT-B, vanilla MedSAM under
ImageNet normalization, and vanilla MedSAM under its own min-max normalization. The
ground-truth box reveals the polyp's location, so these numbers form an upper bound. I keep
them in a separate tier and never rank them against the trained models. The box protocol is
standard in the medical-SAM literature, and I confirmed it with my professor before the
final runs.

### 4.6 The MedSAM three-arm experiment

A review on 2026-07-28 found that my pipeline fed every model ImageNet-standardized inputs.
That is correct for SAM by construction and wrong for MedSAM, whose frozen encoder was
fine-tuned on per-image min-max [0,1] inputs. The obvious fix was a re-run under min-max. I
did not take it, because it would have changed two things at once: min-max normalization is
exactly invariant to brightness and contrast jitter, so correcting the normalization also
weakens the effective augmentation. A one-arm fix would have left me unable to say which
change moved the number. The design therefore uses three arms:

| Arm | Normalization | Brightness/contrast jitter | Role |
|---|---|---|---|
| `medsam` | ImageNet | on | published baseline, never re-run |
| `medsam_ctrl` | ImageNet | off | augmentation control |
| `medsam_minmax` | min-max | on (mostly cancelled by min-max) | corrected arm |

The controlled normalization effect is `medsam_ctrl` → `medsam_minmax`; the size of the
augmentation confound is `medsam` → `medsam_ctrl`.

## 5. Implementation Details

### 5.1 Architecture of the codebase

The pipeline is config-driven. `configs/base.yaml` holds shared hyperparameters;
`configs/run.yaml` selects the model, seed, and output paths. `train.py` trains any model key
under the identical protocol; `evaluate.py` scores all five splits; `zeroshot_eval.py` runs
the untrained oracle-box baselines; `aggregate_results.py` consolidates every run's
`metrics.json` into `results/summary/` without a GPU. Model construction lives in
`src/models/` (U-Net via segmentation-models-pytorch; `sam_adapter.py` for the LoRA wrapper
and light decoder; `zeroshot.py` for the oracle baselines). Metrics live in
`src/metrics/segmentation.py`. Training ran on Google Colab A100 instances through
`notebooks/train_colab.ipynb`, which mirrors checkpoints and results to Google Drive and
skips already-trained pairs.

Two implementation decisions matter more than the rest. First, the training protocol
(normalization and photometric augmentation) is bound to the model key in
`src/config.MODEL_SPECS` and cannot be set in YAML. I chose this after the normalization bug:
a protocol mismatch now requires a code change with review instead of a quiet config edit.
Second, checkpoints follow one path contract (`checkpoints/<model>/seed<seed>/best.pt`), which
lets the aggregator and the benchmark notebook find every run without hand-listing them.

### 5.2 Challenges and how they were addressed

**The MedSAM normalization bug.** The largest challenge was diagnostic. MedSAM + LoRA landed
last on every split (0.661 unseen, 0.500 on ETIS-LaribPolypDB), and my first write-up blamed
medical pretraining. That was the wrong call, and the honest reason is that I had not checked
the input contract. A root-cause investigation (`docs/MEDSAM_INVESTIGATION.md`) ranked four
hypotheses and found that the pipeline's single global ImageNet normalization violated
MedSAM's input contract: its encoder was fit on [0,1] inputs and received inputs in roughly
[−2.1, +2.6]. The fix bound normalization to the model key. The confound analysis then showed
the naive fix would itself be confounded (min-max absorbs brightness/contrast jitter), which
produced the three-arm design in Section 4.6. I kept the published `medsam` arm and never
re-ran it; the corrected arms got separate model keys, checkpoints, and results paths, so the
retraction and the correction both stay visible in the results.

**Prompt protocol for the untrained tier.** A zero-shot SAM needs a prompt. A ground-truth
box is an oracle: it hands the model the location. The tradeoff was between dropping the
untrained tier, which loses the upper bound, and reporting it next to trained models, which
invites a false comparison. I took a third route: keep the oracle tier in a separate table,
flag the negative seen-to-unseen gap as the sign that these numbers track split difficulty
rather than learning, and get my professor's sign-off on the box protocol before the final
runs.

**Compute discipline.** Colab sessions are preemptible and quota-bound. The trainer restores
finished runs from Drive and skips them, so a lost session costs only the run in flight.
Early stopping kept several 50-epoch budgets to 28 to 49 actual epochs.

**Testing without a GPU.** Prompt-derivation math and the config/aggregation layer are
torch-free and covered by 24 unit tests (`pytest tests/`), so I could check the logic that
defines the experiment on my laptop before spending GPU time.

## 6. Results and Analysis

### 6.1 Accuracy and cost (trained, prompt-free tier)

Means ± standard deviation over seeds 42, 43, 44; one A100.

| Model | Seen mDice | Unseen mDice | Gap | Trainable params | Ckpt MB | Train min |
|---|---|---|---|---|---|---|
| SAM-ViT-H + LoRA | 0.887 ± 0.008 | **0.806 ± 0.011** | **0.082 ± 0.004** | 830,177 | 2,533 | 120 |
| SAM-ViT-B + LoRA | 0.865 ± 0.005 | 0.761 ± 0.005 | 0.104 ± 0.007 | 322,273 | 349 | 29 |
| U-Net (ResNet-34) | **0.900 ± 0.008** | 0.755 ± 0.014 | 0.145 ± 0.014 | 24,436,369 | 98 | 10 |
| MedSAM + LoRA (min-max norm) | 0.834 ± 0.001 | 0.725 ± 0.004 | 0.110 ± 0.004 | 322,273 | 349 | 30 |
| MedSAM + LoRA (ImageNet norm, as published) | 0.819 ± 0.006 | 0.661 ± 0.008 | 0.158 ± 0.011 | 322,273 | 349 | 31 |
| MedSAM + LoRA (ImageNet norm, jitter control) | 0.815 ± 0.001 | 0.655 ± 0.002 | 0.160 ± 0.002 | 322,273 | 349 | 30 |

**RQ1 (seen).** U-Net wins at home: 0.900 against SAM-ViT-H's 0.887. The specialist fits the
training distribution tightest, including the best ClinicDB score of any model (0.897).

**RQ2 (unseen).** SAM-ViT-H + LoRA leads every unseen split and holds 0.806 mean unseen Dice
against U-Net's 0.755. Its seen-to-unseen drop is 0.082, about half of U-Net's 0.145. Even
SAM-ViT-B + LoRA (0.761 unseen) edges past U-Net while it trains 1.3% of U-Net's parameters.
The separation is widest on ETIS-LaribPolypDB, the hardest unseen split: SAM-ViT-H 0.745,
SAM-ViT-B 0.692, U-Net 0.690.

**RQ3 (cost).** U-Net trains fastest (10 min) with the smallest checkpoint (98 MB).
SAM-ViT-H + LoRA costs 120 min and a 2.5 GB checkpoint, dominated by the frozen backbone.
SAM-ViT-B + LoRA is the efficiency pick: 29 min, 349 MB, and it still generalizes better
than U-Net. Accuracy per trainable parameter favors the adapters; accuracy per minute favors
U-Net. Which one matters depends on whether the target is the benchmark or the next clinic.
The whole study (18 trained runs) took about 750 GPU-minutes, or 12.5 A100-hours. At Colab
rates that is roughly 15 dollars of compute. The GPU-hours are measured; the dollar figure is
an estimate from the published rate.

### 6.2 Per-split behavior (mean over 3 seeds)

| Model | Kvasir (seen) | ClinicDB (seen) | ColonDB | ETIS | CVC-300 |
|---|---|---|---|---|---|
| SAM-ViT-H + LoRA | 0.906 | 0.869 | 0.785 | 0.745 | 0.887 |
| SAM-ViT-B + LoRA | 0.899 | 0.831 | 0.731 | 0.692 | 0.859 |
| U-Net (ResNet-34) | 0.903 | 0.897 | 0.729 | 0.690 | 0.847 |
| MedSAM + LoRA (min-max) | 0.883 | 0.785 | 0.732 | 0.584 | 0.859 |
| MedSAM + LoRA (as published) | 0.852 | 0.786 | 0.656 | 0.500 | 0.827 |
| MedSAM + LoRA (jitter control) | 0.853 | 0.777 | 0.650 | 0.497 | 0.819 |

The models sit close together on the seen splits and spread apart on the unseen ones.
They separate on generalization. Learning ability is roughly a tie.

### 6.3 The MedSAM diagnosis: matched pairings (RQ4)

Four controlled comparisons, each changing one factor:

| Comparison | Factor changed | Unseen mDice effect |
|---|---|---|
| SAM-ViT-H → SAM-ViT-B | backbone shrinks (SAM weights fixed) | 0.806 → 0.761 (**−0.045**) |
| medsam → medsam_ctrl | jitter off (ImageNet norm fixed) | 0.661 → 0.655 (**−0.006**) |
| medsam_ctrl → medsam_minmax | normalization corrected (jitter-controlled) | 0.655 → 0.725 (**+0.069**) |
| SAM-ViT-B → MedSAM-minmax | pretraining source (backbone and preprocessing matched) | 0.761 → 0.725 (**−0.036**) |

Of MedSAM's original 0.100 unseen deficit against SAM-ViT-B, 0.069 was my preprocessing bug
and 0.036 is the residual effect of the MedSAM checkpoint at matched size. That residual and
the backbone effect (0.045) are close, and each carries a seed spread near 0.005, so the two
read as comparable in size rather than strictly ordered. The checkpoint effect itself bundles
MedSAM's medical training data with its box-prompted training objective; Section 6.4 points to
the objective as the likely driver. The jitter confound was negligible (0.006), which validates
the three-arm design. On this evidence I retracted the earlier claim that medical pretraining is a net drag and
accounts for most of the deficit.

### 6.4 Oracle-box upper bound (RQ5)

Untrained, prompted with a ground-truth-derived box; a separate tier, not a fair peer.

| Model | Seen mDice | Unseen mDice | Gap |
|---|---|---|---|
| MedSAM ViT-B (oracle-box, min-max norm) | 0.912 | **0.925** | −0.012 |
| SAM ViT-H (oracle-box) | 0.860 | 0.905 | −0.046 |
| SAM ViT-B (oracle-box) | 0.821 | 0.880 | −0.060 |
| MedSAM ViT-B (oracle-box, ImageNet norm) | 0.800 | 0.845 | −0.045 |

I read two things from this table. First, the normalization bug replicates with zero
training: the same MedSAM
weights score 0.845 unseen under ImageNet inputs and 0.925 under their own min-max (+0.080).
Second, with both of its contracts honored, vanilla MedSAM is the strongest model in the
study. The negative gaps show that this tier measures split difficulty given a
location hint, not generalization: a genuinely generalizing model should not score higher on
unseen data than seen data. This is the prompt dependence in numbers: the untrained box-prompted model
beats its own trained prompt-free version by +0.099 for SAM-ViT-H (0.806 → 0.905) and by
+0.200 for corrected MedSAM (0.725 → 0.925). Two conditions change between those numbers, the
box arrives and the LoRA training is absent, so each lift reads as the value of location
information net of what training added. MedSAM's fine-tuning always supplied a box, so its
encoder never learned to localize; it pays exactly when the box is taken away.

## 7. Discussion

**The adaptation trade is real and favorable off-distribution.** The frozen SAM backbone
carries visual priors that 1,450 polyp images cannot teach a from-scratch CNN, and rank-4
LoRA is enough to point those priors at the task. The result pattern (specialist wins at
home, adapter wins away) matches the deployment question clinics actually face, and the
gap metric puts the trade in one line: 3% of the trainable parameters buys half the
generalization drop.

**MedSAM is specialized, not weak.** The finding I learned the most from is negative
in form. MedSAM's published claim (specialist-level endoscopy accuracy) and my measured
result (last on every split) are both correct. They measure different contracts. MedSAM
with a box and min-max inputs posts the best unseen number here (0.925). MedSAM without
the box loses the most, because its fine-tuning let the encoder offload localization to
the prompt. A practitioner choosing a model should ask which contract the deployment honors. If no
localizer supplies a box at inference, I would pick prompt-free adaptation of generic SAM
over medical-pretrained MedSAM at the same size; the 0.036 residual in Section 6.3 is the
reason.

**Methods lesson: preprocessing contracts are results.** A one-line normalization
difference produced a 0.069 Dice artifact that I first read as a scientific finding
about medical pretraining. The correction needed a controlled experiment, because the
naive fix was itself confounded through the interaction between min-max normalization
and photometric augmentation. The pattern I would reuse is to bind preprocessing to the
model key in code, keep the published arm, and give the corrected arms separate keys.

**Relation to prior work.** The result extends SAMed's finding (LoRA on frozen SAM works
for medical segmentation) with the measurement SAMed did not make: cross-dataset
generalization under the specialists' own protocol. It also gives PraNet's seen/unseen
framing a new use, separating model classes rather than ranking specialist variants.

## 8. Conclusion

A frozen SAM ViT-H backbone with an 830K-parameter LoRA adapter beats a 24.4M-parameter
specialist U-Net on unseen polyp datasets (0.806 vs 0.755 mean Dice) and halves the
seen-to-unseen drop (0.082 vs 0.145), while the specialist keeps the edge on seen data
and in training cost. Controlled ablations attribute MedSAM's apparent collapse to three
measured causes: a 0.069 preprocessing artifact in my own pipeline, a 0.036 residual for
the MedSAM checkpoint at matched size, and a 0.045 backbone-size effect, the last two
comparable in size, and quantify its prompt dependence (+0.200 from an oracle box against
+0.099 for generic SAM).

**Contributions.** (1) A controlled four-family comparison (specialist CNN, LoRA-adapted
generic SAM at two sizes, LoRA-adapted MedSAM) under one protocol, three seeds, one GPU,
with full cost accounting. (2) Two controlled pairings that separate the pretraining source
from the backbone size, plus a measured preprocessing effect, via matched arms. (3) A documented root-cause investigation and
retraction pattern for a preprocessing confound. (4) A reproducible pipeline: config-driven
trainer, GPU-free tests, mechanical results aggregation.

**Limitations.** I want to be direct about what these numbers do not cover. All five splits
come from PraNet's datasets, so the results do not speak to colonoscopy data under other
equipment or protocols. Three seeds show the spread is real, but the sample is small. The
oracle tier has one run per baseline and remains an upper bound, not a peer. Training cost
is asymmetric in a way that matters to some deployments (U-Net: 10 min, 98 MB; SAM-H: 120
min, 2.5 GB).

**Future work.** Four directions, in the order I would take them. Evaluate on a benchmark
family outside PraNet's five datasets, since that is the limitation that bounds the claim
most. Pair MedSAM with a learned box proposer to test whether its 0.925 oracle ceiling is
reachable without ground truth; the cascade is cheap because both halves already exist.
Probe color sensitivity with the grayscale-replication experiment already designed. Test
larger LoRA ranks and target modules.

## 9. References

[1] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical
image segmentation," in *Proc. MICCAI*, 2015, pp. 234–241.

[2] D.-P. Fan, G.-P. Ji, T. Zhou, G. Chen, H. Fu, J. Shen, and L. Shao, "PraNet: Parallel
reverse attention network for polyp segmentation," in *Proc. MICCAI*, 2020, pp. 263–273.

[3] B. Dong, W. Wang, D.-P. Fan, J. Li, H. Fu, and L. Shao, "Polyp-PVT: Polyp segmentation
with pyramid vision transformers," *CAAI Artificial Intelligence Research*, vol. 2, 2023.

[4] A. Kirillov et al., "Segment Anything," in *Proc. IEEE/CVF ICCV*, 2023, pp. 4015–4026.

[5] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, "Segment anything in medical images,"
*Nature Communications*, vol. 15, no. 654, 2024.

[6] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen,
"LoRA: Low-rank adaptation of large language models," in *Proc. ICLR*, 2022.

[7] K. Zhang and D. Liu, "Customized Segment Anything Model for medical image
segmentation," arXiv:2304.13785, 2023.

[8] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, and
H. D. Johansen, "Kvasir-SEG: A segmented polyp dataset," in *Proc. MMM*, 2020, pp. 451–462.

[9] J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, D. Gil, C. Rodríguez, and F. Vilariño,
"WM-DOVA maps for accurate polyp highlighting in colonoscopy," *Computerized Medical Imaging
and Graphics*, vol. 43, pp. 99–111, 2015. (CVC-ClinicDB)

[10] J. Bernal, J. Sánchez, and F. Vilariño, "Towards automatic polyp detection with a polyp
appearance model," *Pattern Recognition*, vol. 45, no. 9, pp. 3166–3182, 2012. (CVC-ColonDB)

[11] J. Silva, A. Histace, O. Romain, X. Dray, and B. Granado, "Toward embedded detection of
polyps in WCE images for early diagnosis of colorectal cancer," *Int. J. Comput. Assist.
Radiol. Surg.*, vol. 9, no. 2, pp. 283–293, 2014. (ETIS-LaribPolypDB)

[12] D. Vázquez et al., "A benchmark for endoluminal scene segmentation of colonoscopy
images," *Journal of Healthcare Engineering*, vol. 2017. (CVC-300 / EndoScene)

[13] D.-P. Fan, M.-M. Cheng, Y. Liu, T. Li, and A. Borji, "Structure-measure: A new way to
evaluate foreground maps," in *Proc. IEEE ICCV*, 2017, pp. 4548–4557.

[14] D.-P. Fan, C. Gong, Y. Cao, B. Ren, M.-M. Cheng, and A. Borji, "Enhanced-alignment
measure for binary foreground map evaluation," in *Proc. IJCAI*, 2018, pp. 698–704.

## 10. Appendices

### Appendix A: Hyperparameters

| Setting | Value |
|---|---|
| Input size | 352 × 352 |
| Batch size | 4 |
| Optimizer | AdamW, lr 5e-5, weight decay 1e-4 |
| Schedule | cosine |
| Loss | Dice + BCE |
| Epoch budget | 50, early stop patience 8 |
| Checkpoint selection | best validation Dice |
| LoRA | r = 4, alpha = 8, dropout 0.1, query and value projections |
| Seeds | 42, 43, 44 |
| Hardware | NVIDIA A100-SXM4-40GB (Google Colab) |
| Oracle prompt | ground-truth bounding box, 5 px padding |
| Normalization | ImageNet stats (SAM, U-Net); per-image min-max [0,1] (corrected MedSAM arms) |

### Appendix B: Repository guide

Source code: GitHub repository (this project). Entry points: `train.py` (training),
`evaluate.py` (5-split evaluation), `zeroshot_eval.py` (oracle-box baselines),
`aggregate_results.py` (results consolidation). Full per-run and per-seed numbers:
`results/summary/SUMMARY.md`, `summary_flat.csv`, `summary_by_model.csv`. Supporting
documents: `docs/FINDINGS.md` (findings write-up), `docs/MEDSAM_INVESTIGATION.md`
(hypothesis-driven root-cause analysis), `docs/DECISIONS.md` (design decision log),
`docs/PRESENTATION_OUTLINE.md` (talk plan). Tests: `pytest tests/` (24 GPU-free tests).

### Appendix C: Per-run results

The complete per-run table (each model × seed, all five splits, parameters, checkpoint
size, epochs, minutes, device) is generated by `aggregate_results.py` and lives at
`results/summary/SUMMARY.md`. Qualitative mask overlays for every run live under
`results/<model>/seed<seed>/`.
