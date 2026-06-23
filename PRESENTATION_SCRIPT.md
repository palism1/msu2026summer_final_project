# Professor Meeting Script — June 24, 2026
## Can LoRA-Adapted SAM Match Specialist Polyp Networks?

---

## OPENING (1–2 min)

"The project is a cross-dataset generalization study in colonoscopy segmentation.
The core question is: can we take SAM — a general-purpose vision foundation model —
adapt it with a small number of extra parameters using LoRA, and have it match or
beat fully-trained specialist networks, specifically on datasets it has never seen?

Generalization is the real clinical problem. A model trained at one hospital on one
camera system needs to work at another hospital with a different scope. Specialist
networks like U-Net tend to overfit to their training distribution. We want to know
if SAM's broad pretraining gives it an edge there.

Tonight I have Phase 1 (data pipeline) and Phase 2 (U-Net baseline) fully running.
Let me walk you through both."

---

## NOTEBOOK 1 — DATA PIPELINE [~8 min]

### Setup [SHOW: cell 1 output — "Running in Colab: True"]

"Everything runs in Google Colab directly from the GitHub repo. The setup cell
clones the repo and installs dependencies automatically — reproducible from scratch."

---

### Datasets [SHOW: cell 2 output — download progress bars]

"We follow the PraNet evaluation protocol, which is the standard benchmark for this
problem. That means two Google Drive packages:

- TrainDataset: 1,450 images — 900 from Kvasir-SEG and 550 from CVC-ClinicDB,
  packaged flat with no per-dataset separation
- TestDataset: 5 subfolders covering both seen and unseen splits

One thing I had to figure out early: the zip extracts with a nested folder structure
that isn't documented anywhere. I wrote a resolver for that."

---

### Split verification [SHOW: verification table — all ✓]

"The split sizes match exactly what the PraNet paper specifies:
- Train: 1,450
- Seen test: 100 Kvasir + 62 CVC-ClinicDB
- Unseen test: 380 CVC-ColonDB + 196 ETIS-Larib + 60 CVC-300

The seen/unseen distinction is the whole thesis. We train on Kvasir and ClinicDB,
evaluate on all five, and measure how much performance drops on the three datasets
the model has never seen."

---

### Overlap check [SHOW: MD5 hash output — all ✓ No content overlap]

"This was an interesting bug. A filename-based check gave false positives — CVC-ColonDB
was flagging 343 files as overlapping with training. The reason: every dataset
independently numbers its files starting from 1.jpg. Kvasir's 1.jpg and CVC-ColonDB's
1.jpg are completely different images from different hospitals.

I switched to MD5 content hashing — comparing actual byte content. All five test splits
are confirmed clean. No training images leaked into the test sets."

---

### Visualizations [SHOW: training samples + CVC-ColonDB samples]

"You can see the domain shift visually. The training images are from Kvasir and ClinicDB —
bright, clear colonoscopy frames. CVC-ColonDB on the right uses a different endoscope
with a circular field of view and darker, lower-contrast imaging. The model has to
generalize across these imaging differences."

---

### Augmentation [SHOW: augmentation grid]

"The augmentation pipeline: horizontal/vertical flips, up to 90-degree rotation,
random scale ±20%, and color jitter. All spatial transforms happen before the final
resize to 352×352, so the output is always a fixed size regardless of what the
augmentations do.

The color jitter is important — it helps the model become invariant to the lighting
and camera differences between hospitals."

---

### Metrics [SHOW: smoke test table]

"Six metrics, all from the PraNet evaluation protocol:
- Dice and IoU: standard overlap metrics, primary results
- MAE: pixel-level absolute error on the probability map
- Weighted F-measure: edge-aware, weights boundary pixels more heavily
- S-measure: structural similarity, looks at object shape and region statistics
- E-measure: enhanced alignment between prediction and ground truth

The smoke test confirms they all behave correctly — perfect prediction scores 1.0
on all five quality metrics."

---

### Polyp size distribution [SHOW: histogram]

"Mean polyp coverage is 13.1% of the image, with a strong right skew — most polyps
are small. This is clinically realistic and also explains why segmentation is hard.
Small polyps have a high boundary-to-area ratio, so boundary errors hurt the metrics
disproportionately."

---

## NOTEBOOK 2 — U-NET BASELINE [~6 min]

### Architecture [SHOW: model parameter output]

"The baseline is U-Net with a ResNet-34 encoder pretrained on ImageNet.
24.4 million parameters, all trainable. This is the standard specialist architecture
for medical image segmentation — fully supervised, no foundation model involved.
It's the number SAM-LoRA needs to beat."

---

### Training [SHOW: training log first few epochs + last few epochs]

"Loss function: Dice + BCE combined. Optimizer: AdamW with cosine learning rate
schedule. Early stopping with patience 10.

The model converges fast — Dice is already 0.88 by epoch 7. After that it's
incremental improvements. Early stopping triggered at epoch 75."

---

### Training curves [SHOW: loss and validation metrics plots]

"Clean convergence. Loss drops from 1.1 to 0.07. Validation Dice plateaus around
0.91 after epoch 50. No signs of overfitting — the validation curve tracks the
training curve the whole way."

---

### Evaluation table [SHOW: 5-split results table]

"This is the key result for this phase:

  Seen — Kvasir:      0.9109 Dice
  Seen — CVC-ClinicDB: 0.9141 Dice
  Unseen — CVC-ColonDB: 0.7821
  Unseen — ETIS-Larib:  0.7112   ← hardest dataset
  Unseen — CVC-300:     0.8382

Generalization gap: +0.135 Dice points

Seen performance is competitive — PraNet reports 0.898 on Kvasir, we're at 0.911.
But the unseen performance drops significantly, especially on ETIS-Larib, which has
the most different imaging characteristics.

That 0.135 gap is what the thesis is about. Can SAM-LoRA close it?"

---

### Predictions [SHOW: prediction visualization]

"Qualitatively the model looks good on seen data — 0.91 to 0.99 Dice on these
four examples. The third image, the large cauliflower polyp, is almost perfect.
The fourth image is harder — two disconnected polyp regions — and you can see
some noise fragments around the smaller one. That fragmentation on multi-region
cases is a known weakness of purely convolutional models."

---

## NEXT STEPS (1–2 min)

"Phase 3 is SAM-LoRA. SAM's ViT-H encoder has 632 million parameters. We freeze
the entire encoder and inject LoRA adapters into the attention Q and V projections
with rank r=4. That adds roughly 4 million trainable parameters — about 0.6% of the
total model size — plus a lightweight convolutional decoder.

The training protocol is identical to U-Net so the comparison is fair.
I need an A100 or L4 runtime for ViT-H — that's the next Colab session.

After that: run both models over 3 seeds for error bars, then the comparison table
and ablation over LoRA rank. The thesis question gets answered there."

---

## IF ASKED: why LoRA specifically?

"LoRA — Low-Rank Adaptation — factorizes the weight update into two small matrices.
Instead of updating a full d×d weight matrix, you learn A (d×r) and B (r×d) where
r is much smaller than d. For SAM's attention layers with r=4, each Q and V projection
goes from ~4M parameters to ~8K trainable parameters.

It was originally designed for large language models but transfers directly to vision
transformers. The hypothesis is that polyp segmentation requires only a low-rank
adjustment to SAM's already-rich feature space, not a full retraining."

---

## IF ASKED: why not just fine-tune SAM fully?

"Two reasons. First, with 632M parameters and 1,450 training images, full fine-tuning
would immediately overfit. Second, destroying the pretrained representations defeats
the purpose — we want to test whether SAM's pretraining itself provides the
generalization advantage. LoRA preserves the frozen backbone while adapting only
what's necessary."

---

## IF ASKED: comparison to PraNet

"PraNet (2020) is the specialist SOTA we're benchmarking against. It uses a parallel
partial decoder with reverse attention — highly engineered for polyps.
Our U-Net baseline is already competitive on seen data. The interesting comparison
will be the unseen splits, where PraNet's gap is around 0.08–0.10 Dice.
Our U-Net gap is 0.135, so there's room to improve, and that's exactly what
SAM-LoRA is designed to address."
