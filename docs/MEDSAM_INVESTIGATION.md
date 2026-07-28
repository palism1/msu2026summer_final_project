---
status: living
last_updated: 2026-07-28
---

<!-- FILE MAP | Root-cause investigation into MedSAM-ViT-B + LoRA's weak scores (0.819 seen /
     0.661 unseen), raised as a review question on 2026-07-28. Ranked hypotheses, the evidence
     for each from the existing code and results, and the experiments that would separate them.
     Pairs with docs/FINDINGS.md (which currently attributes the deficit to medical pretraining —
     see H1, that attribution is confounded). Doc class: living. -->

# Why MedSAM Underperforms — Investigation

## The observation

MedSAM-ViT-B + LoRA is last on every split, and the margin widens off-distribution:

| Model | Seen mDice | Unseen mDice | ETIS-Larib |
|---|---|---|---|
| SAM-ViT-B + LoRA | 0.865 | 0.761 | 0.692 |
| MedSAM-ViT-B + LoRA | 0.819 | 0.661 | 0.500 |

These two runs are matched on everything the training code controls: identical ViT-B architecture,
identical LoRA config (r=4, α=8, 322,273 trainable params — the counts match exactly), identical
optimizer, schedule, seeds, augmentation, and decoder. The only difference is the initial encoder
weights. MedSAM is a fine-tune *of* SAM ViT-B, so a medically-adapted model losing 0.100 unseen
mDice to its own generic starting point is the result worth explaining.

`docs/FINDINGS.md` currently reads this as "medical pretraining is a net drag." That conclusion is
premature — H1 below is a confound that has to be cleared before the pretraining claim can stand.

---

## H1 — Input normalization mismatch (highest confidence, and it's a bug)

**MedSAM is being fed an input distribution it was never trained on, while SAM is fed exactly the
one it was.**

`src/data/transforms.py:16,24` applies ImageNet standardization to every model:

```python
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

For SAM this is correct by construction. SAM's own `preprocess()` computes
`(x - pixel_mean) / pixel_std` with `pixel_mean=[123.675, 116.28, 103.53]`,
`pixel_std=[58.395, 57.12, 57.375]` — those are the ImageNet constants scaled by 255. Our transform
reproduces SAM's native preprocessing.

MedSAM does not use that path. Both `train_one_gpu.py` and `MedSAM_Inference.py` in
`bowang-lab/MedSAM` feed the image encoder directly, bypassing `sam.preprocess`, after per-image
min-max scaling:

```python
img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
```

The training loader asserts it outright: `"image should be normalized to [0, 1]"`.

So MedSAM's encoder was fit on inputs in **[0, 1]** (mean ≈ 0.45, std ≈ 0.25) and we hand it inputs
in roughly **[−2.1, +2.6]** (mean ≈ 0, std ≈ 1) — about a 4× dynamic-range inflation plus a mean
shift. The encoder is frozen; the only capacity available to absorb that shift is rank-4 LoRA on the
qkv projections. A frozen network's first patch-embed conv and every LayerNorm downstream see
statistics outside their calibration range.

This is asymmetric by exactly the axis the study is trying to measure: the SAM-ViT-B arm gets native
preprocessing, the MedSAM arm gets a mismatched one. Any "medical pretraining hurts" conclusion
drawn from that comparison is confounded with "we preprocessed MedSAM wrong."

**It also contaminates the oracle tier.** `ZeroShotSAM` (`src/models/zeroshot.py:119`) calls
`SamPredictor.set_image()`, which routes through `set_torch_image` → `self.model.preprocess(...)` —
SAM's ImageNet-scale standardization again. So the vanilla MedSAM oracle number (0.845 unseen) is
*also* measured under the wrong normalization and is likewise understated.

**Fix:** make normalization model-dependent — `[0,1]` min-max for MedSAM, ImageNet stats for SAM/
U-Net — rather than a single global transform. **Status: fixed** — `src/normalization.py` +
`src/data/transforms.py` + `src/config.MODEL_SPECS`, see `docs/DECISIONS.md` 2026-07-28.

### H1 addendum — the fix isn't free: affine invariance confounds normalization with augmentation

The naive version of this fix — re-run MedSAM+LoRA once more, this time with `[0,1]` min-max
substituted for ImageNet standardization — turns out to change two things at once, not one.

`min_max_normalize(x) = (x - x.min()) / (x.max() - x.min())` is exactly invariant to any global
affine map `y = a*x + b` with `a > 0`. Checked against
`albumentations/augmentations/pixel/functional.py`: `adjust_brightness_torchvision` computes
`multiply(img, a)` (pure scale); `adjust_contrast_torchvision` computes
`multiply_add(img, f, mean*(1-f))`, and the offset is `mean(gray(x))*(1-f)`, itself a scalar function
of `x` — so contrast is also `f*x + b` for a per-image constant `b`. `adjust_saturation_torchvision`
(`addWeighted`, linear) and hue (an HSV rotation that scales V) are positively homogeneous, so a
multiplicative brightness factor commutes through ColorJitter's shuffled op order regardless of
which op runs first. Net effect: **brightness and contrast jitter cannot reach a min-max-normalized
model** except through uint8 rounding residue; saturation and hue still do.

So a plain `medsam` -> `medsam_minmax` comparison confounds normalization (the thing we want to
measure) with a *weakening of effective augmentation* on the minmax arm — and the sign of that
confound biases toward the dangerous direction: less augmentation predicts worse unseen
generalization, which would deflate exactly the effect we're hoping the fix produces.

**Three-arm design** (`src/config.MODEL_SPECS`, `STANDARD_COLOR_JITTER` / `NO_BC_COLOR_JITTER`):

| Arm | Normalization | ColorJitter | Role |
|---|---|---|---|
| `medsam` | imagenet | b .2 / c .2 / s .2 / h .1 | published baseline, never re-run |
| `medsam_ctrl` | imagenet | b 0 / c 0 / s .2 / h .1 | augmentation control |
| `medsam_minmax` | minmax | b .2 / c .2 / s .2 / h .1 | MedSAM under its own preprocessing |

**Primary comparison: `medsam_ctrl` -> `medsam_minmax`.** Matched on augmentation exposure, differ
only in normalization — the controlled test of H1. **Secondary: `medsam` -> `medsam_ctrl`.** Sizes
the confound itself (how much of any `medsam` -> `medsam_minmax` delta was never about
normalization). `medsam_minmax` keeps full jitter deliberately, so it also sits validly in the main
results table under the same protocol shape as every other trained model.

**Residuals — named, not eliminated:**
1. Uint8 clipping at brightness/contrast factors > 1 leaves a small residue after min-max (favors
   `medsam_minmax` slightly).
2. Contrast's offset only cancels exactly when no hue op follows it in ColorJitter's shuffled order.
3. `A.Rotate`'s border fill pins the per-image min in the minmax arm, a second-order effect on the
   normalization itself.
4. Min-max normalization is itself a per-image contrast normalizer — that is the treatment being
   measured, not a confound to control away.

---

## H2 — MedSAM's encoder was allowed to offload localization to the box prompt

This one is not a bug; it is the genuinely interesting scientific answer, and it survives even if H1
is fixed.

MedSAM's fine-tuning recipe (confirmed in `train_one_gpu.py`):

- the prompt encoder is frozen (`for param in self.prompt_encoder.parameters(): param.requires_grad = False`)
- the image encoder **and** mask decoder are trained
- every training step supplies a bounding box — `medsam_pred = medsam_model(image, boxes_np)`; there
  is no point-prompt or no-prompt path in the forward

Across 1.5M box-prompted pairs, the image encoder is never asked to answer "where is the object."
The box always answers that. Gradient descent has no reason to preserve prompt-free objectness or
saliency structure in the embedding, and some reason to trade it away for boundary and texture
fidelity inside the box.

Our pipeline discards SAM's prompt-based decoder entirely (`LightDecoder`, `sam_adapter.py:69`)
and reads a mask straight off the encoder features with no prompt at all. That reads out precisely
the capability MedSAM's recipe let it shed.

**The existing results already show the signature.** Compare each model's prompt-free score against
its own oracle-box score on unseen splits:

| Model | Prompt-free unseen | Oracle-box unseen | Lift from the box |
|---|---|---|---|
| SAM-ViT-H | 0.806 | 0.905 | **+0.099** |
| MedSAM-ViT-B | 0.661 | 0.845 | **+0.184** |

MedSAM gains roughly twice as much from being told where to look. That is what prompt-dependence
looks like. The comparison is imperfect — it crosses ViT-H against ViT-B — which is why Experiment
B below exists.

---

## H3 — Modality gap: MedSAM's corpus is largely grayscale radiology

MedSAM's ~1.57M image-mask pairs span ten imaging modalities dominated by CT, MRI, X-ray, and
ultrasound. Endoscopy is a small minority slice, and the bulk of the corpus is effectively
single-channel data replicated to 3 channels.

Polyp segmentation leans on properties that corpus barely contains: mucosal color, vascular pattern,
specular highlights, and the red-on-pink contrast separating a polyp from healthy wall. Fine-tuning
SAM's SA-1B-trained features on mostly-achromatic data plausibly degrades exactly those
color-sensitive filters.

Consistent with the per-split numbers: MedSAM's collapse is worst on ETIS-Larib (0.500), the split
with the most distinct scope hardware, illumination, and color balance.

---

## H4 — Resolution mismatch amplifies on a narrowly fine-tuned model

We train at 352×352 (`configs/base.yaml → data.img_size`), so `SAMLoRA._resize_pos_embed`
(`sam_adapter.py:134`) bicubically resamples the positional embedding from a 64×64 grid (1024px) to
22×22. ViT-B's windowed attention (window 14) then tiles a 22×22 grid raggedly.

Both ViT-B arms take this hit identically, so it cannot explain the *differential* on its own. It is
listed because fine-tuning narrows the input regime a model tolerates: SAM saw enormous scale
diversity across SA-1B, whereas MedSAM saw 1024×1024 exclusively. The same perturbation should cost
the more specialized model more. Rank this below H1–H3 and treat it as a modifier, not a cause.

---

## Experiments that separate these

Ordered by information gained per GPU-hour. A–B are cheap; C–D need training runs.

**A. Normalization ablation (no training, ~10 min).** Re-run the vanilla MedSAM oracle-box baseline
twice — once as-is, once with min-max `[0,1]` preprocessing substituted for `sam.preprocess`. Isolates
H1 with zero training. If the `[0,1]` variant jumps meaningfully above 0.845 unseen, H1 is confirmed
and every MedSAM number in the study needs regenerating.

**B. Vanilla SAM-ViT-B oracle-box baseline (no training, ~10 min).** The missing cell of the 2×2.
Fills in the H2 table with a same-backbone comparison, turning "MedSAM gains 2× more from the box"
from suggestive into controlled. `build_zeroshot_sam(checkpoint=sam_vit_b_01ec64.pth,
model_type="vit_b")` already supports this — it is one config line.

**C. MedSAM + LoRA retrained under correct `[0,1]` preprocessing (3 seeds, ~90 min A100).** The
decisive run. Three outcomes, all publishable:
- closes most of the 0.100 gap → the deficit was a preprocessing artifact, and `FINDINGS.md`'s
  pretraining conclusion must be retracted
- closes part of it → both H1 and H2/H3 contribute; report the decomposition
- closes none of it → H1 is cleared, and the H2/H3 story stands on much firmer ground

**D. Color-sensitivity probe (no training, ~15 min).** Evaluate both ViT-B arms on grayscale-
replicated inputs. If MedSAM degrades far less than SAM ViT-B, its features are already
color-insensitive and H3 gains direct support.

Run A and B first. They are nearly free, and A determines whether C is a correction or an ablation.

---

## What to tell the professor now

The honest current position: MedSAM's deficit has at least one identified methodological cause
(H1, a normalization mismatch that handicaps only the MedSAM arm) sitting on top of at least one
substantive one (H2, an encoder fine-tuned to rely on a prompt we never give it). The published
comparison cannot currently distinguish them, and the write-up's "medical pretraining is a net drag"
line overstates what the data supports. Experiments A and C settle it.

---

## How to run these

Code for A, B, and C landed 2026-07-28 (`docs/DECISIONS.md`). D is not wired — no code exists for
the grayscale-replication probe yet.

**A. Vanilla MedSAM under its own `[0,1]` preprocessing (no training, ~2 min).**
```
python zeroshot_eval.py --config configs/run.yaml --baseline vanilla_medsam_minmax
```
Lands at `results/vanilla_medsam_minmax/seed0/metrics.json`. Compare its unseen mDice against the
published `vanilla_medsam` (0.845 unseen) — a meaningful jump confirms H1 at the oracle-tier level.

**B. Vanilla SAM-ViT-B oracle-box baseline (no training, ~2 min).**
```
python zeroshot_eval.py --config configs/run.yaml --baseline vanilla_sam_b
```
Lands at `results/vanilla_sam_b/seed0/metrics.json`. Fills the missing cell of the 2x2 — compare
against `vanilla_medsam` (same ViT-B backbone, same box prompt, different pretraining) and against
`vanilla_medsam_minmax` (H2 evidence: is MedSAM's oracle-box lift still ~2x SAM's once the ViT-B
backbones actually match?).

**C. MedSAM + LoRA retrained under the three-arm design (3 seeds x 2 new arms, ~95 min A100).** Run
`medsam_minmax` and `medsam_ctrl` together — the comparison is uncontrolled if only one exists.
```
python train.py --config configs/run.yaml --model medsam_minmax --seed 42
python train.py --config configs/run.yaml --model medsam_ctrl   --seed 42
# repeat for seed 43, 44 (or drive both through notebooks/train_colab.ipynb's MODELS list)
```
Lands at `checkpoints/medsam_minmax/seed<N>/`, `checkpoints/medsam_ctrl/seed<N>/` and the matching
`results/` paths — the published `checkpoints/medsam/` and `results/medsam/` are never touched.
Read out with `python aggregate_results.py`, then compare:
- **Primary: `medsam_ctrl` -> `medsam_minmax` unseen mDice.** The controlled normalization
  ablation. This is the number that answers H1.
- **Secondary: `medsam` -> `medsam_ctrl`.** Sizes how much of any naive `medsam` -> `medsam_minmax`
  delta would have been augmentation-confound rather than normalization.
- The "medical pretraining is a net drag" line in `docs/FINDINGS.md` stands only if the
  `medsam_ctrl` -> `medsam_minmax` gap reproduces most of the original 0.100 unseen-mDice deficit
  against `SAM-ViT-B + LoRA`; if it closes most of the gap, that line needs retracting instead.

After A, B, and C: `python aggregate_results.py` (no GPU) consolidates every new row into
`results/summary/`, and `docs/FINDINGS.md` gets its caveat updated per `docs/DECISIONS.md`.
