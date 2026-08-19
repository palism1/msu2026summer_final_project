---
status: append-only
last_updated: 2026-08-19
---

<!-- FILE MAP | Decision log: notable choices, why they were made, and when.
     Newest last. Each entry: Decision · Why · Date (+ source commit/file where visible).
     Tags: [TWEAK] revisitable  ·  [DO NOT TOUCH] load-bearing, don't undo without reading why. -->

# Decisions

Running log of design choices for the polyp-segmentation study. Seeded from git history
and code comments so the reasoning survives outside anyone's head.

---

### PraNet split protocol + nested TestDataset path handling — 2026-06-23 [DO NOT TOUCH]
Train = all 1450 pre-packaged images (`TrainDataset/image/`); test = 5 fixed splits
(Kvasir + CVC-ClinicDB seen, CVC-ColonDB + ETIS-Larib + CVC-300 unseen). The PraNet zip
extracts with a doubly-nested `TestDataset/TestDataset/` folder, so `_find_test_root`
probes the nested path first. Matching the published PraNet protocol keeps our numbers
comparable to the literature. (`332cd59`, `src/data/dataset.py`)

### MD5 content-hash overlap check, not filename — 2026-06-23
Each dataset numbers files independently (Kvasir `1.jpg` ≠ CVC-ColonDB `1.jpg`), so
filename comparison gave false train/test overlap positives. We compare file content
hashes instead. (`e00296d`, `notebooks/01_data_pipeline.ipynb`)

### `Resize` is the last spatial op in the train transform — 2026-06-23
`RandomScale` changes image dimensions; without a final fixed `Resize` the batch tensors
had inconsistent sizes and collation broke. Resize-last guarantees a fixed output size.
(`efa9afc`, `src/data/transforms.py`)

### E-measure clipped to [0, 1] — 2026-06-23
The enhanced-alignment term `2ab/(a²+b²)` is in [-1, 1] per pixel; the mean could dip
below 0 for near-empty predictions, producing out-of-range scores. Clipped to [0, 1] to
match the metric's intended range. (`f95fb9f`, `src/metrics/segmentation.py`)

### Colab pulls code via `git fetch + reset --hard origin/main` — 2026-06-23
`git pull` left notebooks on stale local state across re-runs; a hard reset guarantees the
running code matches `main` every session, so results are reproducible. (`c7fa3d4`)

### SAM `pos_embed` interpolated to training resolution at init — 2026-06-26 [DO NOT TOUCH]
SAM's positional embedding is sized for 1024×1024 (64×64 patch grid); at our 352×352 input
`x + pos_embed` shape-mismatched. We bicubic-interpolate `pos_embed` to the 22×22 grid at
model init. Removing this crashes the SAM/MedSAM forward pass. (`eaa7a2d`, `src/models/sam_adapter.py`)

### SAM and MedSAM split into separate notebooks; neck embed_dim read dynamically — 2026-06-26
ViT-H and ViT-B have different neck output channels; hardcoding crashed the decoder. The
decoder now reads `embed_dim` from the encoder neck. SAM (03) and MedSAM (04) live in
separate notebooks because their checkpoints and GPU needs differ. (`eacb3a6`, `6e2ca2d`)

### MedSAM checkpoint from verified Zenodo source + MD5 — 2026-06-30 [DO NOT TOUCH]
Raw `wget` of the HuggingFace URL produced silent partial/corrupt downloads. We pin the
Zenodo record and assert MD5 `3bb6db55…` before use, deleting incomplete files first.
(`30a41d8`, `23c3bf3`, `notebooks/04_medsam_lora.ipynb`)

### Combined Dice + BCE loss — 2026-06-23 [TWEAK]
BCE alone under-segments small polyps; Dice alone is unstable early. The sum stabilizes
early training while optimizing overlap. Same loss for every model so comparisons are fair.
(`train.py`, `src/training/engine.py`)

### Best-only checkpoint on val Dice; no mid-run resume — 2026-06-23 [DO NOT TOUCH]
Training saves `best.pt` only when validation Dice improves, to
`checkpoints/<model>/seed<seed>/best.pt`. There is intentionally no resume-from-checkpoint:
`evaluate.py` and `05_benchmark.ipynb` load models purely by this path + raw `state_dict`.
Changing the path scheme or save cadence breaks both consumers. (`train.py`)

### LoRA on Q/V projections only, r=4 / α=8, frozen encoder + light CNN decoder — 2026-06-26 [TWEAK]
Adapting just the attention Q/V projections (with a small CNN mask decoder replacing SAM's
prompt decoder) trains ~1–3% of parameters — the parameter-efficiency claim of the study.
Rank/alpha are the main ablation knobs. (`src/models/sam_adapter.py`)

### Single config-driven `train.py`; efficiency metrics captured — 2026-07-02
Extracted the training loop (duplicated across notebooks 02/03/04) into one runner driven by
`configs/run.yaml`, so every model trains under an identical protocol on the same GPU. Each
run records accuracy, trainable/total params, checkpoint download size, device, and per-epoch
+ total wall-clock time into `metrics.json` — making the accuracy-vs-cost / simpler-machine
tradeoff measurable. Notebooks 01 and 05 stay on `main`; the superseded per-model training
notebooks 02–04 are preserved on the `backup/per-model-notebooks` branch. (`train.py`, `src/config.py`, `src/training/`)

### Zero-shot vanilla SAM/MedSAM baselines — GT-box prompted — 2026-07-02 [TWEAK protocol]
The "without fine-tuning" half of the comparison. SAM and MedSAM are promptable and cannot segment a
polyp from an image alone, so the vanilla baselines are prompted by the **bounding box of the GT mask**
(padded a few px) — the standard medical-SAM setup, and MedSAM is box-prompt-trained. This is an
ORACLE prompt (it reveals roughly where the polyp is) while the fine-tuned models get no hint; state
that asymmetry when comparing. Protocol is swappable (`ZS_PROMPT` = `box` | `point`, `ZS_BOX_PAD`) at
the top of the model-build cell in `05_benchmark.ipynb`; the pure prompt-derivation math is GPU-free
and unit-tested. `src/models/__init__.py` was made lazy (PEP 562) so importing `box_from_mask` for a
torch-free test does not drag in torch. (`src/models/zeroshot.py`, `tests/test_zeroshot.py`, `notebooks/05_benchmark.ipynb`)

### SAM ViT-B + LoRA variant (`sam_b`) + results aggregator — 2026-07-13
Two additions. (1) **`sam_b`**: a new model choice — SAM ViT-B fine-tuned with LoRA. It has the
**same ViT-B backbone as MedSAM but generic (non-medical) SAM weights**, so comparing `sam_vit_b`
to `medsam` in the benchmark isolates *backbone capacity* from *medical pretraining* — the confound
in the finding that MedSAM (ViT-B) trailed SAM (ViT-H). Implemented by mirroring the `medsam`
pattern: reuses `build_sam_lora(model_type="vit_b")` (no new builder), reads a `sam_b` config block,
lands at `checkpoints/sam_vit_b/seed<seed>` (the existing `[DO NOT TOUCH]` path scheme, only a new
leaf added). Optional in `05_benchmark.ipynb` so an untrained `sam_b` never breaks the benchmark.
(2) **Results aggregator**: `src/results_summary.py` (torch-free) + `aggregate_results.py` consolidate
every per-run `metrics.json` (local or the Drive mirror) into `results/summary/` (SUMMARY.md + CSVs +
JSON) with mean ± std across seeds — so results are viewable without re-running any notebook.
(`src/config.py`, `src/training/engine.py`, `train.py`, `evaluate.py`, `configs/base.yaml`,
`src/results_summary.py`, `aggregate_results.py`, `f7ab197`)

### Docs classified living / immutable / append-only in frontmatter — 2026-07-13
Every committed markdown doc declares its class in YAML frontmatter so a reader knows whether it is
current state or a historical record (convention borrowed from the stroke-burden-index project):
- **living** (`status: living`, `last_updated: YYYY-MM-DD`) — describes current state; edit freely and
  bump `last_updated` on any substantive change. Applies to `docs/PROJECT_PLAN.md`,
  `docs/DELEGATION_PLAN.md`, and (local, gitignored) `HANDOFF.md`.
- **immutable** (`status: immutable`, `date:`, `superseded_by: null`) — a report/review frozen at a
  point in time; substantive changes require a new superseding doc that sets the old one's
  `superseded_by`, and only typo/link fixes happen in place. (None yet; the final write-up
  `docs/FINDINGS.md` will be immutable once published.)
- **append-only** (`status: append-only`) — this decisions log; entries are never rewritten, only
  added, and reversals come as new entries referencing the old one.
- **Exempt:** `README.md` (GitHub renders it as the repo landing page, and it carries the repo map);
  its git history is the authoritative timestamp. If a `last_updated` field ever drifts,
  `git log -1 --format=%cs -- <file>` is the source of truth.

### Model-dependent input normalization + augmentation-matched control arm — 2026-07-28 [DO NOT TOUCH the legacy keys]
Fixes H1 from `docs/MEDSAM_INVESTIGATION.md`: `src/data/transforms.py` applied ImageNet
standardization to every model. That is correct for SAM and U-Net by construction — SAM's own
`Sam.preprocess` computes the same `(x - mean)/std` with the same constants x255 — but wrong for
MedSAM, whose frozen encoder (per `bowang-lab/MedSAM`'s `train_one_gpu.py` / `MedSAM_Inference.py`)
was fit on per-image `[0,1]` min-max inputs and never saw ImageNet statistics. ImageNet stays the
default for `unet`/`sam_lora`/`sam_b`/`medsam` — no published number moves; a corrected MedSAM run
is a *new* model key (`medsam_minmax`), not an edit to `medsam`, so `checkpoints/medsam/...` and
`results/medsam/...` are untouched.

**Why `medsam_ctrl` exists.** `min_max_normalize` is exactly invariant to any affine map
`y = a*x + b` with `a > 0` — precisely what `A.ColorJitter`'s brightness (`multiply(img, a)`) and
contrast (`multiply_add(img, f, mean*(1-f))`, where the offset is itself a scalar mean) compute.
Verified against `albumentations/augmentations/pixel/functional.py`. So a straight `medsam` ->
`medsam_minmax` comparison would confound normalization with augmentation strength: the minmax arm
silently gets weaker effective photometric jitter, biased toward *worse* unseen generalization —
dangerous in the direction of the headline claim. `medsam_ctrl` is the augmentation control:
ImageNet-normalized, brightness/contrast forced to 0 (saturation/hue untouched — those survive
min-max and are not part of the confound). Primary comparison is `medsam_ctrl` -> `medsam_minmax`
(matched except normalization); `medsam` -> `medsam_ctrl` sizes the confound itself.

**Protocol is code, not config.** `src/config.MODEL_SPECS` binds `(normalization, color_jitter)` to
each model key; `reject_protocol_overrides` raises if a YAML block tries to set either under
`medsam:`/`sam:`/`sam_b:`/`model:`. Reason: checkpoint and results paths derive from the model key
alone (`checkpoints/<model>/seed<seed>/best.pt`), so a config-settable protocol would let two
different input pipelines silently write into and overwrite the same published path. All three
MedSAM arms (`medsam`, `medsam_minmax`, `medsam_ctrl`) share one `configs/base.yaml` block —
weights and LoRA settings cannot drift apart between them; only the protocol columns differ.

**`A.Lambda` over `A.Normalize(normalization="min_max")`.** The latter postdates the pinned
`albumentations>=1.3.0` and albucore's implementation is `(max-min+1e-4)`, not MedSAM's
`clip(max-min, 1e-8)`. `src/normalization.min_max_normalize` reproduces MedSAM's formula exactly and
is wrapped in `A.Lambda` so both training and zero-shot inference share one tested implementation.

**Zero-shot fix.** `ZeroShotSAM._set_image` normalizes with the same numpy helper when
`normalization="minmax"`, then feeds SAM's encoder through a `pixel_mean=0/pixel_std=1` context
manager (`_identity_preprocess`) so `Sam.preprocess`'s resize-and-pad still runs but its ImageNet
standardization is skipped — the caller already normalized. This also fixes the oracle-tier
`vanilla_medsam` contamination H1 identified; the corrected oracle row is the new
`vanilla_medsam_minmax` baseline (experiment A), not an edit to the published `vanilla_medsam`.
`vanilla_sam_b` (experiment B) fills the missing cell of the 2x2 (same-backbone oracle comparison).
Both published oracle rows (`vanilla_sam`, `vanilla_medsam`) are blocked in the new
`zeroshot_eval.py` unless `--allow-published` is passed, and even then only if a prior copy of the
row is recoverable from `results/` or the Drive mirror — re-running them from a different code path
must never silently clobber the published numbers.
(`src/normalization.py`, `src/data/transforms.py`, `src/config.py`, `src/models/zeroshot.py`,
`zeroshot_eval.py`, `src/training/engine.py`, `src/training/reporting.py`, `train.py`,
`evaluate.py`, `src/results_summary.py`, `docs/MEDSAM_INVESTIGATION.md`)

### Ensemble, weighted ensemble, and box cascade — pre-registration and design choices — 2026-08-19
Pre-registered before any ensemble or cascade number was computed, per `docs/PLAN_ENSEMBLE.md`.
Written here first so a later result cannot be read as having driven the design.

**Member sets.** Member set A is U-Net plus SAM-ViT-H — the largest architectural distance in
the study. Member set B is the top three rows of the current unseen leaderboard: SAM-ViT-H,
SAM-ViT-B, U-Net. Both sets get a uniform arm (`ens_unet_samh`, `ens_top3`) and a fitted arm
(`ens_unet_samh_fitted`, `ens_top3_fitted`). All four arms are reported, whatever the outcome —
uniform approximately equal to fitted is an acceptable, reportable result, not a failed
experiment.

**Fit slice is the first contiguous half of each seen test split, in sorted path order** —
Kvasir indices 0-49 fit, 50-99 hold out; ClinicDB indices 0-30 fit, 31-61 hold out
(`fit_holdout_indices`, `src/ensemble/combine.py`). Not an interleaved (even/odd) split:
CVC-ClinicDB's 62 seen-test images come from a handful of video sequences, so an interleaved
split would place near-duplicate frames from the same sequence on both sides of the fit/holdout
boundary, leaking sequence identity into the weight fit. A contiguous split keeps whole
sequences on one side. This costs some statistical similarity between the two halves; that cost
is accepted because the risk being guarded against is leakage, not power. The primary endpoint
for every ensemble row stays the untouched unseen test sets (CVC-ColonDB, ETIS-Larib, CVC-300),
which the fit slice never touches.

**Three-tier table split, not two.** `src/results_summary.py` already separated trained
(`prompt-free`) from untrained, GT-prompted (`oracle`) rows. Ensembles and the cascade add a
third tier, `derived` (`TIER`, `tier_of`, `split_by_tier`). They read no ground truth at
inference, so they are fair on accuracy against the prompt-free tier — but each one costs more
compute than any single member and is a combination of the models above rather than a new
architecture, so it is not a peer for a "which architecture generalizes better" table either. The
three tiers are never merged into one ranked table; the derived section's caption names the best
single-model unseen mDice inline so a reader can compare without the tables being merged.

**Empty-box policy defaults to "zero".** When the box cascade's detector proposes no box (an
empty thresholded mask), the cascade writes an all-zero probability map rather than falling back
to the detector's own prediction (`empty_policy="zero"`, `src/ensemble/cascade.py`). The method
invents no detection it did not make. A `"passthrough"` policy exists as a documented ablation,
used only if the empty rate on any split exceeds 2 percent.

**Detector provenance is the float16 prediction cache, not a live U-Net forward pass.** The
cascade reads U-Net's cached probability map from Phase 1 (`predict_cache.py`'s output) to derive
its box, the same array the ensemble's U-Net member uses. This is deliberate — it lets the
cascade run without ever loading the U-Net checkpoint, and it keeps the detector output
consistent across every consumer of that cache. The cached values carry float16 rounding: a
boundary pixel within about 1e-3 of the 0.5 threshold can flip, which can move a box edge by one
pixel. With `box_padding=5` (matching `zeroshot.box_padding`) the effect on the derived box is
nil. This is *not* claimed to be byte-identical to a live U-Net forward pass — only that both
cascade rows (fair and ceiling) share one image loader and one cached-probability convention, so
the comparison between them is apples-to-apples.

**"Oracle box into the trained models" does not exist as a row.** `SAMLoRA` (the trained SAM/
MedSAM architecture) discards SAM's prompt encoder and mask decoder in favor of
`LightDecoder(encoder_features)` — it has no box input, and neither does U-Net. The row is
architecturally unavailable, not merely unrun. Its intent — an oracle-prompted ceiling for the
cascade — is served instead by the existing `vanilla_medsam_minmax` oracle row (0.9246 unseen
mDice) and by Phase 4's `oracle_casc_gtbox_medsam`, a cheap path-matched reproduction of that
ceiling built through the same image loader and box-derivation code the fair cascade row uses.

**Weighting stays seed-matched; the decision threshold stays at 0.5.** Ensemble seed 42 combines
each member at seed 42 only — cross-seed pooling is a different question (ensemble-of-seeds) and
is out of scope. Every published row, fitted or uniform, thresholds at 0.5; a fitted threshold
would be a second confound layered on top of fitted weights. Phase 3 records a threshold
sensitivity sweep (0.3-0.7) on the fit slice only, as a diagnostic, not as a second knob.

(`src/ensemble/cache.py`, `src/ensemble/combine.py`, `src/ensemble/cascade.py`, `src/config.py`,
`src/models/zeroshot.py`, `src/results_summary.py`, `predict_cache.py`, `ensemble_eval.py`,
`cascade_eval.py`, `docs/PLAN_ENSEMBLE.md`)
