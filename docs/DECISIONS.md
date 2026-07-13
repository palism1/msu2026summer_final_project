---
status: append-only
last_updated: 2026-07-13
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
- **Exempt:** `README.md` (GitHub renders it as the repo landing page) and `CLAUDE.md` (the agent repo
  map, consumed as tooling); their git history is the authoritative timestamp. If a `last_updated`
  field ever drifts, `git log -1 --format=%cs -- <file>` is the source of truth.
