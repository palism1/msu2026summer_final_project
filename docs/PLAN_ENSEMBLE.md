# Implementation Plan: Ensemble, Weighted Ensemble, and Box Cascade

**Status:** post-defense portfolio work. Do not touch `docs/SPEAKER_SCRIPT.md`, `docs/PRESENTATION_OUTLINE.md`, or `make_figures.py`.

This plan passed an adversarial review. Nine findings (three blocking) were folded in; the changelog at the end records each change.

---

## What the repository already gives you

**No per-image predictions exist.** Each run directory holds only `metrics.json`, `run.log`, and 16 overlay PNGs. Every part of this experiment needs a fresh inference pass. There is no near-free start.

**All models already evaluate on one common grid.** `src/training/reporting.py:evaluate_all_splits` builds `PolypDataset` with `get_val_transform(352, normalization)`. The transform resizes the image and the mask to 352x352, then normalizes the image only. The ground truth array for a given (split, index) is identical for every member, whatever its normalization. Probability maps are directly averagable.

**Metrics take numpy arrays.** `MetricTracker.update(pred_HxW_float01, gt_HxW_01)`. No torch. An ensemble evaluator needs numpy and scipy only.

**Checkpoints are not in the repository.** `checkpoints/` does not exist locally and `*.pt` is gitignored. The only copies live in Drive at `/content/drive/MyDrive/msu2026_checkpoints/<model_dir>/seed<N>/best.pt`. SAM ViT-H `best.pt` is 2533 MB per seed.

**Trained SAM models need their base backbone file at construction time.** `evaluate.py:56` calls `build_sam_lora(sam_checkpoint=...)`, which builds the model from the base checkpoint before `load_state_dict` overwrites it. Phase 1 therefore needs `sam_vit_h_4b8939.pth` (2.4 GB) and `sam_vit_b_01ec64.pth` in the repository root. Phase 4 needs `medsam_vit_b.pth`.

**`results/` is gitignored.** A fresh Colab clone has no `results/` directory at all. The published numbers live only in the Drive mirror. Any step that reads a published `metrics.json` must run after a restore step.

**Environment is Colab plus an A100.** `notebooks/train_colab.ipynb` clones the repository, installs `requirements.txt` and `segment-anything`, downloads the two PraNet zips and the backbone `.pth` files into the repository root, then shells out to `train.py`.

**The oracle ceiling for the cascade already exists.** `results/vanilla_medsam_minmax/seed0` is "GT box into vanilla MedSAM under min-max normalization", at 0.9246 unseen mDice.

### Interpretations chosen, and why

1. **"Oracle box into the trained models" is not architecturally available.** `SAMLoRA` discards SAM's prompt encoder and mask decoder, and uses `LightDecoder(encoder_features)`. It has no box input. U-Net has none either. The row is dropped. Its intent is served by the existing `vanilla_medsam_minmax` row. Phase 4 adds one cheap path-matched version of that ceiling so the two rows share an image loader.
2. **Ensembles and the cascade get a third tier, `derived`.** They stay fair, because they read no ground truth at inference. They are not peers of a single trained model in a table that asks "which architecture generalizes better". A separate table with an explicit caption keeps both claims readable. The caption names the best single-model unseen mDice inline.
3. **Weight fitting uses the first contiguous half of each seen test split, in sorted path order.** Kvasir indices 0 to 49 fit, 50 to 99 hold out. ClinicDB indices 0 to 30 fit, 31 to 61 hold out. CVC-ClinicDB frames come from video sequences, so an interleaved split would place near-duplicate frames on both sides. A contiguous split separates sequences. It costs some statistical similarity between the halves, and that cost is accepted, because the risk here is leakage, not power. The primary endpoint stays the untouched unseen sets.
4. **Ensembles are seed-matched.** Ensemble seed 42 combines each member at seed 42. This yields three ensemble runs, and mean plus or minus std over seeds 42, 43, 44. Cross-seed pooling is out of scope: it measures ensemble-of-seeds, a different question.
5. **The decision threshold stays at 0.5.** Every published row uses it. A fitted threshold would be a second confound. Phase 3 records a threshold sensitivity diagnostic on the fit slice only.

### Pre-registration

Write these choices into `docs/DECISIONS.md` **before** you see any ensemble number.

- Member set A: U-Net plus SAM-ViT-H. Reason: the largest architectural distance in the study.
- Member set B: the top three rows of the current unseen leaderboard, which are SAM-ViT-H, SAM-ViT-B, U-Net.
- Both sets get a uniform arm and a fitted arm. All four arms get reported, whatever the outcome.
- The fit slice is the first contiguous half of each seen test split, in sorted path order.
- The cascade detector output comes from the float16 prediction cache, not from a live U-Net forward pass.
- Uniform approximately equal to fitted is an acceptable, reportable result.

### Expected ranges, so a bug is visible

| Row | Expect unseen mDice | Suspect a bug below |
|---|---|---|
| Ensemble A, uniform | 0.77 to 0.83 | 0.75 |
| Ensemble B, uniform | 0.77 to 0.83 | 0.75 |
| Fitted minus uniform | -0.005 to +0.015 | absolute value above 0.05 |
| Cascade, U-Net box | 0.75 to 0.88 | 0.60 |
| Cascade ceiling, GT box | 0.92 plus or minus 0.01 | any deviation above 0.02 from 0.9246 |

A uniform ensemble that lands at 0.77 to 0.79 is not a bug. U-Net trains 24.4M parameters and saturates its sigmoid harder than the LoRA models, so a uniform average follows U-Net and can sit below SAM-ViT-H's 0.806. Read the calibration diagnostic before you suspect code. Only a result below 0.75 signals a defect.

The cascade range follows from arithmetic. MedSAM with a correct box scores about 0.92. If U-Net produces a usable box on a fraction `f` of unseen images, the cascade scores about `0.92f`. Detection caps the cascade.

---

## Phase 0: inventory gate

**Goal:** prove every artifact exists before you write inference code.
**Time:** 20 to 40 minutes, most of it downloads.

Implement the gate as `python predict_cache.py --inventory`. It runs inside the Colab session, so it may import torch. Only `--dry-run` stays laptop-safe and torch-free.

The gate prints one table and exits 1 when any row fails.

| Check | Pass condition |
|---|---|
| Base backbones | `sam_vit_h_4b8939.pth`, `sam_vit_b_01ec64.pth`, `medsam_vit_b.pth` exist in the repository root |
| SAM package | `importlib.util.find_spec("segment_anything")` returns a spec |
| Data | `build_splits(cfg.data.root)` then `verify_splits` reports 1450, 100, 62, 380, 196, 60 |
| Trained checkpoints | `best.pt` present locally or in Drive for `{unet, sam_vit_h, sam_vit_b}` times `{42, 43, 44}` |
| Published metrics | `metrics.json` present locally or in Drive for the same nine runs |
| Session disk | at least 3 GB free |

Use `find_spec` and not a real import, so the check reports a missing package cleanly instead of raising an unrelated CUDA error.

Expected checkpoint sizes: 97.9 MB for U-Net, 2533.1 MB for SAM ViT-H, 348.9 MB for SAM ViT-B. Print the measured size next to the expected one.

**Go or no go.** If any SAM-ViT-H seed is missing, retraining costs about 2 hours of A100 time per seed. In that case, reduce member set A to U-Net plus SAM-ViT-B, record the reduction in `docs/DECISIONS.md`, and continue. Do not retrain silently.

---

## Phase 1: results restore and prediction cache

**Goal:** restore the published rows, then write per-image probability maps once.
**Time:** 2 to 3 hours of coding, then one GPU session of 45 to 75 minutes.

### Step 0: restore the Drive results mirror

This step is mandatory and comes first. `results/` is gitignored, so a fresh clone has none of it. Verification, the `inference.json` sidecars, and the cost table all read published rows.

In `notebooks/08_ensemble.ipynb`, step 0 does the following.

1. Mount Drive.
2. Copy `/content/drive/MyDrive/msu2026_checkpoints/results/` into `./results/` with `shutil.copytree(..., dirs_exist_ok=True)`, matching the restore pattern already used in `train_colab.ipynb` cell 4.
3. Print the count of restored `metrics.json` files. Expect 22: 18 trained runs plus 4 oracle rows.
4. Stop the notebook when the count is below 18.

`predict_cache.py` must also fail with a directed message, not a bare `FileNotFoundError`. When the published `metrics.json` is missing it prints: restore the Drive results mirror into `results/` first, see notebook 08 step 0, then exits 2.

### Files to create

**`src/ensemble/__init__.py`** — lazy exports through PEP 562 `__getattr__`, copied in shape from `src/models/__init__.py`. Torch-free tests must import the pure helpers.

**`src/ensemble/cache.py`** — numpy and stdlib only. No torch, no albumentations.

```python
CACHE_VERSION = 1

def cache_dir(cache_root, model_dir, seed) -> Path          # <root>/<model_dir>/seed<N>
def cache_path(cache_root, model_dir, seed, split) -> Path  # .../<split>.npz

def pack_gt(gt: np.ndarray) -> np.ndarray      # (N,H,W) bool -> packed uint8 1-D
def unpack_gt(packed, shape) -> np.ndarray     # -> (N,H,W) float32 in {0,1}
def gt_fingerprint(packed: np.ndarray) -> str  # sha256 hex

@dataclass
class SplitCache:
    probs: np.ndarray        # (N,H,W) float32, decoded from float16
    gt: np.ndarray           # (N,H,W) float32 in {0,1}
    image_paths: list[str]
    mask_paths: list[str]
    gt_sha256: str

def write_split_cache(path, probs, gt, image_paths, mask_paths) -> Path
def read_split_cache(path) -> SplitCache
def write_manifest(dir_path, payload: dict) -> Path
def read_manifest(dir_path) -> dict
def assert_aligned(caches: list[SplitCache]) -> None
def environment_record() -> dict
```

`assert_aligned` raises `ValueError` when `image_paths` or `gt_sha256` differ across members. This converts the alignment assumption into a runtime check.

`environment_record` returns `{"python", "torch", "torchvision", "numpy", "scipy", "albumentations", "segment_anything", "cuda", "cudnn"}`. Read each version through `importlib.metadata.version`, inside a try block, falling back to `"unknown"`. Read the torch and CUDA values only when torch is already imported, so the module stays torch-free.

Storage per `.npz`: `probs_f16`, `gt_packed`, `shape`, `image_paths`, `mask_paths`, `version`. Use `np.savez`, not the compressed form. Size per (model, seed) over all five splits is about 210 MB. Nine runs need about 1.9 GB.

`unpack_gt` must slice the unpacked bit array to `prod(shape)` before the reshape, because `np.packbits` pads to a byte boundary.

**`predict_cache.py`** — repository root CLI, shaped like `zeroshot_eval.py`.

```
python predict_cache.py --inventory
python predict_cache.py --config configs/run.yaml --model sam_lora --seed 42 [--dry-run]
                        [--cache-dir cache] [--accept-drift TOL --reason TEXT]
```

- `--dry-run` imports no torch. It prints the resolved plan and exits 0.
- `--inventory` runs the Phase 0 gate.
- A real run reuses `build_run_plan`, then builds the model with the exact branch logic of `evaluate.py` lines 50 to 85. Pass `encoder_weights=None` for U-Net, as `evaluate.py` does.
- For each of the five test splits, build the same loader as `evaluate_all_splits`: `PolypDataset`, `get_val_transform(plan.img_size, plan.normalization)`, `DataLoader(batch_size=8, shuffle=False, num_workers=plan.num_workers)`.
- Time the forward pass only. Call `torch.cuda.synchronize()` before and after each `model(images)` call. Exclude model construction, checkpoint load, and disk write.
- Write `<cache>/<model_dir>/seed<N>/<split>.npz` and one `manifest.json` per (model, seed).
- Verify, then write the sidecar, then mirror.

### The verification gate and its failure procedure

Verification is on by default. Compute the six metrics from the cached float16 probabilities. Compare against the restored `results/<model_dir>/seed<N>/metrics.json`. Exit 2 when any absolute dice delta exceeds `1e-3`.

`requirements.txt` uses floor pins such as `torch>=2.0.0`, and `run.log` records no versions. A newer torch can move dice by more than the tolerance for reasons that are not bugs. The gate must therefore give the operator a next step.

On failure, print:

1. A per-split table of published dice, cached dice, and the delta.
2. The five worst per-image dice deltas, with the image path for each.
3. The `environment_record()` of this session.
4. This line: rerun with `--accept-drift <tolerance> --reason "<text>"` when the deltas are uniform and small, and record the acceptance in `docs/DECISIONS.md`.

`--accept-drift` requires `--reason`. Both values land in `manifest.json` and in `inference.json` as `accepted_drift` and `drift_reason`. A row that carries `accepted_drift` must show a footnote marker in the summary cost table.

Uniform small deltas across every split point at a library version change. A large delta on one split, or a delta on a few images only, points at a real defect. Say that in the printed message.

### Manifest and sidecar

`manifest.json` fields: `cache_version`, `model`, `model_dir`, `seed`, `img_size`, `normalization`, `checkpoint_path`, `checkpoint_sha256_16`, `device_name`, `created_utc`, `git_rev`, `env`, `accepted_drift`, `drift_reason`, and `per_split: {split: {n_images, gpu_seconds}}`.

`results/<model_dir>/seed<N>/inference.json` fields: `device_name`, `n_images`, `gpu_seconds_total`, `gpu_seconds_per_image`, `batch_size`, `env`, `accepted_drift`, `source: "predict_cache"`.

Never modify the existing `metrics.json`.

**Mirror the sidecar to Drive.** After writing `inference.json` locally, copy it to `plan.drive_results_dir / "inference.json"`, guarded by `reporting.drive_available` exactly as `reporting.mirror_to_drive` does. `discover_metrics` reads whichever root supplies a row first, and merges only that directory's sibling sidecar. Without the mirror, a later session that reads from Drive loses every cost column.

### Files to modify

**`src/config.py`** — add `_DEFAULT_CACHE_ROOT = "cache"` and a `cache_root` resolver reading `cfg["ensemble"]["cache_dir"]`.

**`configs/base.yaml`** — append:

```yaml
ensemble:
  cache_dir: cache          # [TWEAK] per-image probability cache; large, gitignored
  weight_grid_step: 0.05    # [TWEAK] simplex grid resolution for fitted weights
cascade:
  box_padding: 5            # matches zeroshot.box_padding so the cascade and its ceiling agree
  detector_threshold: 0.5
```

**`.gitignore`** — add `/cache/`.

### Tests: `tests/test_ensemble_cache.py` (GPU-free)

1. Write then read returns probabilities within `1e-3`, ground truth exactly, paths exactly.
2. `pack_gt` and `unpack_gt` round trip for shape `(3, 5, 7)`, which is not a byte multiple.
3. `gt_fingerprint` changes when one pixel flips.
4. `assert_aligned` raises on a path mismatch and on a fingerprint mismatch.
5. `cache_path` produces `<root>/<model_dir>/seed<N>/<split>.npz`.
6. `environment_record` returns a dict whose values are all strings, with no exception when a package is absent.

### Run

Restore first, then nine cache builds: `{unet, sam_lora, sam_b}` times `{42, 43, 44}`. Estimated pure GPU compute is about 16 minutes. Copying three SAM ViT-H checkpoints from Drive dominates the session at 5 to 15 minutes each.

**Verify:** every one of the nine runs prints all five split deltas under `1e-3`, or carries a recorded `--accept-drift`. Stop here otherwise. A failure means the cache path does not reproduce the published evaluation, and every later number would be wrong.

---

## Phase 2: flat ensembles and summary integration

**Goal:** the two uniform rows land in `SUMMARY.md` with cost columns.
**Time:** 2 to 3 hours of coding, then 15 minutes of CPU.

### Files to create

**`src/ensemble/combine.py`** — numpy and stdlib. Imports `MetricTracker` lazily.

```python
def normalize_weights(weights) -> np.ndarray
    # rejects negatives and an all-zero vector; returns a vector summing to 1
def average_probs(stacks: list[np.ndarray], weights) -> np.ndarray
    # raises ValueError on a shape mismatch between members
def evaluate_stack(probs, gt, tracker_factory) -> dict
    # -> {dice, iou, mae, wfm, sm, em}
def calibration_stats(probs) -> dict
    # -> {"mean_prob": float, "frac_uncertain": float}  # frac in [0.05, 0.95]
def merge_backbones(values: list[str]) -> str    # "resnet34+vit_h+vit_b"
def merge_normalization(values: list[str]) -> str  # single value when all agree, else "mixed"
```

`merge_backbones` and `merge_normalization` derive the payload metadata from the member specs. Do not hardcode either string. All three ensemble members carry `imagenet` normalization in `MODEL_SPECS`, so both ensembles record `imagenet`, not `mixed`. The cascade mixes `imagenet` and `minmax`, so it records `mixed`. A future member change then cannot make the metadata lie.

**`ensemble_eval.py`** — repository root CLI.

```
python ensemble_eval.py --spec ens_unet_samh --seed 42 [--dry-run]
python ensemble_eval.py --self-check --member unet --seed 42
python ensemble_eval.py --all
```

`--self-check` builds a one-member ensemble with weight 1.0, evaluates it, and compares against `results/<member_dir>/seed<N>/metrics.json`. It prints per-split deltas and exits 1 above `1e-3`. It writes no `metrics.json`. Run it before any ensemble row.

A real run reads every member cache, calls `assert_aligned`, averages, evaluates, and writes `results/<spec>/seed<N>/metrics.json` plus `run.log` through `reporting.Tee`, then calls `reporting.mirror_to_drive`. Torch is never imported.

### Files to modify

**`src/config.py`** — add next to `MODEL_SPECS` and `ZEROSHOT_SPECS`, in the same registry style:

```python
@dataclass(frozen=True)
class EnsembleSpec:
    members: tuple[str, ...]   # model keys from MODEL_SPECS
    weighting: str             # "uniform" | "fitted"
    display: str

ENSEMBLE_SPECS = {
    "ens_unet_samh":        EnsembleSpec(("unet", "sam_lora"), "uniform", ...),
    "ens_top3":             EnsembleSpec(("unet", "sam_lora", "sam_b"), "uniform", ...),
    "ens_unet_samh_fitted": EnsembleSpec(("unet", "sam_lora"), "fitted", ...),
    "ens_top3_fitted":      EnsembleSpec(("unet", "sam_lora", "sam_b"), "fitted", ...),
}
```

Add `EnsemblePlan`, `build_ensemble_plan(cfg, key, seed, overrides)`, and `describe_ensemble_plan(plan)`. Resolve `member_dirs` with the existing `_checkpoint_name(model, _resolve_backbone(model, cfg))`, so `sam_lora` maps to `sam_vit_h`.

**`src/results_summary.py`** — six changes.

1. `MODEL_DISPLAY` gains all six new keys: `"Ensemble: U-Net + SAM-ViT-H (uniform)"`, `"Ensemble: U-Net + SAM-ViT-H (fitted)"`, `"Ensemble: top-3 (uniform)"`, `"Ensemble: top-3 (fitted)"`, `"Cascade: U-Net box -> MedSAM min-max"`, `"Cascade: GT box -> MedSAM min-max (path-matched ceiling)"`.
2. `TIER` gains explicit entries for all six. `tier_of` becomes:

```python
def tier_of(model_dir: str) -> str:
    if model_dir in TIER:
        return TIER[model_dir]
    if model_dir.startswith(("vanilla", "oracle")):
        return "oracle"
    if model_dir.startswith(("ens_", "casc_")):
        return "derived"
    return "prompt-free"
```

3. `split_by_tier` initializes three keys: `prompt-free`, `derived`, `oracle`.
4. `discover_metrics` merges a sibling `inference.json` into `payload["inference"]`, but only when the payload has no `inference` key. Read the sidecar from `mpath.parent`, the same directory that supplied the chosen row. New rows carry the block inline. Published rows keep their files untouched.
5. `flatten` **always** sets four new keys, defaulting to `None`: `infer_gpu_seconds_per_image`, `infer_n_images`, `unseen_dice_per_gpu_second`, `accepted_drift`. Setting them unconditionally is load-bearing, for this reason: `_write_csv` at `src/results_summary.py:322` derives its field names from `rows[0].keys()`, then restricts each `writerow` dict to those names. A key that appears only on a later row is **dropped without an error**. `DictWriter` never raises. The column simply vanishes from `summary_flat.csv` and from `summary.json`, and the cost table silently loses rows.
6. `_AGG_METRICS` gains `infer_gpu_seconds_per_image` and `unseen_dice_per_gpu_second`. `render_markdown` gains two sections and keeps the first heading byte-identical, because `tests/test_results_summary.py` asserts on its exact text.

New section order:

```
## Prompt-free (trained, mean ± std over seeds)      <- exact existing text, unchanged
## Derived methods (combinations of the models above; no ground truth at inference)
## Oracle-box baselines (not trained ...)             <- unchanged
## Inference cost (rows that recorded it)
## Per run (each model × seed)
```

The derived caption must state that these methods read no ground truth at inference, that they are fair on accuracy, that they cost more compute, and it must name the best single-model unseen mDice inline.

The cost table columns are: Model, Mean unseen mDice, GPU-s per image, Delta GPU-s versus reference, Cost multiple, Unseen Dice per GPU-second. Add `COST_REFERENCE_MODEL_DIR = "sam_vit_h"` as a module constant with a comment. Rows without an inference block are omitted from this table only. A row with `accepted_drift` gets a footnote marker.

Add `write_run_metrics(results_root, model_key, seed, payload)`. Make the existing `write_zeroshot_metrics` delegate to it with `seed=0`, so `05_benchmark.ipynb` keeps working.

Add `build_derived_payload(...)` producing the shape `flatten` reads. Field rules:

| Field | Value | Reason |
|---|---|---|
| `params.trainable` | sum over members | the ensemble costs all of them |
| `params.total` | sum over members | same |
| `checkpoint_size_mb` | sum over members | download footprint |
| `timing.total_seconds` | sum of member `timing.total_seconds` | renders as a correct "Train min" |
| `timing.epochs_run` | 0 | no training happened here |
| `best_val_dice` | `None` | no validation loop |
| `backbone` | `merge_backbones(member backbones)` | correct for two and three members alike |
| `normalization` | `merge_normalization(member normalizations)` | `imagenet` for both ensembles |

That mapping makes every existing cost column in the tier tables correct with no renderer change.

**`aggregate_results.py`** needs no change.

### Tests

`tests/test_ensemble_combine.py` (GPU-free):

1. `average_probs([p], [1.0])` returns `p` under `np.array_equal`.
2. Two members with uniform weights equals the arithmetic mean.
3. `normalize_weights` rejects a negative entry and an all-zero vector.
4. `average_probs` raises on a member shape mismatch.
5. `evaluate_stack` on a perfect prediction gives dice above 0.999.
6. `evaluate_stack` on a random `(5, 16, 16)` stack equals a manual `MetricTracker` loop on all six metrics.
7. `merge_normalization(["imagenet", "imagenet", "imagenet"]) == "imagenet"`, and `merge_normalization(["imagenet", "minmax"]) == "mixed"`.
8. `merge_backbones(["resnet34", "vit_h", "vit_b"]) == "resnet34+vit_h+vit_b"`.

`tests/test_results_summary.py` additions:

9. All six new keys are in `MODEL_DISPLAY`.
10. `tier_of("ens_top3") == "derived"`, `tier_of("casc_unet_medsam") == "derived"`, `tier_of("oracle_casc_gtbox_medsam") == "oracle"`.
11. A derived row never appears in the prompt-free section, mirroring the existing honesty test.
12. **Silent column loss.** Build a summary from one legacy row and one row carrying an inference block, in both orders. Assert that `summary_flat.csv` contains the `infer_gpu_seconds_per_image` header **and** the populated value, in both orders. Assert the same for `summary.json`. Do not assert that an exception is raised: the failure mode is a missing column, not an error.
13. `discover_metrics` merges a sidecar `inference.json` from the directory that supplied the chosen row, and does not overwrite an inline `inference` key.
14. The cost table lists only rows with an inference block, and it prints the delta against `sam_vit_h`.

### Run

1. `python ensemble_eval.py --self-check --member unet --seed 42`, then the same for `sam_lora` and `sam_b`.
2. `python ensemble_eval.py --spec ens_unet_samh --seed {42,43,44}`.
3. `python ensemble_eval.py --spec ens_top3 --seed {42,43,44}`.
4. `python aggregate_results.py`.

**Verify:** `SUMMARY.md` shows a derived table with two rows, three seeds each, and a cost table with `sam_vit_h` as reference. Ensemble A unseen mDice sits in 0.77 to 0.83. Read the calibration diagnostic before you call a 0.78 result a defect.

---

## Phase 3: weighted ensembles

**Goal:** fit member weights on seen data only, and report the delta against uniform.
**Time:** 1.5 to 2 hours of coding, then 15 minutes of CPU.

### Additions to `src/ensemble/combine.py`

```python
def fit_holdout_indices(n: int) -> tuple[list[int], list[int]]
    # contiguous: fit = range(0, n // 2), holdout = range(n // 2, n)

def simplex_grid(k: int, step: float) -> list[tuple[float, ...]]

@dataclass(frozen=True)
class WeightFit:
    weights: tuple[float, ...]
    fit_dice: float
    uniform_dice: float
    n_fit_images: int
    grid_step: float
    n_candidates: int

def fit_weights(member_stacks, gt, step=0.05, threshold=0.5) -> WeightFit
def build_fit_stack(plan, read_cache) -> tuple[list[np.ndarray], np.ndarray, dict]
```

`fit_holdout_indices` is contiguous, not interleaved. For Kvasir at n equal to 100 it returns indices 0 to 49 and 50 to 99. For ClinicDB at n equal to 62 it returns 0 to 30 and 31 to 61. The fit slice totals 81 images. `build_splits` sorts every test list, so the order is deterministic.

`fit_weights` maximizes the mean per-image Dice of the thresholded weighted average. It calls `dice_score` only, never the full metric suite. Two members give 21 candidates, three members give 231. Runtime is about 20 seconds.

Tie-break rule, in order: highest Dice, then smallest L2 distance to the uniform vector, then lexicographic. The uniform tie-break biases toward the null hypothesis, which pre-empts a cherry-picked-optimum objection.

`build_fit_stack` raises `ValueError` when any requested split key is outside `SEEN_SPLITS`. This is the leakage guard, in code, not in a comment.

### Additions to `ensemble_eval.py`

When `spec.weighting == "fitted"`:

1. Read the two seen split caches for every member.
2. Take the first contiguous half of each seen split as the fit slice.
3. Fit the weights.
4. Fit again on the second half, and record those weights as a stability check. The two halves now hold different video sequences, so agreement between them is evidence, not an artifact.
5. Evaluate the fitted ensemble on all five full splits, for table comparability.
6. Evaluate again on the second half of the two seen splits, and record it as `eval_seen_holdout`.
7. Record a threshold sensitivity sweep at 0.3, 0.4, 0.5, 0.6, 0.7, on the fit slice only. Headline numbers stay at 0.5.

The `ensemble` extras block:

```json
"ensemble": {
  "members": ["unet", "sam_lora"],
  "member_dirs": ["unet", "sam_vit_h"],
  "member_seeds": [42, 42],
  "weighting": "fitted",
  "weights": [0.35, 0.65],
  "uniform_weights": [0.5, 0.5],
  "threshold": 0.5,
  "fit": {
    "splits": ["seen_kvasir", "seen_clinicdb"],
    "index_rule": "first_contiguous_half",
    "n_fit_images": 81,
    "fit_dice": 0.9012,
    "uniform_fit_dice": 0.8994,
    "grid_step": 0.05,
    "n_candidates": 21,
    "stability_second_half_weights": [0.35, 0.65],
    "image_paths_sha256": "..."
  },
  "eval_seen_holdout": { "seen_kvasir": {}, "seen_clinicdb": {} },
  "calibration": { "unet": {}, "sam_vit_h": {} },
  "threshold_sensitivity_fit_slice": { "0.3": 0.89, "0.4": 0.90, "0.5": 0.90 }
}
```

### Tests: `tests/test_ensemble_weights.py` (GPU-free)

1. `fit_holdout_indices(100) == (list(range(50)), list(range(50, 100)))`.
2. `fit_holdout_indices(62) == (list(range(31)), list(range(31, 62)))`.
3. The two lists are disjoint and complete for n in 0 to 9, and every fit index is smaller than every holdout index.
4. `simplex_grid(2, 0.5) == [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]`. Every vector sums to 1 within `1e-9`.
5. Member A equals the ground truth and member B is noise. The fitted weight for A exceeds 0.8.
6. When two candidates tie on Dice, `fit_weights` returns the one closer to uniform.
7. **Leakage check.** Pass `build_fit_stack` a reader stub that raises on any split key outside `SEEN_SPLITS`. Assert that the fit completes and that the recorded fit split list equals the two seen splits.
8. `build_fit_stack` raises `ValueError` when a caller passes `cvc_colondb`.

### Run

`python ensemble_eval.py --spec ens_unet_samh_fitted --seed {42,43,44}`, the same for `ens_top3_fitted`, then `python aggregate_results.py`.

**Verify:** the derived table shows four rows. Report the fitted minus uniform delta on the held-out seen half and on the unseen mean, with the seed spread. If the two are within one seed standard deviation, report the null result plainly.

---

## Phase 4: box cascade

**Goal:** U-Net proposes a box, MedSAM segments inside it, no ground truth at inference.
**Time:** 2 to 3 hours of coding, then one GPU session of 45 to 60 minutes.

### File to create: `src/ensemble/cascade.py`

Pure numpy and scipy at module level. Torch and `segment_anything` import lazily inside the runner, mirroring `src/models/zeroshot.py`.

```python
def largest_component(binary: np.ndarray) -> np.ndarray | None
    # scipy.ndimage.label with an 8-connectivity structure.
    # Returns the largest component as a bool mask, or None for an empty input.
    # Ties go to the lowest label index, which np.argmax already gives.

def box_from_prediction(prob, threshold=0.5, padding=5) -> np.ndarray | None
    # threshold, then largest_component, then zeroshot.box_from_mask(component, padding)

def load_image_uint8(path, img_size) -> np.ndarray
    # PIL open, convert RGB, cv2.resize INTER_LINEAR. Matches A.Resize exactly.

@dataclass
class CascadeSplitResult:
    probs: np.ndarray     # (N,H,W) float32
    n_empty: int
    n_multiblob: int
    gpu_seconds: float

def run_cascade_split(zs, image_paths, detector_probs, gt, img_size,
                      box_padding, detector_threshold, empty_policy,
                      box_source, on_image=None) -> CascadeSplitResult
```

`box_from_prediction` reuses `zeroshot.box_from_mask` for the padding and clamping math. That import is torch-free.

`empty_policy` values:
- `"zero"` (default): write an all-zero probability map, do not call MedSAM. The method invents no detection.
- `"passthrough"`: copy the detector probability map unchanged.

`box_source` values:
- `"prediction"`: box from the cached detector probabilities.
- `"gt"`: box from the cached ground truth, for the ceiling row only.

The runner reads the detector probabilities from the Phase 1 cache. It never loads the U-Net checkpoint. The cascade therefore uses **the same float16-cached array** that the ensemble member uses. It does not use a live U-Net forward pass, and the cached values carry float16 rounding. A boundary pixel within about `1e-3` of 0.5 can flip, which can move a box edge by one pixel. With padding 5 the effect on the box is nil, and both cascade rows read the same source. State the provenance this way in `docs/DECISIONS.md`. Do not claim the detector output is byte-identical to a live U-Net run.

### File to modify: `src/models/zeroshot.py`

Refactor only. Extract the prompt-to-probability body of `predict_prob` into a new public method:

```python
def predict_prob_from_box(self, image_uint8, box, out_hw) -> np.ndarray
```

`predict_prob` keeps its signature and now derives the ground-truth box, then calls the new method. The published `vanilla_*` rows must stay reachable through an unchanged code path.

Add a regression test in `tests/test_zeroshot.py`: with a fake predictor, `predict_prob` calls `predict_prob_from_box` with the box that `box_from_mask` returns.

### File to modify: `src/config.py`

```python
@dataclass(frozen=True)
class CascadeSpec:
    detector: str | None      # model key, or None for the ceiling row
    segmenter: str            # a ZEROSHOT_SPECS key
    box_source: str           # "prediction" | "gt"
    empty_policy: str         # "zero" | "passthrough"
    tier: str                 # "derived" | "oracle"
    display: str

CASCADE_SPECS = {
    "casc_unet_medsam":         CascadeSpec("unet", "vanilla_medsam_minmax",
                                            "prediction", "zero", "derived", ...),
    "oracle_casc_gtbox_medsam": CascadeSpec(None, "vanilla_medsam_minmax",
                                            "gt", "zero", "oracle", ...),
}
```

`build_cascade_plan` enforces two invariants and raises `ValueError` otherwise:

- `box_source == "gt"` requires `tier == "oracle"`.
- `box_source == "prediction"` requires a detector and `tier == "derived"`.

This makes the fair-versus-ceiling boundary a code invariant, in the style the repository already uses for `reject_protocol_overrides`.

The segmenter weights and normalization come from the existing `ZEROSHOT_SPECS["vanilla_medsam_minmax"]` and `cfg["zeroshot"]["checkpoints"]`. Add no second source of MedSAM configuration.

### File to create: `cascade_eval.py`

```
python cascade_eval.py --spec casc_unet_medsam --seed 42 [--dry-run]
python cascade_eval.py --spec oracle_casc_gtbox_medsam
```

The ceiling row uses seed 0, like every other oracle row. It writes `metrics.json` with the `cascade` extras block:

```json
"cascade": {
  "detector": "unet", "detector_seed": 42,
  "detector_source": "float16 prediction cache",
  "segmenter": "vanilla_medsam_minmax",
  "box_source": "prediction", "empty_policy": "zero",
  "box_padding": 5, "detector_threshold": 0.5,
  "per_split": { "cvc_colondb": { "n_images": 380, "n_empty": 7, "n_multiblob": 91 } },
  "notes": "MedSAM's pretraining corpus included colonoscopy polyp data, so 'unseen' here means unseen by the U-Net detector only."
}
```

Payload cost fields: `params.trainable` is U-Net's 24,436,369, because MedSAM trains nothing. `params.total` is the sum of both. `timing.total_seconds` is U-Net's training time. `backbone` is `"resnet34->vit_b"`. `normalization` is `"mixed"`, which `merge_normalization` produces from `imagenet` and `minmax`. The ceiling row records backbone `"vit_b"` and normalization `"minmax"`. `inference.stage_gpu_seconds_per_image` names both stages separately.

### Tests: `tests/test_cascade.py` (GPU-free)

1. `largest_component` on an all-zero mask returns `None`.
2. Blobs of 9 and 4 pixels: only the 9-pixel blob comes back.
3. Two equal-size blobs: the lower label index wins, and the result repeats across runs.
4. Diagonally touching pixels join under 8-connectivity.
5. `box_from_prediction` on a single blob equals `box_from_mask` on that blob.
6. An all-below-threshold probability map returns `None`.
7. A small second blob does not enter the returned box.
8. Padding clamps to the image bounds.
9. `build_cascade_plan` raises when `box_source="gt"` and `tier="derived"`.
10. With a stub segmenter, `empty_policy="zero"` gives an all-zero map, `"passthrough"` gives the detector map, and both increment `n_empty`.

### Run

1. `python cascade_eval.py --spec oracle_casc_gtbox_medsam` first. About 10 minutes.
2. **Gate.** The result must land within 0.02 of the published 0.9246 unseen mDice. A larger deviation means the image loader differs from the published path. Fix that before running the fair row.
3. `python cascade_eval.py --spec casc_unet_medsam --seed {42,43,44}`. About 10 minutes each.
4. `python aggregate_results.py`.

**Verify:** the derived table gains one row with three seeds. Report `n_empty` per split. If `n_empty` exceeds 2 percent of any split, run the `passthrough` variant as a single documented ablation and report both.

---

## Phase 5: documentation

**Goal:** the numbers survive. `results/` is gitignored, so only `docs/` reaches the remote.
**Time:** 1 to 2 hours. No GPU.

1. Append one entry to `docs/DECISIONS.md`, newest last, following the existing format. Cover the pre-registered member sets, the first-contiguous-half fit slice and the video-sequence reason for it, the three-tier table split, the `zero` empty-box policy, the float16-cached detector provenance, and the reason the oracle-box-into-trained-models row does not exist. Record any accepted verification drift here too.
2. Add a section to `docs/FINDINGS.md` and bump `last_updated`. Transcribe the new rows, the fitted minus uniform delta with its seed spread, the cascade `n_empty` counts, and the cost table. Include this disclosure line for the cascade and the ceiling rows: MedSAM's pretraining corpus included colonoscopy polyp data, so "unseen" on those two rows means unseen by the U-Net detector only, the same caveat that already applies to the oracle tier.
3. Add the new modules and CLIs to the repository map in `README.md`, and add the three new commands to the CLI Usage block.
4. Create `notebooks/08_ensemble.ipynb`. Step 0 restores the Drive results mirror. The remaining cells run Phase 1, 2, 3, and 4 end to end in one Colab session, in the shape of `train_colab.ipynb`: one knob cell, one environment cell, then the run cells. The cache must not outlive the session, so the whole chain runs together.
5. Do not touch `docs/SPEAKER_SCRIPT.md`, `docs/PRESENTATION_OUTLINE.md`, `docs/PROJECT_REPORT.md`, or `make_figures.py`. `make_figures.py` reads `summary_by_model.csv` through `csv.DictReader` keyed by display name, so new rows and new columns do not reach it.

---

## Cost accounting

Definition, stated once in `docs/FINDINGS.md`: per-image GPU seconds is the measured forward time divided by the image count, at batch size 8, on one A100, with `torch.cuda.synchronize()` around the timed region. Model construction, checkpoint load, and disk write are excluded. Do not extrapolate to another batch size.

| Configuration | GPU-s per image (estimate) | Source |
|---|---|---|
| U-Net | 0.003 | cache manifest |
| SAM-ViT-B + LoRA | 0.008 | cache manifest |
| SAM-ViT-H + LoRA (reference) | 0.035 | cache manifest |
| Ensemble A | 0.038 | sum of members |
| Ensemble B | 0.046 | sum of members |
| Cascade | 0.75 | U-Net plus MedSAM through `SamPredictor` |

The cascade is about 20 times the reference cost per image, because `SamPredictor.set_image` encodes one image at a time at 1024x1024. Report that plainly.

Report `combine_cpu_seconds_per_image` in a separate field. State in the caption that the cache is an experiment artifact, not a deployment trick. A deployed ensemble pays the full sum of member GPU seconds.

**Total added compute for the whole experiment: about 60 minutes of A100 time.** One seed of SAM-ViT-H training took 120 minutes. The entire experiment costs less GPU time than half of one training run.

Wall clock per phase: Phase 0, 20 to 40 minutes. Phase 1, 2 to 3 hours of coding plus a 45 to 75 minute session. Phase 2, 2 to 3 hours plus 15 minutes. Phase 3, 1.5 to 2 hours plus 15 minutes. Phase 4, 2 to 3 hours plus a 45 to 60 minute session. Phase 5, 1 to 2 hours. Total about 10 to 14 hours of work.

---

## Risks, ranked

1. **Checkpoints or base backbones missing.** Three SAM ViT-H seeds need 7.6 GB, and the base `.pth` files add 3.2 GB. Phase 0 gates on all of them. Fallback: drop to U-Net plus SAM-ViT-B and record the change.
2. **A fresh clone has no `results/`.** Verification, the sidecars, and the cost table all read published rows. Phase 1 step 0 restores the Drive mirror first, and `predict_cache.py` mirrors each sidecar back to Drive. Both directions are needed, because `discover_metrics` merges only the sidecar next to the row it chose.
3. **Colab session loss mid-experiment.** The 2 GB cache lives on session disk. Mitigation: one notebook runs the whole chain; every `metrics.json` mirrors to Drive at once; `--cache-dir` accepts a Drive path.
4. **Library version drift breaks the verification gate.** `requirements.txt` uses floor pins and `run.log` records no versions. Mitigation: `inference.json` records the environment, the gate prints per-split and per-image deltas, and `--accept-drift` with a mandatory `--reason` gives a documented override.
5. **Calibration mismatch between architectures.** U-Net saturates its sigmoid harder than the LoRA models, so a uniform average follows U-Net. Mitigation: report the calibration diagnostic, let the fitted arm absorb the mismatch, and expect 0.77 to 0.83 rather than a guaranteed gain. Do not add temperature scaling; it would need more fitted parameters and more seen-data fitting.
6. **Fit-slice leakage through video frames.** CVC-ClinicDB comes from 29 video sequences. The contiguous split separates sequences, which an interleaved split does not. The trade is some distribution difference between the halves. The primary endpoint stays the untouched unseen sets.
7. **float16 rounding in the cache.** Thresholded metrics are almost immune. MAE, Sm, and Em can move in the fourth decimal. A boundary pixel can flip and move a box edge by one pixel, which padding 5 absorbs. The `1e-3` verification tolerance covers the metric side. State both in the docs.
8. **Overwriting a published row.** Never write into `results/unet`, `results/sam_vit_h`, `results/sam_vit_b`, or any `vanilla_*` directory, except the new `inference.json` sidecar. `predict_cache.py` must refuse any other write there.
9. **MedSAM prompt API.** The cascade reuses `SamPredictor` exactly as `vanilla_medsam_minmax` did, including the min-max path through `_identity_preprocess`. Box coordinates stay in the 352x352 frame, because `set_torch_image` records that original size. The published 0.9246 row is the evidence the path works. The Phase 4 ceiling gate re-proves it.
10. **Small or absent gains.** Members share training data, loss, and resolution, so their errors correlate. Realistic ensemble gain is +0.01 to +0.03 unseen mDice, and a small loss is possible. The cascade is capped by U-Net detection. Both null results are pre-registered as reportable.
11. **Silent CSV column loss.** `_write_csv` restricts every row to `rows[0]`'s field names. A key present only on later rows disappears from `summary_flat.csv` and `summary.json` with no error. `flatten` must set every new key unconditionally, and the Phase 2 test asserts the column survives in both row orders.
12. **No repository map.** The project has no root `CLAUDE.md`. Out of scope for this plan. Raise it separately.

---

## Run commands, in order

```bash
cd /content/msu2026summer_final_project
python -m pytest tests/ -q                                   # all phases, GPU-free
# notebook 08 step 0: copy Drive results mirror into ./results/
python predict_cache.py --inventory                          # Phase 0 gate
python predict_cache.py --model unet     --seed 42           # Phase 1, repeat 9 times
python ensemble_eval.py --self-check --member unet --seed 42 # Phase 2 gate
python ensemble_eval.py --spec ens_unet_samh --seed 42       # Phase 2, repeat per spec and seed
python ensemble_eval.py --spec ens_unet_samh_fitted --seed 42 # Phase 3
python cascade_eval.py --spec oracle_casc_gtbox_medsam       # Phase 4 gate
python cascade_eval.py --spec casc_unet_medsam --seed 42     # Phase 4
python aggregate_results.py                                  # regenerate SUMMARY.md
```

Start with Phase 0. Open a Colab session, mount Drive, and copy `msu2026_checkpoints/results/` into `./results/`.

---

## Review changelog

Nine findings from the adversarial review, all resolved.

1. **Fresh clone has no `results/` (blocking).** Added Phase 1 step 0 (Drive results restore), a directed failure message in `predict_cache.py`, and the `inference.json` mirror back to Drive.
2. **Parity fit split leaks video frames (blocking).** Replaced even/odd with the first contiguous half in sorted path order; the stability refit now uses the second half, which holds different sequences.
3. **Inventory gate omitted base artifacts (blocking).** The gate now checks the three base `.pth` files, the `segment_anything` package, `verify_splits` counts, trained checkpoints, published metrics, and free disk.
4. **No failure procedure for the verify gate.** Added `environment_record()`, a four-item failure printout, and `--accept-drift TOL --reason TEXT` with a cost-table footnote.
5. **CSV failure mode misstated.** The real failure is silent column loss, not an exception; the test asserts the column survives in both row orders.
6. **"Byte-identical detector output" was false.** Restated as float16-cached detector output; a one-pixel box-edge flip is absorbed by padding 5.
7. **Derived payload metadata was wrong.** `merge_backbones` and `merge_normalization` now derive both fields from member specs; ensembles record `imagenet`, the cascade records `mixed`.
8. **"Unseen" disclosure for the cascade.** MedSAM's pretraining corpus included colonoscopy polyps, so "unseen" on the cascade and ceiling rows means unseen by the detector only. Stated in FINDINGS.md and in the row's `notes`.
9. **Expected-range false alarm.** Widened the uniform expectation to 0.77 to 0.83; 0.75 stays the only suspect-a-bug line.
