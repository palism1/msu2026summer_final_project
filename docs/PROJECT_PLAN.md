---
status: living
last_updated: 2026-07-28
---

<!-- FILE MAP | Project plan & pick-up point: what's done, what's missing, and the exact next
     steps. Read this first when resuming work. Pairs with docs/DECISIONS.md (why choices were
     made) and CLAUDE.md (repo map). Update the status table + checklist as items land.
     Doc class: living — bump last_updated above on any substantive change (see DECISIONS.md). -->

# Project Plan — Where to Pick Up

**Study:** Can LoRA-adapted SAM (and MedSAM) match specialist polyp networks on the PraNet
benchmark while degrading less on unseen datasets — and at what training / model-size / compute cost?

**Last updated:** 2026-07-28

---

## TL;DR — where we are

**Full results are in** (U-Net, SAM ViT-H + LoRA, SAM ViT-B + LoRA, MedSAM ViT-B + LoRA, three seeds
each, plus the two published oracle-box baselines) — see `docs/FINDINGS.md`. Among the *prompt-free*
models, SAM-ViT-H + LoRA generalizes best (0.806 unseen mDice vs U-Net 0.755) at 0.4% of the
parameters; MedSAM-ViT-B + LoRA trails every model, including the smaller-backbone SAM-ViT-B + LoRA.

**Open question raised in review (2026-07-28, `docs/MEDSAM_INVESTIGATION.md`):** MedSAM's deficit was
partly attributed to "medical pretraining is a net drag," but the training pipeline fed MedSAM's
frozen encoder ImageNet-standardized inputs when MedSAM was fit on per-image `[0,1]` min-max inputs —
a normalization bug (H1) that could account for some or all of the gap. Fixed in code: normalization
and augmentation policy are now bound to the model key (`src/config.MODEL_SPECS`), and three new
experiments isolate the confound:

- **A — vanilla MedSAM under its own preprocessing** (`vanilla_medsam_minmax`, no training, ~2 min).
- **B — vanilla SAM-ViT-B oracle-box baseline** (`vanilla_sam_b`, no training, ~2 min) — fills the
  missing cell of the 2x2.
- **C — MedSAM + LoRA retrained under a three-arm design** (`medsam_minmax` + `medsam_ctrl`, 3 seeds
  each, ~95 min A100 total) — `medsam_ctrl` is an augmentation-matched control, needed because
  min-max normalization is exactly invariant to ColorJitter's brightness/contrast (see
  `docs/MEDSAM_INVESTIGATION.md`, H1 addendum).

None of A/B/C have GPU numbers yet. Published paths (`checkpoints/medsam/`, `results/medsam/`,
`results/vanilla_sam/`, `results/vanilla_medsam/`) are untouched — the new experiments write to new
model keys / baseline names only.

**Pick up at:** [Run & train](#run--train-the-only-remaining-work) — run experiments A, B, then C on
Colab (see `docs/MEDSAM_INVESTIGATION.md` "How to run these" for exact commands), re-run
`aggregate_results.py`, then update the caveat in `docs/FINDINGS.md`.

---

## Professor's requirements — status

The professor's proposed change: *figure out fine-tuning for each model, add dataset + extra
structure, compare each model with fine-tuning and without, run each on the same GPU, form a
comparison across vanilla SAM, vanilla MedSAM, fine-tuned SAM and fine-tuned MedSAM, report how big
each model is, and look into costs (training time, hardware needed).*

| Requirement | Status | Where / note |
|---|---|---|
| Fine-tuning for each model | Done | LoRA on SAM ViT-H and MedSAM ViT-B; U-Net trained as the from-scratch specialist |
| Add dataset + "extra structure" | Done | PraNet 5-split protocol; CNN mask decoder on SAM/MedSAM (`src/models/sam_adapter.py`) |
| Run each on the **same GPU** | Done | `train.py` one protocol; `metrics.json` records `device` / `device_name` |
| Report **how big** each model is | Done | `metrics.json` params + `checkpoint_size_mb`; benchmark prints a 5-model param table |
| **Cost**: how long to train, what HW | Done | `metrics.json → timing` + `device_name` |
| Compare **with fine-tuning and without** | **Code done — awaiting run** | Zero-shot path built (`src/models/zeroshot.py`); wired into `05_benchmark.ipynb` |
| 4-way: vanilla SAM, vanilla MedSAM, fine-tuned SAM, fine-tuned MedSAM | **Code done — awaiting run** | All 5 models (2×2 + U-Net) in the benchmark's `all_models`; runs once checkpoints exist |

Every requirement is implemented. The last two now just need a GPU run to produce the actual numbers.

---

## What's done

- **Config-driven runner** — `train.py` + `configs/run.yaml` + `src/config.py` + `src/training/`.
  Pick a model with `run.model`; every model trains identically. `--dry-run` validates offline.
- **Efficiency metrics** — `results/<model>/seed<seed>/metrics.json`: accuracy (5 splits + gap),
  params, checkpoint size, device, per-epoch + total training time. Overlays + `run.log`; Drive-mirrored.
- **Colab wrapper** — `notebooks/train_colab.ipynb` (3 cells: mount Drive → pull code + data +
  checkpoint → `train.py`).
- **Zero-shot vanilla baselines** *(new)* — `src/models/zeroshot.py`: raw SAM / MedSAM run
  inference-only, prompted by the GT bounding box (no LoRA, no training, 0 trainable params).
  Pure prompt-derivation (`box_from_mask` / `point_from_mask`) is GPU-free and unit-tested
  (`tests/test_zeroshot.py`, 9 tests). `src/models/__init__.py` is lazy so these import without torch.
- **Benchmark wired to the full 4-way** *(new)* — `notebooks/05_benchmark.ipynb` now builds and
  compares **five** models (U-Net, SAM+LoRA, MedSAM+LoRA, vanilla SAM, vanilla MedSAM): param/size
  table, 5-split metrics, seen-vs-unseen bars, generalization gap, parameter-efficiency scatter
  (zero-shot drawn at 0 trainable), qualitative panels. Prompt protocol is swappable at the top of
  the model-build cell (`ZS_PROMPT`, `ZS_BOX_PAD`).
- **Historical per-model notebooks** (`02`–`04`) preserved on `backup/per-model-notebooks`.

---

## Run & train — the only remaining work

Everything below needs a GPU; no more code changes are required.

The original five-model comparison (U-Net, SAM-ViT-H + LoRA, SAM-ViT-B + LoRA, MedSAM-ViT-B + LoRA,
three seeds each, plus the two published oracle-box baselines) is **done** — see `docs/FINDINGS.md`.
What's left is experiments A, B, C from `docs/MEDSAM_INVESTIGATION.md` (~190 min A100 total,
~2.1 GB Drive), run in that order since A determines whether C is a correction or an ablation:

1. **A — vanilla MedSAM under its own preprocessing** (no training, ~2 min):
   `python zeroshot_eval.py --config configs/run.yaml --baseline vanilla_medsam_minmax`.
2. **B — vanilla SAM-ViT-B oracle-box baseline** (no training, ~2 min):
   `python zeroshot_eval.py --config configs/run.yaml --baseline vanilla_sam_b`.
3. **C — MedSAM + LoRA retrained under the three-arm design** (3 seeds x 2 arms, ~95 min A100):
   train `medsam_minmax` and `medsam_ctrl` together at seeds 42/43/44, via
   `notebooks/train_colab.ipynb` (`MODELS = ['medsam_minmax', 'medsam_ctrl']`) or
   `python train.py --config configs/run.yaml --model <medsam_minmax|medsam_ctrl> --seed <N>`.
4. **Consolidate** — `python aggregate_results.py` (no GPU, no notebook re-run).

Exact commands and what each comparison answers: `docs/MEDSAM_INVESTIGATION.md` → "How to run
these". All three write to new model keys / baseline names — `checkpoints/medsam/`,
`results/medsam/`, `results/vanilla_sam/`, `results/vanilla_medsam/` are untouched by design.

---

## Open decision (does NOT block training)

**Prompting protocol for the vanilla baselines.** Default is **`box`** (bounding box from the GT
mask) — recommended, standard in the medical-SAM literature, and MedSAM is box-prompt-trained. It is
applied identically to vanilla SAM and vanilla MedSAM so they stay comparable. Caveat to document: it
is an *oracle* prompt (reveals roughly where the polyp is) while the fine-tuned models get no hint.
To change it, edit `ZS_PROMPT` (`box` | `point`) / `ZS_BOX_PAD` at the top of the model-build cell in
`05_benchmark.ipynb` — no other code changes. Worth a quick professor sign-off before the final run,
since it defines what the "without fine-tuning" numbers mean.

---

## How to run (reference)

**Train (Colab):** open `train_colab.ipynb`, set `MODELS` / `SEED` / `EPOCHS` in cell 1, Run all.
Already-trained models are restored from Drive and skipped; new checkpoints/results are mirrored to
Drive automatically. **Train (local CLI):** `python train.py --config configs/run.yaml --model M`.
Offline check: `python train.py --config configs/run.yaml --dry-run`.

**Compare models:** once the three checkpoints exist, run `notebooks/05_benchmark.ipynb`.

**Tests:** `pytest tests/ -q` (GPU-free; 24 tests incl. the zero-shot prompt math).

---

## Task checklist

- [x] Implement zero-shot vanilla SAM + MedSAM inference wrappers (GT-box prompted)
- [x] GPU-free unit tests for prompt derivation
- [x] Extend `05_benchmark.ipynb` to the four-way (2×2) comparison + U-Net
- [x] Train U-Net, fine-tuned SAM ViT-H, fine-tuned MedSAM ViT-B at **seed 42** → metrics.json
- [x] Run `05_benchmark.ipynb` → five-model comparison (seed 42); results read out in `06_findings.ipynb`
- [x] Multi-seed support in trainer + benchmark aggregation (mean ± std)
- [x] Add `sam_b` (SAM ViT-B + LoRA) to isolate the MedSAM backbone confound
- [x] Results aggregator (`aggregate_results.py` → `results/summary/`)
- [x] Confirm prompting protocol with the professor (default = `box`)
- [x] Train seeds **43 and 44** for the prompt-free models → fills mean ± std
- [x] Train `sam_b` (SAM ViT-B + LoRA) → the backbone-confound row
- [x] Write up accuracy-vs-cost findings in `docs/FINDINGS.md`
- [x] Diagnose the MedSAM normalization bug (H1) and land the fix in code
  (`src/normalization.py`, `src/config.MODEL_SPECS`, `zeroshot_eval.py`) — `docs/DECISIONS.md`,
  `docs/MEDSAM_INVESTIGATION.md`
- [ ] Run experiment A — `vanilla_medsam_minmax` (no training, ~2 min)
- [ ] Run experiment B — `vanilla_sam_b` (no training, ~2 min)
- [ ] Run experiment C — `medsam_minmax` + `medsam_ctrl`, 3 seeds each (~95 min A100)
- [ ] Re-run `aggregate_results.py` to absorb the new rows
- [ ] Update the `docs/FINDINGS.md` normalization caveat with the A/B/C results; retract or confirm
  the "medical pretraining is a net drag" line depending on what `medsam_ctrl` -> `medsam_minmax`
  shows
