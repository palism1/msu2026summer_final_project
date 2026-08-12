---
status: living
last_updated: 2026-08-12
---

<!-- FILE MAP | Project plan & pick-up point: what's done, what's missing, and the exact next
     steps. Read this first when resuming work. Pairs with docs/DECISIONS.md (why choices were
     made) and README.md (repo map). Update the status table + checklist as items land.
     Doc class: living — bump last_updated above on any substantive change (see DECISIONS.md). -->

# Project Plan — Where to Pick Up

**Study:** Can LoRA-adapted SAM (and MedSAM) match specialist polyp networks on the PraNet
benchmark while degrading less on unseen datasets — and at what training / model-size / compute cost?

**Last updated:** 2026-08-12

---

## TL;DR — where we are

**All runs are complete.** Six trained models (U-Net, SAM ViT-H + LoRA, SAM ViT-B + LoRA, MedSAM
ViT-B + LoRA, MedSAM min-max, MedSAM ctrl; three seeds each) and four oracle-box baselines (vanilla
SAM, vanilla MedSAM, vanilla SAM-B, vanilla MedSAM min-max) all have numbers in
`results/summary/SUMMARY.md` (20 runs, regenerated 2026-08-12). See `docs/FINDINGS.md` for the
write-up.

Headline unseen mDice, three-seed means: SAM-ViT-H + LoRA 0.806, SAM-ViT-B + LoRA 0.761, U-Net
0.755, MedSAM + LoRA 0.661, MedSAM min-max 0.725, MedSAM ctrl 0.655. Oracle-box baselines (untrained):
vanilla MedSAM min-max 0.925, vanilla SAM 0.905, vanilla SAM-B 0.880, vanilla MedSAM 0.845.

**Resolved (was open, 2026-07-28, `docs/MEDSAM_INVESTIGATION.md`):** MedSAM's original deficit was
partly attributed to "medical pretraining is a net drag." Part of that deficit was a normalization
bug: the training pipeline fed MedSAM's frozen encoder ImageNet-standardized inputs, but MedSAM was
fit on per-image `[0,1]` min-max inputs. The fix is in code: normalization and augmentation policy are
now bound to the model key (`src/config.MODEL_SPECS`). Three experiments isolated the confound and are
now complete:

- **A — vanilla MedSAM under its own preprocessing** (`vanilla_medsam_minmax`). Done.
- **B — vanilla SAM-ViT-B oracle-box baseline** (`vanilla_sam_b`). Done. Fills the last cell of the 2x2.
- **C — MedSAM + LoRA retrained under a three-arm design** (`medsam_minmax` + `medsam_ctrl`, 3 seeds
  each). Done. `medsam_ctrl` is an augmentation-matched control, needed because min-max normalization
  is exactly invariant to ColorJitter's brightness/contrast (see `docs/MEDSAM_INVESTIGATION.md`, H1
  addendum).

The normalization bug explains part of the original gap: MedSAM min-max recovers +0.070 unseen Dice
over MedSAM + LoRA (0.655 to 0.725). The remaining −0.036 against generic SAM at the same backbone is
the true pretraining effect.

**Pick up at:** presentation prep. See `docs/PRESENTATION_OUTLINE.md` for the current deck plan.

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
| Compare **with fine-tuning and without** | Done | Zero-shot path (`src/models/zeroshot.py`); results in `results/summary/SUMMARY.md` |
| 4-way: vanilla SAM, vanilla MedSAM, fine-tuned SAM, fine-tuned MedSAM | Done | All models and baselines have run; see `docs/FINDINGS.md` |

Every requirement is implemented and has numbers.

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

## Run & train — complete

All training and baseline runs are done. The six-model comparison (U-Net, SAM-ViT-H + LoRA,
SAM-ViT-B + LoRA, MedSAM-ViT-B + LoRA, MedSAM min-max, MedSAM ctrl; three seeds each) plus the four
oracle-box baselines are in `results/summary/SUMMARY.md`. See `docs/FINDINGS.md` for the write-up and
`docs/MEDSAM_INVESTIGATION.md` for what each comparison answers.

Experiments A, B, C (isolating the MedSAM normalization confound) are also complete:

1. **A — vanilla MedSAM under its own preprocessing.** Done.
2. **B — vanilla SAM-ViT-B oracle-box baseline.** Done.
3. **C — MedSAM + LoRA retrained under the three-arm design** (3 seeds x 2 arms). Done.
4. **Consolidated** — `python aggregate_results.py` (regenerated 2026-08-12).

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
- [x] Run experiment A — `vanilla_medsam_minmax`
- [x] Run experiment B — `vanilla_sam_b`
- [x] Run experiment C — `medsam_minmax` + `medsam_ctrl`, 3 seeds each
- [x] Re-run `aggregate_results.py` to absorb the new rows
- [x] Update the `docs/FINDINGS.md` normalization caveat with the A/B/C results
