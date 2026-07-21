---
status: living
last_updated: 2026-07-21
---

<!-- FILE MAP | Project plan & pick-up point: what's done, what's missing, and the exact next
     steps. Read this first when resuming work. Pairs with docs/DECISIONS.md (why choices were
     made) and CLAUDE.md (repo map). Update the status table + checklist as items land.
     Doc class: living — bump last_updated above on any substantive change (see DECISIONS.md). -->

# Project Plan — Where to Pick Up

**Study:** Can LoRA-adapted SAM (and MedSAM) match specialist polyp networks on the PraNet
benchmark while degrading less on unseen datasets — and at what training / model-size / compute cost?

**Last updated:** 2026-07-21

---

## TL;DR — where we are

**Results are complete.** All three seeds (42, 43, 44) are trained for the four prompt-free models
(U-Net, SAM ViT-H + LoRA, SAM ViT-B + LoRA, MedSAM ViT-B + LoRA), the two oracle-box baselines
(vanilla SAM, vanilla MedSAM) are run, `aggregate_results.py` has consolidated every run's
`metrics.json` into `results/summary/`, and the write-up is in **`docs/FINDINGS.md`** (source of
truth for all numbers — read that first for the full picture, including per-split behavior, the
backbone-vs-pretraining ablation, and the oracle-box caveat).

**Headline (from FINDINGS.md):** among the prompt-free models, **SAM ViT-H + LoRA generalizes best
to unseen data (0.806 mean unseen mDice vs U-Net's 0.755)** while training only **~3%** of U-Net's
parameters (830,177 vs 24.4M). The oracle-box vanilla SAM/MedSAM scores are a ceiling check (negative
generalization gap), not a win over fine-tuning — see FINDINGS.md for detail.

The runnable, chart-bearing readout is `notebooks/07_report.ipynb` (the earlier
`notebooks/06_findings.ipynb` remains as the original two-tier illustration).

**Pick up at:** [Remaining work](#remaining-work) — professor sign-off on the oracle-box prompt
protocol is the only open item; results themselves are done.

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
| Report **how big** each model is | Done | `metrics.json` params + `checkpoint_size_mb`; benchmark prints a per-model param table |
| **Cost**: how long to train, what HW | Done | `metrics.json → timing` + `device_name` |
| Compare **with fine-tuning and without** | Done | Zero-shot path (`src/models/zeroshot.py`) wired into `05_benchmark.ipynb`; results in `docs/FINDINGS.md` |
| 4-way: vanilla SAM, vanilla MedSAM, fine-tuned SAM, fine-tuned MedSAM | Done | All trained + oracle models compared; results in `docs/FINDINGS.md` |

Every requirement is implemented and run. See `docs/FINDINGS.md` for the numbers.

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
- **Benchmark wired to the full comparison** — `notebooks/05_benchmark.ipynb` builds and compares
  the four trained prompt-free models (U-Net, SAM+LoRA, SAM-ViT-B+LoRA, MedSAM+LoRA) plus vanilla
  SAM and vanilla MedSAM (oracle-box, zero-shot): param/size table, 5-split metrics, seen-vs-unseen
  bars, generalization gap, parameter-efficiency scatter (zero-shot drawn at 0 trainable),
  qualitative panels. Prompt protocol is swappable at the top of the model-build cell (`ZS_PROMPT`,
  `ZS_BOX_PAD`). Full results consolidated in `docs/FINDINGS.md`.
- **Historical per-model notebooks** (`02`–`04`) preserved on `backup/per-model-notebooks`.

---

## Remaining work

Results are complete — every model × seed pair is trained, benchmarked, and rolled up in
`docs/FINDINGS.md`. What's left is not GPU work:

1. **Professor sign-off on the oracle-box prompt protocol** for the vanilla SAM/MedSAM baselines
   (see [Open decision](#open-decision-does-not-block-training) below).
2. **Mark `docs/FINDINGS.md` `immutable`** once that sign-off lands (per the doc-class convention in
   CLAUDE.md / DECISIONS.md) — it is deliberately left `living` until then.

No further training runs, aggregator runs, or code changes are anticipated.

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

**Compare models:** once checkpoints exist, run `notebooks/05_benchmark.ipynb`. For the consolidated,
already-computed results, read `docs/FINDINGS.md` (or run `notebooks/07_report.ipynb`) directly.

**Tests:** `pytest tests/ -q` (GPU-free; 34 tests incl. the zero-shot prompt math).

---

## Task checklist

- [x] Implement zero-shot vanilla SAM + MedSAM inference wrappers (GT-box prompted)
- [x] GPU-free unit tests for prompt derivation
- [x] Extend `05_benchmark.ipynb` to the four-way (2×2) comparison + U-Net
- [x] Train U-Net, fine-tuned SAM ViT-H, fine-tuned MedSAM ViT-B at **seed 42** → metrics.json
- [x] Run `05_benchmark.ipynb` → initial multi-model comparison (seed 42); results read out in `06_findings.ipynb`
- [x] Multi-seed support in trainer + benchmark aggregation (mean ± std)
- [x] Add `sam_b` (SAM ViT-B + LoRA) to isolate the MedSAM backbone confound
- [x] Results aggregator (`aggregate_results.py` → `results/summary/`)
- [x] Train seeds **43 and 44** for the prompt-free models (T4/L4) → fills mean ± std
- [x] Train `sam_b` (SAM ViT-B + LoRA), T4 → the backbone-confound row
- [x] Re-run `aggregate_results.py` to absorb the new rows
- [x] Write up accuracy-vs-cost findings in `docs/FINDINGS.md`
- [ ] Confirm prompting protocol with the professor (default = `box`)
- [ ] Mark `docs/FINDINGS.md` immutable on publish (after professor sign-off)
