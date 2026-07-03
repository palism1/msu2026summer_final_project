<!-- FILE MAP | Project plan & pick-up point: what's done, what's missing, and the exact next
     steps. Read this first when resuming work. Pairs with docs/DECISIONS.md (why choices were
     made) and CLAUDE.md (repo map). Update the status table + checklist as items land. -->

# Project Plan — Where to Pick Up

**Study:** Can LoRA-adapted SAM (and MedSAM) match specialist polyp networks on the PraNet
benchmark while degrading less on unseen datasets — and at what training / model-size / compute cost?

**Last updated:** 2026-07-02

---

## TL;DR — where we are

**All the code is done, including the vanilla-vs-fine-tuned comparison. The only thing left is to
run it on a GPU.** The four-way comparison the professor asked for (vanilla SAM, vanilla MedSAM,
fine-tuned SAM, fine-tuned MedSAM, plus U-Net as a reference) is fully wired into
`notebooks/05_benchmark.ipynb`; the zero-shot baselines are implemented in `src/models/zeroshot.py`
and unit-tested. Nothing else needs to be written.

**Pick up at:** [Run & train](#run--train-the-only-remaining-work) — train the three models, then run
the benchmark. That produces every number (accuracy, size, training time, hardware) for all five models.

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

1. **Train all three models in one Colab session** — open `notebooks/train_colab.ipynb`, set
   `MODELS`/`SEED`/`EPOCHS` in **cell 1** (no `run.yaml` edits, no commit/push), Run all.
   U-Net + MedSAM fit a T4; SAM ViT-H wants an L4/A100 — either run everything on an L4, or run
   `['unet', 'medsam']` on a T4 first and `['sam_lora']` on an L4 later (the notebook skips
   models that already finished; checkpoints + results are auto-mirrored to Drive during training,
   so a disconnect never loses a completed run).
2. **Run the benchmark** — `notebooks/05_benchmark.ipynb` on an L4/A100 (it loads a second ViT-H for
   zero-shot, ~2.5 GB extra VRAM). Out comes the full five-model comparison and the accuracy-vs-cost story.

The vanilla SAM/MedSAM baselines need **no training** — the benchmark downloads the base weights and
runs them zero-shot. Their "cost" is 0 training time, which is the point of the comparison.

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

- [x] Implement zero-shot vanilla SAM inference wrapper
- [x] Implement zero-shot vanilla MedSAM inference wrapper (same protocol)
- [x] GPU-free unit tests for prompt derivation
- [x] Extend `05_benchmark.ipynb` to the four-way (2×2) comparison + U-Net
- [ ] Confirm prompting protocol with the professor (default = `box`)
- [ ] Train U-Net baseline (T4) → checkpoint + metrics.json
- [ ] Train fine-tuned MedSAM ViT-B (T4) → checkpoint + metrics.json
- [ ] Train fine-tuned SAM ViT-H (L4 / A100) → checkpoint + metrics.json
- [ ] Run `05_benchmark.ipynb` → five-model comparison
- [ ] Write up accuracy-vs-cost findings (size, training time, hardware needed)
