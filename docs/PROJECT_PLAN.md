<!-- FILE MAP | Project plan & pick-up point: what's done, what's missing, and the exact next
     steps. Read this first when resuming work. Pairs with docs/DECISIONS.md (why choices were
     made) and CLAUDE.md (repo map). Update the status table + checklist as items land. -->

# Project Plan — Where to Pick Up

**Study:** Can LoRA-adapted SAM (and MedSAM) match specialist polyp networks on the PraNet
benchmark while degrading less on unseen datasets — and at what training / model-size / compute cost?

**Last updated:** 2026-07-02

---

## TL;DR — where we are

The training and measurement **infrastructure is complete**: one config-driven runner trains any
model under an identical protocol on the same GPU, and each run records accuracy, model size,
checkpoint footprint, device, and wall-clock training time. The **missing piece is the
vanilla-vs-fine-tuned comparison** — there is currently no zero-shot (un-fine-tuned) SAM or MedSAM
anywhere in the code, so two of the professor's four comparison cells don't exist yet.

**Pick up at:** [Remaining work](#remaining-work-pick-up-here) → build the two zero-shot baselines
and extend `notebooks/05_benchmark.ipynb` to the full four-way comparison.

---

## Professor's requirements — status

The professor's proposed change: *figure out fine-tuning for each model, add dataset + extra
structure, compare each model with fine-tuning and without, run each on the same GPU, form a
comparison across vanilla SAM, vanilla MedSAM, fine-tuned SAM and fine-tuned MedSAM, report how big
each model is, and look into costs (training time, hardware needed).*

| Requirement | Status | Where / note |
|---|---|---|
| Fine-tuning for each model | Done | LoRA on SAM ViT-H and MedSAM ViT-B (Q/V proj, r=4 / α=8, frozen encoder + light CNN decoder); U-Net trained as the from-scratch specialist |
| Add dataset + "extra structure" | Done | PraNet 5-split protocol (`src/data/`); the "extra structure" is the CNN mask decoder bolted onto SAM/MedSAM (`src/models/sam_adapter.py`) |
| Run each on the **same GPU** | Done | `train.py` runs every model under one protocol; `metrics.json` records `device` + `device_name` |
| Report **how big** each model is | Done | `metrics.json → params.total / trainable / trainable_pct` and `checkpoint_size_mb`; `05_benchmark` prints a param table |
| **Cost**: how long to train, what HW was needed | Done | `metrics.json → timing` (per-epoch, total, mean, epochs_run, stop reason) + `device_name` |
| Compare **with fine-tuning and without** | **Missing** | No zero-shot inference path exists; eval assumes the fine-tuned CNN-decoder head |
| 4-way: vanilla SAM, vanilla MedSAM, fine-tuned SAM, fine-tuned MedSAM | **2 of 4** | Fine-tuned SAM + fine-tuned MedSAM exist; both **vanilla** baselines are missing |

Net: **5 of 7 done.** The remaining two are one coherent chunk of work (below).

---

## What's done

- **Config-driven runner** — `train.py` + `configs/run.yaml` (layered on `configs/base.yaml`) +
  `src/config.py` + `src/training/{engine,reporting}.py`. Pick a model with `run.model`
  (`unet | sam_lora | medsam`); every model trains identically. `--dry-run` validates offline (no GPU).
- **Efficiency metrics** — each run writes `results/<model>/seed<seed>/metrics.json` with accuracy
  (all 5 splits + generalization gap), params (total / trainable / %), checkpoint size, device name,
  and per-epoch + total training time. Overlays + `run.log` alongside; all mirrored to Drive.
- **Colab wrapper** — `notebooks/train_colab.ipynb` (3 cells: mount Drive → pull code + data +
  checkpoint → `train.py`).
- **Benchmark notebook** — `notebooks/05_benchmark.ipynb` compares the trained models (param table,
  per-split metrics, seen-vs-unseen bars, generalization-gap row, parameter-efficiency scatter,
  qualitative panels). **Currently wires up 3 models** (U-Net, SAM+LoRA, MedSAM+LoRA).
- **Historical per-model notebooks** (`02_unet_baseline`, `03_sam_lora`, `04_medsam_lora`) preserved
  on the `backup/per-model-notebooks` branch; removed from `main` as superseded.

---

## Remaining work (pick up here)

Goal: complete the four-way **vanilla vs fine-tuned** comparison so the professor's checklist is
fully satisfied.

1. **Zero-shot vanilla SAM baseline** — inference-only wrapper (no LoRA, no training) that segments
   using a prompting protocol (see [open decision](#open-decision)).
2. **Zero-shot vanilla MedSAM baseline** — same wrapper, MedSAM ViT-B weights, **same** protocol so
   the two vanilla models are directly comparable.
3. **Give the vanilla models a `metrics.json`-style record** — params + checkpoint size + **inference**
   time. Training cost = zero, which is itself the headline ("fine-tuning cost buys X Dice over
   zero-shot").
4. **Extend `notebooks/05_benchmark.ipynb`** to add both vanilla models to the `models` dict, yielding
   the full table: vanilla SAM · vanilla MedSAM · fine-tuned SAM · fine-tuned MedSAM (+ U-Net as a 5th
   reference). U-Net has no "vanilla" mode — it is a from-scratch specialist, not a foundation model.

---

## Open decision

**Prompting protocol for the vanilla baselines — blocks the vanilla eval, but NOT training.**
SAM and MedSAM are *promptable*: they do not emit a polyp mask from an image alone (the current eval
path `sigmoid(model(img))` only works because the fine-tuned models carry an added CNN decoder). To
score vanilla SAM/MedSAM we must pick one protocol and apply it identically to both:

- **Box prompt from ground truth** *(recommended)* — standard in the medical-SAM literature; MedSAM is
  literally trained for box-prompted segmentation. Caveat to document: this is an *oracle-prompted*
  baseline (it is told roughly where the polyp is) while the fine-tuned models get no such hint — so if
  fine-tuning matches or beats it, that is a strong result.
- **Automatic mask generation** (point grid → pick the polyp-overlapping mask) — no location leakage,
  but messier and needs a selection rule.
- **Single center / random point** — cleanest, but weakest signal.

Confirm with the professor before implementing; it changes what the "without fine-tuning" numbers mean.

---

## Sequencing — is the vanilla work worth doing *before* training?

**No — do not gate training on it. Train first; build the vanilla baselines in parallel.** Rationale:

- **Training is the long pole and the critical path.** The fine-tuned checkpoints are required for
  *both* the "with fine-tuning" numbers and the comparison, so start the GPU work as early as possible.
- **The vanilla path shares no code with training** — it is an inference-only wrapper. Building it does
  not benefit from, or block, the training runs; it can be written on CPU / a T4 while training runs.
- **Vanilla numbers are only interpretable next to fine-tuned numbers**, which come from training.
  Finishing the vanilla eval first would leave it sitting idle with nothing to compare against.

Two things *are* worth doing up front (neither blocks training): (a) lock the prompting protocol above,
and (b) optionally a short **smoke test** — a few-epoch train of one model wired through the benchmark —
to confirm the harness produces sane, comparable numbers before committing full GPU budget.

**Recommended order:**
1. Kick off full training for all three models (below).
2. In parallel: lock the protocol, build the two zero-shot wrappers, extend `05_benchmark`.
3. When checkpoints land, run the full four-way benchmark once.

---

## How to run

**Train one model** (repeat per model):
1. Edit `configs/run.yaml`: set `run.model` (`unet | sam_lora | medsam`), `seed`, `epochs`; set
   `output.drive_results_dir` to your Drive path.
2. **Commit and push** the config change first — `train_colab.ipynb` does `git reset --hard origin/main`,
   so edits made only inside Colab are wiped.
3. Open `notebooks/train_colab.ipynb` in Colab, pick the GPU (T4 for U-Net / MedSAM ViT-B; L4 or A100
   for SAM ViT-H, ~2.4 GB), run the 3 cells.
4. Outputs: `checkpoints/<model>/seed<seed>/best.pt` and `results/<model>/seed<seed>/metrics.json`
   (+ overlays + `run.log`), mirrored to Drive.
5. Offline sanity check: `python train.py --config configs/run.yaml --dry-run`.

**Compare models:** after the checkpoints exist, run `notebooks/05_benchmark.ipynb`.

---

## Task checklist

- [ ] Train U-Net baseline (T4) → checkpoint + metrics.json
- [ ] Train fine-tuned MedSAM ViT-B (T4) → checkpoint + metrics.json
- [ ] Train fine-tuned SAM ViT-H (L4 / A100) → checkpoint + metrics.json
- [ ] Confirm prompting protocol for vanilla baselines (professor sign-off)
- [ ] Implement zero-shot vanilla SAM inference wrapper
- [ ] Implement zero-shot vanilla MedSAM inference wrapper
- [ ] Record vanilla params / checkpoint size / inference time
- [ ] Extend `05_benchmark.ipynb` to the four-way (2×2) comparison + U-Net
- [ ] Write up accuracy-vs-cost findings (size, training time, hardware needed)
