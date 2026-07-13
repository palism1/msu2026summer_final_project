---
status: living
last_updated: 2026-07-13
---

<!-- Delegation plan for the remaining work. Each task is scoped to the owner best suited to it.
     Doc class: living — bump last_updated above on any substantive change (see DECISIONS.md).
     Authored by Fable as orchestrator. Execution is per-task; each is self-contained.
     Updated: corrected Fix 1 (already implemented), added Task A (results aggregator, do first),
     recorded Fix 2 in-progress state. -->

# Delegation Plan — Remaining Work

**Orchestrator:** Fable (me). Subagents are NOT being used this session (Mikko's preference);
code changes are done inline by the orchestrator. Synthesis (write-up) stays with the orchestrator;
the professor sign-off has no owner but Mikko.

**Skill posture (Mikko's steer): use the right skill where it helps, skip it when overkill.**
Applied here: test-driven-development (light) for code changes with a testable contract;
verification-before-completion (run tests + dry-run, show output) before any "done"; NOT brainstorming
(designs are decided) and NOT systematic-debugging (no bug in play).

Grounding facts already verified against the code (so nothing gets re-derived from the stale HANDOFF):
- **Fix 1 is already implemented in code.** `train_colab.ipynb` cell 1 has `SEEDS = [42, 43, 44]`
  and cells 4-6 loop `model × seed`; `05_benchmark.ipynb` cells 13-14 already aggregate per-seed
  `metrics.json` into mean ± std for the prompt-free models, reading `m['eval'][key]['dice']` exactly
  as `reporting.py:build_metrics_payload` writes it. Nothing to build — only the seed 43/44 GPU runs.
- `train.py` accepts `--seed`/`--model`; per-seed outputs route to `.../seed<seed>/` (src/config.py).
- `build_sam_lora(model_type="vit_b")` already exists; the `sam_vit_b_01ec64.pth` download URL is
  already present in `train_colab.ipynb` cell 3. The checkpoint-path scheme is `[DO NOT TOUCH]`.

---

## Task A — Results aggregator (NEW · do first)

**Owner:** orchestrator (me), inline. **Status:** DONE — `src/results_summary.py` +
`aggregate_results.py` + `tests/test_results_summary.py` (5 tests) landed and verified; the CLI was
smoke-tested against a fixture and produces `results/summary/{SUMMARY.md,summary_flat.csv,
summary_by_model.csv,summary.json}`.
**Goal:** one folder you open to see everything the notebooks already produced — accuracy, params,
size, training time, per split, per model, per seed — pulled from local `results/` or the Drive
mirror, **without re-running any notebook**. If a model/seed's output is already there, it is simply
collected; nothing retrains or re-evaluates.

**What it reads (the "churned-out data"):**
- Every `results/<model>/seed<seed>/metrics.json` — local first, then the Drive mirror
  (`/content/drive/MyDrive/msu2026_checkpoints/results/<model>/seed<seed>/metrics.json`).
- Any saved overlay PNGs already next to those metrics, and any figures `05_benchmark`/`06_findings`
  persisted (best-effort copy; confirmed at implementation time — today they mainly render inline).

**What it writes — a single `results/summary/` folder:**
- `SUMMARY.md` — the one human file to open: a per-`model × seed` table (5-split mDice, seen/unseen
  means, generalization gap, trainable params, checkpoint size, epochs, train minutes, device) plus a
  mean ± std-by-model table.
- `summary_flat.csv` — one row per `model × seed`.
- `summary_by_model.csv` — mean ± std per model (matches benchmark cell 14).
- `summary.json` — full machine-readable dump.
- `figures/` — copied overlays/benchmark figures when present.

**How (design):**
- Extract the load-and-aggregate logic (currently inline in `05_benchmark.ipynb` cell 14's
  `_load_seed_metrics` + per-split mean/std) into a **torch-free** `src/results_summary.py` so the
  notebook cell and the new script share one source of truth. Add a thin `aggregate_results.py` at the
  repo root (mirrors `train.py`/`evaluate.py` placement) that calls it and writes `results/summary/`.
- Torch-free (stdlib + optional pandas) so it runs anywhere with no GPU. Idempotent; prints a coverage
  report (which `model × seed` were found vs missing) and a `--skip-if-done` flag.
- **Automatically absorbs later work:** once Fix 1's seeds 43/44 and Fix 2's `sam_b` runs land on
  Drive, re-running `aggregate_results.py` picks them up with no code change.

**Skill fit:** light TDD — add a `tests/test_results_summary.py` that feeds two fake per-seed
metrics dicts and asserts the mean/std and the flat-table shape (pure, no GPU). Then implement.
**Verification:** new tests green in the full suite; run `aggregate_results.py` against a small
fixture dir and confirm `SUMMARY.md` + CSVs render.

## Task B (was Fix 2) — SAM ViT-B + LoRA variant (isolate the MedSAM backbone confound)

**Owner:** orchestrator (me), inline. **Status:** DONE (code) — source + notebooks + evaluate.py
wired and verified (31 tests green, `sam_b` dry-run shows `sam_vit_b/seed43`, both notebooks parse).
Only the GPU training run of `sam_b` remains, which is Mikko's on Colab.
**Why:** MedSAM (ViT-B) came out weakest, but it is a smaller backbone than SAM (ViT-H). A SAM ViT-B
+ LoRA run has the *same backbone size as MedSAM* with *generic SAM weights*, so it separates "medical
pretraining" from "backbone capacity." Implemented as a new `sam_b` model choice mirroring `medsam`.

**Done (source):**
- `src/config.py`: `sam_b` added to `MODEL_CHOICES`, `_resolve_backbone` (→ vit_b from a `sam_b`
  block), `_checkpoint_name` (→ `sam_vit_b`, distinct from `medsam` and `sam_vit_h`).
- `src/training/engine.py`: `sam_b` dispatch branch → `build_sam_lora(model_type="vit_b", ...)`.
- `train.py`: `sam_b` added to `--model` choices.
- `configs/base.yaml`: new `sam_b` block (`vit_b`, `sam_vit_b_01ec64.pth`, LoRA r=4/α=8).
- `tests/test_config.py`: failing-first tests for `sam_b` path resolution + updated `MODEL_CHOICES`.

**Done (notebooks + CLI):**
- `notebooks/train_colab.ipynb`: cell 3 now loops over `('sam_lora','sam') / ('sam_b','sam_b')` and
  fetches whichever checkpoint(s) the run needs; cell 1 documents `sam_b` as an option.
- `notebooks/05_benchmark.ipynb`: `sam_vit_b` added as an **optional** checkpoint (`AVAILABLE_OPTIONAL`,
  never raises if untrained), built into `models` when present, and added to the multi-seed
  aggregation's `PROMPT_FREE_DIRS`.
- `evaluate.py`: `sam_b` added to `--model` choices + build dispatch (reads the `sam_b` config block).
**Guard held:** reused `build_sam_lora(model_type="vit_b")` — no new builder; only ADDED a path leaf,
never changed the existing `[DO NOT TOUCH]` scheme.

## Fix 1 — Multi-seed (2-3 seeds) — CODE COMPLETE

**Owner:** Mikko (GPU runs only). No code work: the trainer loop and the benchmark's mean ± std
aggregation already exist and are correct (see grounding facts). Run seeds 43 and 44 on Colab; the
skip-if-done logic means finished pairs are not retrained. Task A then surfaces the multi-seed spread
in one place.

## Fix 3 — Accuracy-vs-cost write-up

**Owner:** orchestrator (me). **Consumes Task A's `SUMMARY.md`.** Draft `docs/FINDINGS.md`: per model,
mDice seen vs unseen, gap, trainable params, ckpt size, train time, hardware. Lead with the honest
headline (SAM ViT-H + LoRA generalizes best among prompt-free models at 0.792) and the oracle-prompt
caveat. State limitations: single seed until Fix 1's runs, MedSAM confound until Task B's run, oracle
asymmetry. Draftable now on single-seed numbers; finalized once Task A reflects the new runs.

## Fix 4 — Professor sign-off on the oracle prompt protocol

**Owner:** Mikko (human). No agent can make the scientific call. I can draft the one-paragraph ask
(default `box` from GT mask; the oracle caveat; the `point` alternative) on request.

---

## Execution order

| # | Task | Owner | Depends on |
|---|---|---|---|
| 1 | **Task A — results aggregator** | Me, inline | nothing (runs against existing results now) |
| 2 | **Task B — `sam_b` variant** (finish notebooks + verify) | Me, inline | in progress |
| 3 | Fix 1 seed 43/44 runs | Mikko (Colab) | — (code done) |
| 4 | Task B `sam_b` GPU run | Mikko (Colab) | Task B code |
| 5 | Re-run aggregator to absorb new rows | Me / Mikko | Tasks A, 3, 4 |
| 6 | Fix 3 write-up | Me | Task A summary |
| 7 | Fix 4 professor sign-off | Mikko | — |

**Final verification (orchestrator):** full `pytest` green; `sam_b` dry-run shows the new paths;
notebooks parse; `aggregate_results.py` produces `results/summary/`; the write-up's numbers match the
regenerated summary. GPU runs that produce those numbers are Mikko's to execute on Colab.
