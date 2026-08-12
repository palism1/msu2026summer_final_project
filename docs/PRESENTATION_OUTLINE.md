---
class: living
last_updated: 2026-08-11
---

# Final Presentation Outline — 15 minutes

One block per slide. To rearrange the talk, move whole blocks. Each block carries a
status tag:

- **READY** — the numbers and assets exist on disk today.
- **BUILD** — the figure or diagram does not exist yet and must be made.

Experiments A, B, and C completed 2026-08-12; every slide now has final numbers.

Target: **15 content slides + backup**, about one minute per slide. The two program
decks (Bhowmik, Ahmed; same advisor) run 27-28 slides, but both were longer talks.
This outline keeps their skeleton (title with committee, problem, related work,
diagram, methods, results, limitations, future work, acknowledgements with repo link)
and compresses it to the 15-minute budget.

Number sources: `results/summary/SUMMARY.md` (regenerated 2026-08-12, 20 runs, A100).

---

## Slide 1 — Title · 0:30 · READY
- Project title: *LoRA-Adapted Foundation Models vs. Specialist U-Net for
  Cross-Dataset Polyp Segmentation*
- Presenter, advisor, committee members (both program decks name the committee here).
- GitHub repository link on the title slide (program convention).

## Slide 2 — Clinical problem · 1:00 · READY
- Missed polyps during colonoscopy raise interval cancer risk.
- Segmentation models help, but each clinic's data looks different: scope, lighting,
  population. A model trained at one site degrades at another.
- One line: "The question is not accuracy on the training benchmark. It is accuracy
  at the next clinic."

## Slide 3 — Headline result (preview) · 1:00 · READY
- State the finding up front; the talk then earns it.
- "A frozen SAM backbone with a 830K-parameter LoRA adapter beats a 24.4M-parameter
  specialist U-Net on unseen datasets: 0.806 vs 0.755 mean Dice, with roughly half
  the generalization drop (0.082 vs 0.145)."
- One small two-bar chart: unseen mDice, SAM-H+LoRA vs U-Net.

## Slide 4 — Related work · 1:00 · READY
- Numbered citations, each with a one-line limitation (Ahmed-deck pattern):
  1. PraNet (2020) — defined the 5-split seen/unseen protocol; specialist CNN.
  2. SAM (2023) — general segmenter; weak on medical images out of the box.
  3. MedSAM (Nat. Commun. 2024) — claims to beat specialists on endoscopy; box-prompted.
  4. SAMed (2023) — LoRA on frozen SAM for medical segmentation; the method this
     project adapts.
- The unanswered question: does LoRA adaptation buy *generalization*, measured on the
  same benchmark specialists were built on?

## Slide 5 — Study design · 1:00 · READY
- Four trained models, identical protocol, 3 seeds each: U-Net (ResNet-34, 24.4M
  trainable), SAM-ViT-H+LoRA (830K), SAM-ViT-B+LoRA (322K), MedSAM-ViT-B+LoRA (322K).
- `sam_b` vs `medsam` isolates pretraining source at equal backbone size.
- Separate untrained tier: oracle-box baselines (upper bound, not a competitor).

## Slide 6 — Data and protocol · 1:00 · BUILD (diagram)
- PraNet protocol diagram: 1450 training images → 2 seen test splits (Kvasir,
  CVC-ClinicDB) + 3 unseen splits (CVC-ColonDB, ETIS-Larib, CVC-300).
- "Seen" = learning ability. "Unseen" = generalization (PraNet's own framing).
- 352×352, Dice+BCE loss, best-val-Dice checkpoint, mean over seeds 42/43/44.

## Slide 7 — Method: LoRA on frozen SAM · 1:30 · BUILD (diagram)
- SAMed-style architecture figure: frozen ViT blocks in gray, LoRA bypass (r=4, α=8,
  Q/V only) in color, light CNN decoder, no prompts at inference.
- Annotate: 830K trainable = 3.4% of U-Net's parameter count.
- One line on the U-Net baseline setup (same loss, same data, same schedule).

## Slide 8 — Result 1: seen datasets · 1:00 · READY
- Grouped bars, Kvasir + ClinicDB: U-Net 0.900 mean, SAM-H+LoRA 0.887.
- Message: the specialist wins at home. This is the expected result, stated plainly.

## Slide 9 — Result 2: unseen datasets (centerpiece) · 1:30 · BUILD (radar or bars)
- Radar chart, one axis per dataset, one polygon per model. U-Net's polygon collapses
  on the unseen axes; SAM-H+LoRA stays round.
- Numbers: unseen mDice SAM-H 0.806, SAM-B 0.761, U-Net 0.755, MedSAM 0.661.
- ETIS-Larib, hardest split: SAM-H 0.745, U-Net 0.690, MedSAM 0.500.

## Slide 10 — Result 2b: the generalization drop · 1:00 · BUILD (delta chart + overlays)
- One bar per model: seen mDice minus unseen mDice. U-Net 0.145, SAM-H 0.082,
  SAM-B 0.104, MedSAM 0.158.
- Below or beside: 3-row qualitative grid (image, GT, U-Net, SAM-H) on ETIS/ColonDB
  failures. Overlays already exist under `results/<model>/seed<seed>/`.

## Slide 11 — Result 3: the MedSAM puzzle · 1:00 · READY
- MedSAM's paper claims it beats U-Net specialists on polyp endoscopy. Here it is
  last on every split (0.661 unseen; 0.500 on ETIS-Larib).
- Tension slide: published claim vs measured result. Sets up the investigation.
- Tease the resolution: the deficit bundled three causes, and we measured each.

## Slide 12 — Result 3b: the diagnosis · 1:00 · READY
- The bug: the pipeline fed MedSAM ImageNet-standardized inputs (about [−2.1, +2.6]);
  MedSAM's frozen encoder was fit on per-image min-max [0,1].
- Three-arm result (3 seeds each, jitter-controlled): published 0.661 → ctrl 0.655
  (jitter confound −0.006, negligible) → minmax 0.725 (**normalization bug +0.070**).
- Corrected pretraining effect at equal backbone: 0.761 (SAM-B) vs 0.725 = −0.036,
  smaller than the backbone effect (−0.045). The old "net drag" claim is retracted.

## Slide 13 — Oracle-box ceiling and the punchline · 1:00 · READY
- Untrained models with a ground-truth box; negative seen−unseen gap = ceiling, not
  generalization. SAM-H 0.905, SAM-B 0.880, MedSAM(ImageNet) 0.845.
- **The punchline: MedSAM with both contracts honored (box + min-max) is the best
  number in the study: 0.925 unseen.** The bug replicates untrained (+0.080).
- Prompt dependence quantified: a box lifts SAM-H +0.099, corrected MedSAM +0.200.
  "MedSAM is not weak; it is specialized, and it pays exactly when the box is gone."
- Cut candidate: if time runs short, fold the punchline into slide 12 and move the
  rest to backup.

## Slide 14 — Limitations · 0:30 · READY
- One benchmark family (PraNet splits), polyps only.
- Three seeds; oracle tier has one run.
- Cost asymmetry: U-Net trains in 10 min with a 98 MB checkpoint; SAM-H+LoRA takes
  120 min and a 2.5 GB checkpoint (dominated by the frozen backbone).

## Slide 15 — Conclusion and contributions · 1:00 · READY
- Restate: LoRA adaptation of a frozen foundation model halves the generalization
  drop at 3% of the trainable parameters.
- Contributions list (program convention): controlled 4-model comparison, the
  backbone-vs-pretraining decomposition, the MedSAM preprocessing investigation,
  a reproducible pipeline (repo link).

## Slide 16 — Acknowledgements + Thank you · 0:30 · READY
- Advisor, committee, MSU. Repository link again. Questions.

---

## Backup slides (after the end slide, for Q&A)
- B1. Full 6-metric table per model per split (from `results/summary/`).
- B2. Per-seed variance table (mean ± std, seeds 42/43/44).
- B3. Hyperparameters: LoRA r=4 α=8 Q/V, schedule, augmentation, hardware.
- B4. Oracle-box protocol detail and the prompt-dependence numbers.
- B5. MedSAM investigation timeline (from `docs/MEDSAM_INVESTIGATION.md`).
- B6. Why min-max normalization needs an augmentation control (affine invariance).

---

## Time check

| Section | Slides | Minutes |
|---|---|---|
| Title + problem + headline | 1-3 | 2:30 |
| Related work + design + data | 4-6 | 3:00 |
| Method | 7 | 1:30 |
| Results 1-2 (main story) | 8-10 | 3:30 |
| Result 3 (MedSAM subplot) | 11-13 | 3:00 |
| Limitations + conclusion + thanks | 14-16 | 2:00 |
| **Total** | **16** | **15:30** |

Slide 13 is the designated cut if rehearsal runs over.

## Assets to build (ranked)
1. Radar chart of per-split Dice (slide 9) — the single most memorable figure.
2. Generalization-drop delta bars (slide 10) — no surveyed paper shows this; distinctive.
3. LoRA architecture diagram (slide 7) — SAMed-style, gray frozen blocks.
4. PraNet split diagram (slide 6).
5. Qualitative overlay grid (slide 10) — select 3 rows from existing overlays.
Data for 1, 2, and 5 exists today in `results/summary/` and `results/<model>/seed<seed>/`.
