---
class: living
last_updated: 2026-08-19
---

# Speaker Script · Defense v4 (15 minutes)

Read this aloud as written, or use it as the spine and improvise around it.
Each block names the slide, gives a time budget, and marks chart walkthroughs
with **[CHART]**. Total spoken time at a normal pace: about 16 minutes. To get
back under 15, trim the "How they differ" paragraph on slide 5.

Numbers come from `results/summary/SUMMARY.md` (22 runs: 18 trained + 4 oracle; A100, seeds 42/43/44).

---

## Slide 1 · Title · 0:30

Good morning. My name is Mikko Palis, and this is my final project for the MSU
2026 summer program. The title is long, so here is the plain version: I took a
large general-purpose vision model, added a small trainable adapter, and
tested whether that combination handles new hospitals better than a model
built specifically for the task. The claim on the screen is the whole talk in
one sentence. Everything is on GitHub at the link shown, and I will show it
again at the end.

## Slide 2 · The clinical problem · 1:00

Start with why this matters. During a colonoscopy, doctors miss about one in
four adenomas. That number comes from a 43-study meta-analysis by Zhao and
colleagues in Gastroenterology, 2019.

**[CHART]** The dot grid on the right makes that concrete: each dot is an
adenoma, and the red ones, 26 out of 100, are the ones that get missed. A
missed polyp can surface later as an interval cancer.

Software that outlines polyps on screen during the exam can help. The catch:
every clinic's images look different. Different camera hardware, different
lighting, different patients. A model trained at one clinic loses accuracy at
the next. So I trained one specialist and three adapted general models, and I
scored all of them on clinics they never saw. The question for the next
fourteen minutes: which approach keeps its accuracy at the next clinic?

## Slide 3 · Headline result · 1:00

Here is the answer up front, with three terms defined so the chart makes
sense. SAM is a big general segmentation model from Meta, trained on everyday
photos. LoRA is a small trainable add-on, about 830 thousand parameters, and
SAM itself stays frozen. The Dice score measures overlap between the model's
outline and the true outline, zero to one, higher is better.

**[CHART]** Two bars, and the y-axis starts at zero, so the heights are the
honest comparison. The red bar is the adapted general model at 0.806 on data
from new clinics. The gray bar is the specialist U-Net at 0.755. The whiskers
on top are one standard deviation over three training runs, and the difference
is about four times the largest of them, so the ranking held on every run. Two
things to remember:
the adapted model wins by 0.051 Dice on new data, and it does that while
training only 3.4 percent as many parameters. The rest of the talk supports
this slide.

## Slide 4 · Related work · 1:00

Four prior works set this up. PraNet, from Fan and colleagues at MICCAI 2020,
is a specialist CNN, and it defined the benchmark I use: train on two
datasets, test on five. SAM, from Kirillov and colleagues at
ICCV 2023, is Meta's promptable segmenter, trained on 1.1 billion masks of
everyday photos; out of the box it does poorly on medical images, and several
independent evaluations confirmed that. MedSAM, from Ma and colleagues in
Nature Communications 2024, retrained SAM on 1.57 million medical image-mask
pairs, but it needs a hint box drawn around the target every time it runs.
SAMed, from Zhang and Liu in 2023, showed that a small LoRA adapter can adapt
frozen SAM for prompt-free medical segmentation; that is the recipe I adapted.

The red line is the point of this slide: none of the four measured whether the
adapted general model also generalizes better on the specialists' own
benchmark. That measurement is this project.

## Slide 5 · Study design · 1:00

Four trained models, one protocol, three random seeds each. The U-Net is the
specialist. SAM-ViT-H plus LoRA is the big general model with the adapter.
SAM-ViT-B plus LoRA is the same idea on a smaller backbone. MedSAM-ViT-B plus
LoRA is the medically retrained version at that same size, and that pairing is
deliberate: SAM-B against MedSAM holds the architecture, the adapter, the data,
and the schedule fixed and changes only the starting weights. It measures what
MedSAM's medical retraining adds or costs at this size.

**[CHART]** The bar chart on the right shows the sizes on a linear scale. The
blue bar is the U-Net's 24.4 million trainable parameters. The red and green
slivers next to it are the LoRA adapters, 830 thousand and 322 thousand. The
chart exists to make one point visually: the adapters train about three
percent of what the specialist trains.

How they differ, briefly. The U-Net is small and fast: 24 million
parameters, ten minutes to train, a 98-megabyte file; it is the practical
pick when the deployment site matches the training data and compute is
tight. ViT-H is SAM's largest encoder, about 636 million parameters in
total; it carries the strongest general features, wins on generalization
here, and costs two hours of training, a 2.5-gigabyte checkpoint, and the
heaviest inference. ViT-B is SAM's smallest encoder, about 91 million
parameters; it trains in half an hour and lands between the two, so it is
the middle option when SAM-H is too heavy to serve. MedSAM shares the ViT-B
backbone, but its weights were retrained on 1.57 million medical images;
prompt-free it trails the generic version, and with a box it posts the best
number in the study, so it fits a pipeline where a detector or a clinician
supplies the box.

There is also a separate tier of untrained models that get a hint box around
the true answer. It serves only as an upper bound, and it returns at the end
of the talk.

## Slide 6 · The benchmark protocol · 1:00

**[CHART]** Read the diagram left to right. Two datasets supply the training
images: 900 from Kvasir and 550 from CVC-ClinicDB, 1450 total. Testing happens
on five datasets. The two gray boxes at the top are held-out images from the
same two datasets; scores there measure learning. The three orange boxes are
whole datasets the model never saw: CVC-ColonDB, ETIS-Larib, and CVC-300,
which is the EndoScene test set. Scores there measure generalization. That
split is PraNet's own framing, so the specialists' side designed this test,
and any advantage in the setup belongs to them.

The details at the bottom: 352 by 352 inputs, Dice plus cross-entropy loss,
best-validation checkpointing, and every number you will see is a mean over
seeds 42, 43, and 44.

## Slide 7 · Method · 1:30

This slide is the method, and it needs one piece of background. A neural
network is a stack of layers, and each layer holds millions of numbers called
weights. Training means adjusting those weights, a little at a time, until the
outputs improve. The usual way to adapt a model to a new task is to adjust all
of them.

**[CHART]** This study does something cheaper. In the diagram, the gray blocks
are SAM's encoder, the part that turns an image into features. Gray means
frozen: those weights never change, they stay exactly as Meta shipped them.
The small blue boxes are the LoRA adapters. Next to certain layers, LoRA
attaches a pair of tiny matrices. The frozen layer's output passes through
unchanged, and the adapter adds a small learned correction on top. During
training, only those corrections change. Think of the frozen model as a
printed textbook and the adapters as sticky notes in the margins: the book is
never rewritten, and everything specific to polyps goes on the notes.

At the end of the stack, a small decoder, also trained, turns the corrected
features into a polyp outline. The trainable part, adapters plus decoder, is
830 thousand weights, 3.4 percent of the U-Net's 24.4 million. And at test
time the model receives the raw image and nothing else: no clicks, no boxes,
the same conditions the U-Net gets.

The U-Net baseline trains with the same loss, the same data, and the same
schedule, so the only difference is the architecture.

## Slide 8 · Result 1, familiar data · 1:00

First result: on familiar data, the specialist keeps a small lead.

**[CHART]** This is a dot plot, and the axis is truncated: it starts at 0.75,
which is fair for dots because the position carries the value, and it lets you
see small differences. Each dot is a model's mean Dice on one seen dataset,
and the horizontal whiskers are one standard deviation over the three seeds.
The top group is Kvasir, the bottom group is ClinicDB. On the combined seen
average the U-Net sits at 0.900 and SAM-H plus LoRA at 0.887. Notice how wide
the whiskers are relative to that difference: the lead is 0.013, about one and
a half times the seed-to-seed spread. This is the expected result. The specialist
was built and trained for exactly this data, and at home it wins.

## Slide 9 · Result 2, new data · 1:30

Now the centerpiece. On new datasets, the ranking reverses.

**[CHART]** Each line in this chart belongs to one model. The left end is its
score on familiar data, the right end is its score on the three unseen
datasets, so a steeper line means more accuracy lost in the move. Watch the
blue U-Net line: it starts highest on the left and crosses below both SAM
lines on the way down. The red line, SAM-H plus LoRA, starts slightly lower
and stays flattest, ending at 0.806. The final order on new data: SAM-H 0.806,
SAM-B 0.761, U-Net 0.755, MedSAM 0.661. On the hardest dataset, ETIS-Larib,
the spread is wider still: SAM-H 0.745, U-Net 0.690, MedSAM 0.500. Remember
that MedSAM number; it becomes a puzzle in a minute.

## Slide 10 · The drop, quantified · 1:00

Same story, measured directly.

**[CHART]** The left chart plots the drop itself: seen score minus unseen
score, so shorter bars are better. The U-Net loses 0.145 Dice when it moves to
new data. SAM-H plus LoRA loses 0.082, which is 44 percent less. The error
bars are one standard deviation over seeds, and the two bars stay separated.

**[CHART]** The right grid shows what the numbers look like on real images
from CVC-ColonDB, a dataset neither model saw. The color fill is the model's
outline, the white line is the truth. Top row, an easy case: both models
succeed. Middle row: the U-Net outlines a fold of tissue instead of the polyp
and scores 0.560, while SAM-H stays on the polyp at 0.955. Bottom row: the
U-Net collapses to 0.114, and SAM-H finds the polyp at 0.846. The failures are
confident and wrong, which is exactly the failure mode you fear at a new
clinic.

## Slide 11 · Result 3, the MedSAM surprise · 1:00

Now the puzzle. The medical model came last. Why?

**[CHART]** The card at the top states the published claim: Ma and colleagues
report MedSAM beating specialist U-Nets on endoscopy, with a box prompt. The
bars below are what I measured, prompt-free: MedSAM last at 0.661, on every
dataset, and 0.500 on the hardest one.

One mismatch is visible before any debugging: their comparison was
box-prompted on both sides, because the specialist baselines received the
same box as an extra input channel; in my study every trained model runs
without a box. Beyond that, three candidate causes could each explain
the deficit: a preprocessing bug in my pipeline, the smaller ViT-B backbone,
and the box dependence itself. The next slide separates them.

## Slide 12 · The diagnosis · 1:30

Start with the bug, because the mechanism explains the whole initial number.
Before any model sees an image, a preprocessing step rescales the pixel
values. Generic SAM was built to expect one specific scaling, the ImageNet
standard, and my pipeline applied that scaling to every model. For SAM that is
correct by construction; it reproduces SAM's own preprocessing. But when
MedSAM's authors retrained SAM, they also changed the preprocessing: their
encoder saw pixels scaled between zero and one. My pipeline fed it the
SAM-style range instead, roughly minus two to plus two and a half, almost
five times wider. And the encoder is frozen, so it cannot recalibrate;
the small adapters sit in the middle of the network and cannot repair the
input scaling either. The model computed features on inputs unlike anything
in its training.

**[CHART]** Three controlled runs, three seeds each, separate that from
everything else. The first gray dot is the run you already saw, 0.661. The
second dot is a control: it disables only the brightness and contrast
augmentation, because the scaling fix silently disables those too. The score
moved 0.006, so the augmentation change explains nothing. The jump to the red
dot is the fix itself: feed MedSAM inputs scaled zero to one, the way it was
trained, and the score rises 0.069, from 0.655 to 0.725. One more check
closes the case. The same rescaling applied to the untrained, box-prompted
MedSAM, with no training in the loop at all, lifts it from 0.845 to 0.925.
One cause, the input scaling, accounts for both jumps.

The dashed green line above is the same-size generic SAM at 0.761. Even after
the fix, MedSAM sits 0.036 below it. For scale, moving from the small to the
large backbone under the same recipe is worth 0.045, though that move also
grows the adapter, so treat the two numbers as comparable in size rather than
strictly ordered. My earlier read, that medical pretraining was a net drag,
was wrong; the bug explained most of the deficit. Cause three, the box,
explains where the residual comes from, and it is next.

## Slide 13 · The hint-box ceiling · 1:00

**[CHART]** In this chart each row is one model. The hollow circle is the
trained, prompt-free score you have already seen; the filled circle is the
same backbone, untrained, given the oracle box; the connector is the
difference between those two conditions. Note that two things change between
the circles, the box arrives and the training goes away, so read the
connector as the value of being told where to look, net of everything my
training added. For SAM-H that difference is 0.099. For the corrected MedSAM
it is 0.200, up to 0.925, the best number in the entire study. Treat these as
a ceiling, and here is the evidence: these models score higher on data
labeled unseen than seen, and real generalization should not do that. The box
supplies the location, and finding the polyp is half the task.

So the resolution of the puzzle: MedSAM is specialized. It depends on the hint
box, and it loses the most accuracy when the box is gone.

## Slide 14 · Limitations · 0:30

Four honest limits. One benchmark family, the PraNet splits, polyps only.
Three seeds per trained model, and the hint-box tier has one run each.
Training cost differs a lot between the models, and the next slide gives the
bill. And my hint boxes came from the answer key; boxes from a
real detector would score lower.

## Slide 15 · Compute cost · 0:40

**[CHART]** This table is the bill for the whole study. Eighteen trained runs:
four main models and two diagnostic arms, three seeds each. Total training
time: about 750 GPU-minutes, or 12.5 hours on one A100. SAM-H dominates the
bill: its three seeds took six of those hours, about two hours per run, while
a U-Net seed took ten minutes. At Colab's rates, about 1.20 dollars per
A100-hour, the entire study cost about 15 dollars of compute; at a large cloud
provider's on-demand rate, closer to 50. Those dollar figures are estimates
from public rates, and the GPU-hours are measured. Two takeaways: rigor was
cheap here, because three seeds tripled the cost and the total still stayed
near 15 dollars, and the expensive part was the big frozen backbone, which
also weighs on inference.

## Slide 16 · Conclusion · 1:00

Back to the opening question: which approach keeps its accuracy at the next
clinic? The adapted general model. A small add-on to frozen SAM cuts the
accuracy lost on new clinics by 44 percent, training 3.4 percent as many
parameters as the specialist.

**[CHART]** The strip at the bottom compresses the study: the question, the
method, the answer with the numbers.

In practice: pick the specialist where it trained, pick the adapted general
model when the next clinic's data is unknown, and pick box-prompted MedSAM
when something else supplies a good box. Four contributions: the controlled
four-model comparison, two controlled pairings that separate the pretraining
source from the backbone size, the MedSAM preprocessing bug found, measured,
and corrected, and a reproducible pipeline on GitHub.

## Slide 17 · Thanks · 0:30

Thank you to my advisor, my committee, and MSU. The repository link is on the
screen, and I am happy to take questions.

---

## Q&A pointers (backup slides)

- "How stable are these numbers?" Go to B2, per-seed variance. The largest
  standard deviation is 0.014; the SAM-H versus U-Net difference on unseen
  data is 0.051.
- "What were the hyperparameters?" Go to B3. LoRA rank 4, alpha 8, query and
  value only; up to 50 epochs with early stopping; one A100; 7 to 143 minutes
  per run.
- "How exactly did the oracle-box tier work?" Go to B4. A tight bounding
  rectangle of the ground-truth mask, frozen published weights, no training.
  The negative seen-minus-unseen difference is the proof it measures a
  ceiling.

---

## FAQ · answers for questions that can trip you up

**Why three seeds and not one?**
Training is stochastic: the adapter and decoder start from random weights, and
data shuffling and augmentation differ run to run. The per-seed table shows
why one run misleads: the U-Net's unseen score ranges from 0.743 to 0.774
across seeds, and its best seed lands above SAM-B's worst. A single-seed study
could reverse the SAM-B versus U-Net ranking by luck. Three seeds give a mean
and a spread for every claim.

**Why was MedSAM trained nine times?**
Three training arms, three seeds each. The published arm reproduces the
original configuration, ImageNet normalization plus color jitter, and scored
0.661. The control arm turns jitter off and changes nothing else; it exists
because the normalization fix also forces jitter off, and it moved the score
only 0.006. The fixed arm uses per-image min-max scaling and scored 0.725.
The difference between the control and the fixed arm, 0.069, is the measured
size of the bug. The other models kept one arm each because nothing about
their preprocessing was in question.

**Then why not ten seeds?**
Cost against payoff. One SAM-H seed takes about two hours of A100 time, so ten
seeds per model would push the study past 40 GPU-hours. The spreads I measured
are small, at most 0.014, and the headline difference is 0.051, more than
three times that. More seeds would tighten intervals that are already tight
relative to the effect.

**Why seeds 42, 43, and 44 specifically?**
They were fixed in advance and never changed, so nothing was selected after
seeing results. The values themselves carry no meaning.

**Is the 0.051 difference statistically significant?**
I did not run a formal hypothesis test; with three runs per group a p-value
would carry little weight. What I can say: the difference is more than three
times the largest seed spread, and it holds on each unseen dataset separately.
ColonDB: 0.785 against 0.729. ETIS-Larib: 0.745 against 0.690. CVC-300: 0.887
against 0.847. A consistent direction across datasets and seeds is the honest
evidence at this scale.

**Why Dice as the headline metric?**
Dice is the standard metric on this benchmark, so my numbers compare directly
with the literature. Backup B1 carries the full six-metric table, and the
model ranking does not change under IoU or the other metrics.

**Why not give the trained models a box too?**
The deployment target is automatic detection during an exam, where nobody
draws a box on every frame. Giving one model a box changes the task from
finding and tracing to tracing alone. I quantify the box's value separately in
the oracle tier, and it is worth 0.200 Dice to MedSAM, which is exactly why it
cannot be handed out unevenly.

**Does the SAM-B versus MedSAM pair really isolate medical pretraining?**
It isolates the starting weights: same architecture, same adapter size, same
data, schedule, and seeds; only the initial encoder differs. Those starting
weights bundle two changes MedSAM's authors made together: training on medical
images and training with a box supplied on every example. So the pair measures
MedSAM's checkpoint as delivered, and the residual 0.036 deficit is consistent
with box dependence rather than with the medical data itself. The oracle tier
supports that reading: given a box, the corrected MedSAM posts the best score
in the study, so its medical features are strong when the box supplies the
location.

**Is the 0.045 difference between SAM-H and SAM-B a pure backbone-size effect?**
Approximately, and I say so with a caveat. The recipe is held fixed, rank 4 on
the query and value projections, but the adapter count scales with the
backbone, 322 thousand against 830 thousand trainable parameters, so adapter
capacity moves together with size. A strict size claim would need the adapter
capacity matched across backbones, and I did not run that. Also, 0.045 and the
0.036 pretraining residual are close, and each carries a seed spread near
0.005, so I treat them as comparable in magnitude rather than strictly
ordered.

**Why freeze the backbone instead of fine-tuning all of SAM?**
Three reasons. Full fine-tuning of the 630-million-parameter ViT-H did not fit
the compute budget. SAMed showed the frozen-plus-LoRA recipe works. And the
frozen backbone is the hypothesis itself: the claim under test is that general
features survive the domain move; updating them would test something else.

**Why 352 by 352 inputs when SAM uses 1024?**
The benchmark protocol fixes 352 by 352, which is what the U-Net consumes. The
SAM encoders internally require 1024 by 1024, so those models upsample first.
Both details are on backup B3.

**You reported a bug in your own pipeline. Why should the committee trust the
rest?**
The bug was found by treating a suspicious result as a claim to check, and the
diagnosis is itself a controlled experiment: a jitter-only control moved the
score 0.006 while the normalization fix moved it 0.069, and the same fix
replicates on untrained models, 0.845 to 0.925. Every run, seed, and config is
in the repository. Finding, measuring, and correcting the bug is one of the
contributions.

**Could a stronger specialist close the difference?**
Possibly some of it, and I did not test that. My U-Net is a standard ResNet-34
baseline; for reference, its unseen mean of 0.755 is close to what the PraNet
paper reports for its own purpose-built model on the same splits. The
comparison here holds training conditions equal across families rather than
chasing the best specialist, and the mechanism I measured, a larger drop for
the model that learns only from 1450 endoscopy images, would still apply.
Testing against stronger specialists is future work.

**Is SAM-H practical in a clinic? It is huge.**
The checkpoint is 2.5 gigabytes and the encoder is heavy, and I did not
measure latency, so I will not claim real-time readiness. SAM-B plus LoRA is
the middle option: 0.761 unseen, still above the U-Net, at a fraction of the
compute. Distillation and pruning are the standard next steps.

**Would this transfer to other organs or modalities?**
Untested. The study covers polyps in colonoscopy, one benchmark family. The
recipe is generic, and SAMed showed it on CT, so the transfer is plausible,
and plausible is all I can claim.
