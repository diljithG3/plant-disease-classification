# Project Phases

Single source of truth for what each phase covers and where things stand. README.md
and CLAUDE.md link here rather than duplicating this detail — update this file
whenever a phase starts, finishes, or its plan changes.

## Status at a glance

| # | Phase | Status |
|---|---|---|
| 0 | Project scaffolding | Done |
| 1 | EDA | Done |
| 2 | Data pipeline | Done |
| 3 | Training/eval harness | Done |
| 4 | CNN from scratch | Done |
| 5 | Transfer learning: feature extraction | Done |
| 6 | Partial fine-tuning | Done |
| 7 | Full fine-tuning | In progress |
| 8 | Compare all approaches | Not started |
| 9 (optional) | Robustness check | Not started |

---

## Phase 0 — Project scaffolding

**Goal:** folder structure, config system that resolves local-vs-Colab paths
automatically, dataset fetch script, dependency list.

**Status:** Done.

**Deliverables:** `configs/config.yaml`, `src/utils/env.py`, `src/utils/config.py`,
`src/data/fetch_data.py`, `requirements.txt`, `.gitignore`, `README.md`.

**Key decisions:** dataset kept outside Drive sync (`~/plant_disease_dataset` local by
default, overridable per-machine via `DATA_ROOT`/`.env` -- this dev machine uses
`C:\Diljith_AK\projects\plant_dataset`; `/content/dataset` Colab); all paths resolved
through config, never hardcoded; no git yet.

---

## Phase 1 — EDA

**Goal:** class distribution, a visual sample per class, and a full-dataset image
manifest (size/mode/corruption) to catch data-quality issues before building the
pipeline.

**Status:** Done — `notebooks/01_eda.ipynb`.

**Findings:**
- 38 classes, 54,305 images total.
- **36x class imbalance**: `Potato___healthy` (152 images) to
  `Orange___Haunglongbing_(Citrus_greening)` (5,507 images).
- **One stray RGBA image** among 54,304 RGB images
  (`Pepper,_bell___healthy`) — must `.convert("RGB")` on load.
- Zero corrupt/unreadable files; every image is a uniform 256x256.

**How findings feed forward:** the imbalance ratio drives Phase 2's class-weight
computation and Phase 3's weighted sampler/loss decision; the RGBA outlier drives
Phase 2's `Dataset.__getitem__` implementation.

---

## Phase 2 — Data pipeline

**Goal:** a fixed, reproducible train/val/test split; a PyTorch `Dataset` +
`DataLoader`; transforms (resize, augmentation, normalization); per-class training
weights. This split and these transforms are reused unchanged by every later phase.

**Status:** Done — `notebooks/02_data_pipeline.ipynb`.

**Deliverables:** `src/data/splits.py` (fixed per-class split), `src/data/transforms.py`
(train/eval transforms + `denormalize` for display), `src/data/dataset.py`
(`PlantDiseaseDataset`, `compute_class_weights`, `make_dataloaders`).

**What it does:**
- Stratified 70/15/15 split, done per-class with a fixed seed (not one global
  shuffle), cached to `experiments/data/splits.csv` so it's computed once and never
  drifts between phases or machines.
- `class_to_idx` mapping cached to `experiments/data/class_to_idx.json` — stays
  identical across every phase for checkpoints/metrics to remain comparable.
- Train transforms: resize to `image_size` (224), mild augmentation (horizontal flip,
  rotation, color jitter), then ImageNet-stats normalization. Eval transforms: resize
  + normalize only, no augmentation. Normalization is fixed to ImageNet stats across
  every phase (including the from-scratch CNN) so preprocessing is never a confound
  in the Phase 8 comparison, and so it already matches what Phases 5-7's pretrained
  backbones expect.
- Per-class weights computed from the train split only (inverse frequency), exposed
  for Phase 3 to consume — Phase 2 doesn't decide how they're applied.

**Findings/verification:**
- Split sizes: train 38,012 / val 8,146 / test 8,147 (of 54,305 total).
- Per-class `train_pct` clusters tightly around 70% (69.7%-70.1%, std 0.06) despite
  the 36x imbalance — the per-class split strategy works as intended.
- The Phase 1 RGBA outlier was loaded explicitly through `PlantDiseaseDataset` and
  confirmed to produce a normal 3-channel tensor — the `.convert("RGB")` fix is
  verified against the actual file, not just luck of random batch sampling.
- Class weight range: 0.143 (`Orange___Haunglongbing_(Citrus_greening)`, most common)
  to 5.186 (`Potato___healthy`, rarest).
- Train vs. eval batches visually confirmed distinct (flip/rotation/color jitter
  present in train, clean resize in eval) — augmentation verified wired correctly,
  not just present in code.

**Open decision carried to Phase 3:** how to apply `compute_class_weights` — weighted
loss, `WeightedRandomSampler`, or both. Left open until Phase 4's baseline shows how
much the imbalance actually hurts minority-class recall.

---

## Phase 3 — Training/eval harness

**Goal:** a training loop, checkpointing, TensorBoard + CSV logging, and an
evaluation function reporting accuracy, macro-F1, per-class recall, and a confusion
matrix. Built once, reused unmodified through Phases 4-7 so the Phase 8 comparison is
apples-to-apples.

**Status:** Done — `notebooks/03_training_harness.ipynb`.

**Deliverables:** `src/training/engine.py` (`train_one_epoch`, `evaluate` —
model-agnostic), `src/training/trainer.py` (`train_model`: training loop, early
stopping, checkpointing, resume-from-checkpoint), `src/training/metrics.py`
(`compute_metrics`, `plot_confusion_matrix`), `src/training/checkpoint.py`
(save/load), `src/training/logging_utils.py` (`RunLogger`: TensorBoard + CSV),
`src/utils/seed.py`, `src/utils/device.py`. `src/data/dataset.py` gained
`make_weighted_sampler` as the alternative to weighted loss.

**Validated with:** a throwaway tiny CNN (`DummyCNN`, defined in the notebook, not in
`src/`) on a small per-class subset (5 train + 2 val images/class) — fast enough for
local CPU. This is plumbing validation only; the real Phase 4 architecture is a
separate concern.

**Findings/verification:**
- Full round-trip confirmed: `best.pt` reloaded into a **fresh** model instance
  reproduces the exact `val_macro_f1` it was saved with (diff < 1e-6) — not just "the
  code ran," the save/load path is byte-correct.
- `class_to_idx` round-trips through the checkpoint unchanged — verified with an
  assertion, not assumed.
- `compute_metrics` returns correctly-shaped `per_class_recall` (38,) and
  `confusion_matrix` (38, 38) even when the subset used for validation only touches a
  handful of images per class.
- `checkpoint_metric` is `macro_f1` (see `configs/config.yaml`), not accuracy —
  consistent with Phase 1's imbalance finding.
- **Bug caught and fixed during this phase:** `tqdm`'s per-batch progress updates,
  captured via headless notebook execution, saved as ~48 separate stream-output
  fragments in one cell instead of one block — harmless at this phase's tiny subset
  scale, but would have bloated every real Phase 4-7 run (hundreds of batches/epoch).
  Fixed by raising `tqdm`'s `mininterval` in `engine.py` and adding
  `scripts/run_notebook.py`, which coalesces consecutive same-stream output before
  saving. All future notebook validation goes through this script (see CLAUDE.md).

**Resolved:** how to make the class-imbalance handling *available* — both weighted
loss (`nn.CrossEntropyLoss(weight=...)`, demonstrated in the notebook) and
`make_weighted_sampler` are implemented and callable. **Still open, deliberately:**
*which* Phase 4 should actually use — that's an empirical question for the from-scratch
CNN's baseline results, not a Phase 3 concern.

---

## Phase 4 — CNN from scratch

**Goal:** the core learning phase. Start with a minimal conv-block CNN, get the full
loop running end-to-end on Colab GPU, then iterate deliberately (batchnorm, dropout,
augmentation, depth) while observing the effect of each change on train/val curves.

**Status:** Done — `notebooks/04_cnn_from_scratch.ipynb`.

**Deliverables so far:** `src/models/cnn.py` (`SimpleCNN`: 3 conv blocks doubling
channels as spatial size halves via `MaxPool2d`, `AdaptiveAvgPool2d(1)`, linear head
— deliberately no batchnorm/dropout, so later additions have this baseline to compare
against), `notebooks/04_cnn_from_scratch.ipynb` (loads Phase 2's fixed split unchanged,
trains via the unmodified Phase 3 harness, evaluates on the held-out test set,
compares per-class recall against train-set frequency).

**Key decision:** first run is **unweighted** `CrossEntropyLoss` on purpose — Phase
2/3 deliberately left open *whether* to use weighted loss / `make_weighted_sampler`
(both already implemented), to be answered empirically from this baseline's
per-class recall rather than assumed upfront.

**Baseline results (Colab GPU, full split, `phase4_simplecnn_baseline`, 30 epochs,
unweighted `CrossEntropyLoss`, Adam lr=1e-3):**
- **Test accuracy 96.82%, test macro-F1 95.91%** (best val macro-F1 during training:
  0.9616, epoch 29/30 — early stopping never triggered, ran the full `max_epochs=30`
  ceiling).
- Strong for a deliberately minimal, unregularized 3-conv-block CNN — consistent with
  PlantVillage-color being a well-documented "easy" benchmark in the literature (clean
  lab-condition backgrounds), not a sign anything is wrong. This is exactly why the
  optional Phase 9 robustness check exists — this number is unlikely to hold up on
  real-world/field images.
- **Imbalance question, answered empirically as planned:** per-class recall does
  *not* cleanly track train-set frequency. Worst class is `Tomato___Early_blight`
  (recall 0.767, a *mid-range* 700 train images) — worse than `Potato___healthy`
  (recall 0.913, only 106 images). The confusion matrix confirms why: misclassified
  `Tomato___Early_blight` images are disproportionately predicted as
  `Tomato___Late_blight` (same finding for `Corn_(maize)___Cercospora_leaf_spot
  Gray_leaf_spot` → `Corn_(maize)___Northern_Leaf_Blight`) — the model is confusing
  visually similar diseases *within* the same plant species, not failing uniformly on
  rare classes. This points toward model capacity/normalization as a more targeted fix
  than weighted loss / `make_weighted_sampler` for this specific weakness — the
  imbalance handling built in Phase 2/3 remains available but isn't yet clearly
  justified by this evidence.
- **Bug caught and fixed during this phase:** `trainer.py` saved `last.pt` *before*
  updating `best_metric` for the current epoch, so a resumed run could start from a
  stale (lower) `best_metric` than what `best.pt` actually held — risking a later,
  worse epoch being wrongly accepted as a new best after a resume. Didn't affect this
  run's actual result (`best.pt` was always written correctly; only `last.pt`'s
  bookkeeping was stale, and the one resume this run needed landed with zero epochs
  left to run). Fixed by moving the `last.pt` save to after the best-metric check.

**Decision: iterate through all three deliberate changes (batchnorm, augmentation,
added depth) as separate runs, each compared back to this baseline, before starting
Phase 5.** Considered stopping here (96.8% is already strong, and PlantVillage has
limited headroom) or doing only one iteration (e.g. batchnorm alone) before moving to
transfer learning — chose full iteration instead, on the reasoning that Phase 4's own
goal is to *observe* what each change does, and the from-scratch CNN should be
reasonably well-iterated before Phase 8 compares it against fully-realized transfer
learning variants.

**Iteration 1 — BatchNorm (`phase4_simplecnn_batchnorm`, `SimpleCNNBatchNorm` in
`src/models/cnn.py`, `BatchNorm2d` after each conv, before the ReLU):**
- **Test accuracy 97.31%, test macro-F1 96.45%** (baseline: 96.82% / 95.91% —
  +0.49pp / +0.54pp). Best val macro-F1 0.9700 at epoch 28/30 — ran the full ceiling
  again, no early stopping.
- **Hypothesis confirmed:** the two classes targeted for their same-species
  disease-confusion pattern got the two biggest recall gains of all 38 classes —
  `Tomato___Early_blight` 0.767→0.853 (+8.7pp), `Corn_(maize)___Cercospora_leaf_spot
  Gray_leaf_spot` 0.857→0.961 (+10.4pp).
- **Not a uniform win.** `Corn_(maize)___Northern_Leaf_Blight` dropped 12.9pp (the
  biggest drop) — the exact class `Gray_leaf_spot` was being confused *as* in the
  baseline, so this reads as the decision boundary between the two shifting rather
  than the confusion being cleanly resolved. `Tomato___Late_blight` (-3.8pp) likely
  the same story relative to `Early_blight`. Two previously-perfect classes also
  slipped a few points — plausibly run-to-run variance (single run, not
  seed-averaged) rather than a systematic effect.
- **Decision: keep BatchNorm, carry it into the depth iteration** rather than testing
  added depth against the plain baseline — macro-F1 (the metric chosen for the
  imbalance concern) improved via the predicted mechanism, not by accident.

**Iteration 2 — Stronger augmentation (`phase4_simplecnn_augmented`, plain
`SimpleCNN`, notebook-local config copy — not a `configs/config.yaml` change —
with `rotation_degrees` 15→30 and `color_jitter` brightness/contrast/saturation
0.1→0.2 each):**
- **Test accuracy 95.83%, test macro-F1 94.75% — worse than the baseline on both
  metrics** (-0.99pp / -1.16pp). Best val macro-F1 only 0.9524 (epoch 25), below both
  other runs, with a visibly choppier val curve — consistent with added augmentation
  making training harder within this fixed 30-epoch budget, rather than reducing
  overfitting that wasn't really present in the baseline to begin with.
- **Mixed on the targeted hypothesis, net negative overall:**
  `Tomato___Early_blight` improved slightly (+3.3pp, far less than BatchNorm's
  +8.7pp), but `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` — the other
  targeted class — got *worse* (-7.8pp, second-biggest drop of any class). Unlike
  BatchNorm, not a "helps the target, costs elsewhere" tradeoff — a fairly clean net
  negative.
- **Decision: dropped.** No change to `configs/config.yaml` — Phase 2's augmentation
  defaults stay fixed through Phase 8. Still a useful result: ruled out a
  plausible-sounding idea with real evidence rather than assumption.

**Iteration 3 — Depth on top of BatchNorm (`phase4_simplecnn_batchnorm_deep`,
`SimpleCNNBatchNormDeep` in `src/models/cnn.py` — `SimpleCNNBatchNorm` plus a 4th
conv block, 128→256 channels, 28→14 spatial, continuing the same doubling/halving
pattern; not combined with augmentation, per Iteration 2's result):**
- **Test accuracy 98.97%, test macro-F1 98.46% — the clear best of all four
  variants** (+1.66pp / +2.01pp over BatchNorm alone; +2.15pp / +2.55pp over the
  original baseline). Best val macro-F1 0.9889 at epoch 26/30. Train loss (0.0355)
  and val loss (0.0352) essentially identical at epoch 29 — no overfitting despite
  ~4x the parameters (98K → 399K).
- **Depth added real value on top of BatchNorm, not just noise:**
  `Tomato___Early_blight` recall 0.767 → 0.853 → **0.960** across the three
  iterations — both changes helped, compounding rather than substituting for each
  other. Also partially recovered one of BatchNorm's regressions
  (`Corn_(maize)___Northern_Leaf_Blight` 0.946 → 0.816 → 0.918 — most of the way
  back, not fully). The handful of small drops vs. BatchNorm alone (all ≤3.2pp) are
  minor next to the gains.
- **The imbalance question is now about as resolved as it's going to get:**
  `Potato___healthy` — the original 36x-imbalance's smallest class (106 train
  images) — reached **100% recall** in this run. Not via weighted loss or
  `make_weighted_sampler` (neither ever adopted) — capacity and training stability
  mattered more for this dataset's actual failure mode (confusing similar diseases)
  than raw class frequency did.

**Final decision: `SimpleCNNBatchNormDeep` (`phase4_simplecnn_batchnorm_deep`) is
the Phase 4 result carried into the Phase 8 comparison.** Clear winner on both
metrics, no overfitting signal, and the only variant that directly addresses the
weakness the baseline's own confusion matrix identified. Augmentation and the
plain-BatchNorm variant are documented above but not carried forward.

**Phase 4 summary:** started from a deliberately minimal baseline (96.82% /
95.91%), iterated through all three planned candidates with real evidence for each
(BatchNorm kept, augmentation dropped, depth-on-BatchNorm adopted), and reached
98.97% / 98.46% test accuracy / macro-F1 — a +2.15pp / +2.55pp improvement over the
baseline, achieved through capacity and training-stability changes rather than
imbalance-specific handling. Ready for Phase 5.

---

## Phase 5 — Transfer learning: feature extraction

**Goal:** pretrained backbone, frozen entirely, train only a new classifier head.
Same harness, same split, same metrics as Phase 4.

**Backbone decision:** ResNet50 (torchvision, `IMAGENET1K_V2` weights) — chosen over
EfficientNet-B0 for its simpler, more standard block structure (`layer1`-`layer4`),
which keeps Phase 6's "unfreeze the last N blocks" straightforward to reason about.
This choice carries forward into Phase 6/7 as well, so all three transfer-learning
variants stay comparable to each other, not just to Phase 4.

**Status:** Done — `notebooks/05_feature_extraction.ipynb` (Colab GPU, T4).

**Deliverables:** `src/models/transfer.py` (`build_resnet50_feature_extractor`:
loads the pretrained ResNet50, freezes every backbone parameter, replaces `fc` with
a fresh trainable linear head), `notebooks/05_feature_extraction.ipynb` (loads Phase
2's fixed split unchanged, trains via the unmodified Phase 3 harness, evaluates on
the held-out test set, compares per-class recall against train-set frequency — same
shape as Phase 4's baseline section; no iterations, since Phase 5 is a single
well-defined configuration, not an exploration).

**Key decisions carried over from Phase 4:** unweighted `CrossEntropyLoss` (matches
Phase 4's baseline choice, so any accuracy/macro-F1 difference reflects the backbone
change, not a different imbalance-handling strategy); Adam optimizer, `lr=1e-3`,
`max_epochs=30`, `early_stopping_patience: 7` on `macro_f1` — identical to every
Phase 4 run. Optimizer is built over `filter(requires_grad)` so only `fc`'s 77,862
parameters (of ResNet50's 23,585,894 total — 0.33%) actually train.

**Local smoke test (CPU, 3 train + 2 val images/class subset, 1 epoch), run before
the real Colab training:** model builds correctly (77,862/23,585,894 trainable
params — confirms the backbone is fully frozen and only `fc` is trainable), harness
trains end-to-end, checkpoint round-trip verified exact (`val_macro_f1` matches to
<1e-6 on reload into a fresh model instance).

**Results (Colab T4 GPU, full split, `phase5_resnet50_feature_extraction`, 30
epochs, unweighted `CrossEntropyLoss`, Adam lr=1e-3):**
- **Test accuracy 97.88%, test macro-F1 97.30%** (best val macro-F1 during training:
  0.9742, epoch 23/30 — early stopping came close but never triggered: 6 of the
  required 7 non-improving epochs elapsed after epoch 23 before hitting the
  `max_epochs=30` ceiling). ~138 minutes total wall-clock training time (T4 GPU,
  1188 train batches/epoch) — despite only the 77,862-parameter head being
  trainable, every batch still needs a full forward pass through the 23.5M-parameter
  frozen backbone, so this wasn't the fast run a "just training a linear head"
  framing might suggest.
- **Beats Phase 4's unweighted baseline (96.82% / 95.91%) by +1.06pp / +1.39pp, but
  falls short of Phase 4's fully-iterated final result, `SimpleCNNBatchNormDeep`
  (98.97% / 98.46%), by -1.09pp / -1.16pp.** A frozen ImageNet backbone with only a
  linear head outperforms an *un-iterated* from-scratch CNN, but a from-scratch CNN
  that's been deliberately iterated (batchnorm + depth) on this exact dataset still
  wins outright. Not a surprising result in hindsight — PlantVillage's 38k+ training
  images give a from-scratch model plenty to learn from directly, and frozen generic
  ImageNet features (natural-image categories, not leaf-disease textures) aren't
  necessarily the closest match for this domain. This is exactly the kind of
  comparison Phase 8 exists to formalize across all four approaches.
- **`Tomato___Early_blight` is a chronic weak point across every approach so far, not
  just Phase 4's baseline.** Worst-recall class again (0.820, a *mid-range* 700
  train images) — same class that was the worst for every Phase 4 variant
  (baseline 0.767 → BatchNorm 0.853 → BatchNorm+Deep 0.960). Frozen-backbone feature
  extraction lands between Phase 4's baseline and its fully-iterated result on this
  specific class, consistent with the overall aggregate ranking above.
- **Imbalance still doesn't cleanly predict difficulty**, reinforcing Phase 4's
  finding with an independent architecture: `Potato___healthy` (the smallest class,
  106 train images) reaches 0.913 recall — not the worst — while
  `Tomato___Tomato_mosaic_virus` (only 261 train images) reaches a perfect 1.000.
  Same-species disease confusion, not raw frequency, remains the dominant failure
  mode.

**Next:** Phase 6 (partial fine-tuning) — unfreeze the last N blocks of this same
ResNet50 backbone, small learning rate, retrain. Ready to start.

---

## Phase 6 — Partial fine-tuning

**Goal:** unfreeze the last N blocks of the backbone, small learning rate, retrain.

**Unfreeze depth decision:** `layer4` only (ResNet50's last of four residual
blocks) — the conservative starting point. Keeps a clean progression across the
three transfer-learning phases: Phase 5 froze everything, Phase 7 unfreezes
everything, so Phase 6 sits in between as "just the last block." `layer1`-`layer3`
(generic low-level features: edges, textures) stay frozen.

**Status:** Done — `notebooks/06_partial_finetune.ipynb` (Colab GPU, T4).

**Deliverables:** `src/models/transfer.py` gained
`build_resnet50_partial_finetune` (same starting point as Phase 5's
`build_resnet50_feature_extractor`, but leaves `layer4` trainable alongside the
fresh `fc` head), `notebooks/06_partial_finetune.ipynb` (same shape as Phase 5's
notebook — loads Phase 2's fixed split, trains via the unmodified Phase 3 harness,
evaluates on the held-out test set, compares per-class recall against train-set
frequency).

**Key decision — discriminative learning rate:** the optimizer uses two parameter
groups instead of Phase 5's single one: `fc` at `lr=1e-3` (same as Phase 5),
`layer4` at `lr=1e-4` (10x lower). `layer4` already holds useful pretrained
features; a large gradient from the still-untrained, randomly-initialized head
early in training would otherwise risk overwriting them before the head has
learned anything useful to backpropagate. Everything else (loss, `max_epochs=30`,
`early_stopping_patience: 7`) matches Phase 4/5 unchanged.

**Local smoke test (CPU, 3 train + 2 val images/class subset, 1 epoch), run before
the real Colab training:** model builds correctly (15,042,598/23,585,894 trainable
params, matching `layer4` + `fc`'s combined size exactly), an explicit check
confirms `layer1`-`layer3` are still fully frozen, the two-param-group optimizer
works with the harness unchanged, checkpoint round-trip verified exact.

**Results (Colab T4 GPU, full split, `phase6_resnet50_partial_finetune`,
unweighted `CrossEntropyLoss`, discriminative Adam lr — `fc` 1e-3, `layer4`
1e-4):**
- **Test accuracy 99.68%, test macro-F1 99.36%** — the best result of any phase so
  far, beating even Phase 4's fully-iterated `SimpleCNNBatchNormDeep` (98.97% /
  98.46%). Best val macro-F1 0.9974 at epoch 17/30 — **early stopping actually
  triggered this time** (epoch 24, exactly 7 non-improving epochs after epoch 17),
  the first phase where it did; Phase 4 and Phase 5 both ran the full 30-epoch
  ceiling. ~108 minutes total training time (T4 GPU, 25 epochs) — faster overall
  than Phase 5's 138 minutes/30 epochs despite more trainable parameters per step,
  simply because it converged and stopped sooner.
- **Beats every prior approach on both metrics:** vs. Phase 5 (frozen backbone,
  97.88% / 97.30%): +1.80pp / +2.06pp. vs. Phase 4's iterated CNN (98.97% /
  98.46%): +0.71pp / +0.90pp. Letting just the last residual block adapt to this
  dataset's textures, on top of ImageNet's pretrained features, outperforms both
  "frozen generic features" (Phase 5) and "everything learned from scratch"
  (Phase 4) — consistent with the standard transfer-learning intuition that a
  little targeted adaptation on top of strong pretrained features beats either
  extreme when there's a reasonably large but not huge dataset to fine-tune on.
- **`Tomato___Early_blight` — the chronic weak class in both Phase 4 and Phase
  5 — is resolved.** Recall jumped to 0.993, no longer near the bottom of the
  per-class ranking at all. Unfreezing `layer4` let the network adapt its
  higher-level features to the specific textures distinguishing this disease,
  something Phase 5's frozen generic ImageNet features apparently couldn't do.
- **A new worst class emerged instead: `Corn_(maize)___Northern_Leaf_Blight`**
  (recall 0.939, 690 train images) — notably the same class Phase 4's BatchNorm
  iteration had *damaged* (0.946 → 0.816) as a side effect of fixing a different
  class's confusion, before partially recovering in the deep variant (→0.918).
  This class's difficulty looks architecture-independent rather than tied to any
  one approach's specific weakness — likely genuine visual similarity to
  `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` (the class Phase 4 found it
  swapped confusion with), a real property of the data Phase 8 should keep in mind.
- **Imbalance still doesn't predict difficulty:** `Potato___healthy` (the
  smallest class, 106 train images) reaches a perfect 1.000 recall.

**Next:** Phase 7 (full fine-tuning) — unfreeze the entire ResNet50 backbone, very
low learning rate. Ready to start.

---

## Phase 7 — Full fine-tuning

**Goal:** unfreeze the entire backbone, very low learning rate (typically a lower LR
on the backbone than the head).

**Status:** In progress — implemented and locally smoke-tested; the real 30-epoch
run still needs to happen on Colab GPU.

**Deliverables so far:** `src/models/transfer.py` gained `build_resnet50_full_finetune`
(same ResNet50 as Phase 5/6, but every parameter is trainable — nothing frozen),
`notebooks/07_full_finetune.ipynb` (same shape as Phase 5/6's notebooks — loads
Phase 2's fixed split, trains via the unmodified Phase 3 harness, evaluates on the
held-out test set, compares per-class recall against train-set frequency).

**Key decision — discriminative learning rate, continuing Phase 6's pattern:** two
parameter groups — `fc` at `lr=1e-3` (unchanged since Phase 5), the entire backbone
at `lr=1e-5` (10x lower than Phase 6's `layer4` rate of `1e-4`), since every
pretrained layer is moving now, not just the last block, so it needs the most
conservative rate of the three transfer-learning phases to avoid wrecking
ImageNet's pretrained features before the freshly-initialized head has learned
anything useful to backpropagate. Everything else (loss, `max_epochs=30`,
`early_stopping_patience: 7`) matches every previous phase unchanged.

**Local smoke test (CPU, 3 train + 2 val images/class subset, 1 epoch):** model
builds correctly (23,585,894/23,585,894 trainable — confirms nothing is frozen),
two-param-group optimizer split verified to cover every parameter exactly once
(23,508,032 backbone + 77,862 `fc` = the full total), harness trains end-to-end,
checkpoint round-trip verified exact. Plumbing confirmed correct; these numbers are
not meaningful results, just a harness check.

**Next step:** run `notebooks/07_full_finetune.ipynb` on Colab GPU for the real
30-epoch run, then fill in this section with actual test accuracy/macro-F1,
per-class recall findings (especially whether `Corn_(maize)___Northern_Leaf_Blight`
— Phase 6's worst class — remains difficult), and the comparison against Phase 6's
99.68%/99.36%, Phase 5's 97.88%/97.30%, and Phase 4's 98.97%/98.46% results.

---

## Phase 8 — Compare all approaches

**Goal:** one table/plot comparing Phases 4-7 on the identical held-out test set —
accuracy, macro-F1, training time, epochs to converge. The payoff of keeping the
harness identical since Phase 3.

**Status:** Not started.

---

## Phase 9 — Robustness check (optional)

**Goal:** run the best model on real-world / out-of-distribution images (e.g.
PlantDoc) to see how far accuracy drops outside PlantVillage's lab-condition,
plain-background photos. Not required for the core learning goal — a capstone if
there's time.

**Status:** Not started.
