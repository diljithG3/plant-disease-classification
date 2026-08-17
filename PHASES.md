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
| 5 | Transfer learning: feature extraction | Not started |
| 6 | Partial fine-tuning | Not started |
| 7 | Full fine-tuning | Not started |
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

**Goal:** pretrained backbone (ResNet or EfficientNet — TBD when we get there),
frozen entirely, train only a new classifier head. Same harness, same split, same
metrics as Phase 4.

**Status:** Not started.

---

## Phase 6 — Partial fine-tuning

**Goal:** unfreeze the last N blocks of the backbone, small learning rate, retrain.

**Status:** Not started.

---

## Phase 7 — Full fine-tuning

**Goal:** unfreeze the entire backbone, very low learning rate (typically a lower LR
on the backbone than the head).

**Status:** Not started.

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
