# Project details: reproducing QTabGAN

This document describes the whole project end to end: what we set out to test, how the code works,
every stage of experiments in the order we ran them, what each stage found, and what the results
do and don't support. For the full numeric record see `REPRODUCIBILITY.md`; for figure captions see
`figures/FIGURES.md`.

---

## 1. The goal

**QTabGAN** (Kumari, Achutha, Sivaraman, *A Hybrid Quantum-Classical GAN for Tabular Data Synthesis*,
arXiv:2602.12704) generates synthetic tabular data with a GAN whose generator starts from a
variational quantum circuit. The paper reports strong results, for example on the Adult census
dataset: a classifier trained on real data scores 0.7760 accuracy and one trained on QTabGAN's
synthetic data scores 0.7740 (a gap of only 0.2 percentage points).

No code was released, so we wrote an independent implementation and asked two questions:

- **(A) Reproduction:** is there any reasonable setting of the paper's unstated hyperparameters
  that reproduces its Adult result (accuracy gap 0.002, weighted-F1 gap about 0.008)?
- **(B) Attribution:** does the *quantum* part actually help? We compare QTabGAN with the
  identical network fed ordinary Gaussian noise instead of the circuit's output, giving both the
  same tuning budget.

---

## 2. What we built

### Model (`qtabgan/`)

```
[noise source] -> 2^n vector --+
                               +--> classical mapper (MLP) --> synthetic row
one-hot class label -----------+
discriminator: [row ; label] --> MLP --> real/fake score
```

- **Noise source**, selectable with `--source`:
  - `quantum`: QTabGAN. An 8-qubit variational circuit (Hadamards, optional noise-encoding RY
    rotations, then layers of trainable RY/RZ rotations and a ring of CNOTs). The circuit is
    measured with a finite number of shots, and the 256-outcome probability vector feeds the mapper.
    Simulated in PyTorch (`qtabgan/sim.py`), checked against PennyLane.
  - `frozen_quantum`: the same circuit with its angles never trained.
  - `gaussian`: a plain N(0, I) vector of the same width. This is the **control**. Everything
    else (mapper, discriminator, losses, data pipeline) is identical, so the only difference is
    QTabGAN's 64 circuit angles.
- **Data pipeline** (`qtabgan/data.py`), following paper Sec. 4.1: numeric columns clipped to their
  1st–99th percentiles, z-scored, scaled into [0, π]; categorical columns one-hot encoded.
- **Configurable unknowns** (`qtabgan/config.py`): each of the 29 settings is tagged as stated by the
  paper, ambiguous, or unstated. The paper states 10, is ambiguous on 5 and silent on 14 (layers,
  shots, optimiser, learning rates, epochs, batch size, loss, network widths, categorical output,
  how noise enters the circuit, seeds, simulator, and more).

### Evaluation (`qtabgan/evaluate.py`)

We reproduced the **CTAB-GAN+ evaluation protocol** exactly from its public code. QTabGAN compares
itself against CTAB-GAN+'s published numbers, so this is the natural yardstick.

- **Data:** Adult, 48,842 rows. Stratified 80/20 split (`random_state=42`): 39,073 train rows,
  9,769 test rows. The generator only ever sees the train rows.
- **ML utility (train on synthetic, test on real):** 5 classifiers (logistic regression, decision
  tree, random forest, MLP, SVM) are each trained once on real data and once on synthetic data, then
  scored on the same real test rows. We report **real minus synthetic**, averaged over the 5, for
  accuracy, F1 and AUC.
- **Statistical similarity:**
  - **avg JSD:** category-share error per categorical column (JS distance).
  - **avg WD:** numeric-column error (Wasserstein distance).
  - **Diff. Corr:** how much the column-to-column relationships differ. It is the Frobenius norm
    of the difference between the real and synthetic association matrices (Pearson / Cramér's V /
    correlation ratio).
- **Reference points** (`scripts/reference_points.py`):
  - **Floor:** a "generator" that samples every column independently, so it learns no
    relationships.
  - **Ceiling:** the real training rows passed off as synthetic data.
- **Paper-style evaluation** (`paper_guess_utility`, added as a separate function; the CTAB-GAN+
  evaluation was never changed). A single kNN classifier (k=5) on unscaled features, weighted F1.
  We chose it because a probe of 8 common classifiers (`scripts/probe_real_accuracy.py`) found it
  reproduces the paper's real-data accuracy (0.778 vs the paper's 0.776). Under CTAB-GAN+'s
  protocol, no classifier scores below 80.5% on real data.

| reference (test split) | accuracy gap (pp) | F1 gap | avg JSD | Diff. Corr |
|---|---|---|---|---|
| floor (no learning) | 9.34 | 0.276 | 0.006 | 2.71 |
| ceiling (real data) | 0.18 | 0.001 | 0.002 | 0.03 |

Note that the floor has an almost perfect JSD, because JSD only looks at one column at a time. F1 gap
and Diff. Corr are the informative metrics.

### Problems we found in the paper (details: `REPRODUCIBILITY.md` Section 4)

1. **The real-data baseline doesn't match the protocol it is compared against.** The paper's
   real-data accuracy (77.6%) is impossible under the protocol its baseline numbers come from, and
   the paper never names its classifier.
2. **The similarity metrics are defined differently from the baseline numbers** they are compared
   with: JS divergence vs JS distance, and average |ΔPearson| vs a Frobenius norm. These live on
   different scales.
3. **The baselines were not re-run.** They are copied from CTAB-GAN+'s 2021 table.
4. **The paper describes the circuit input in two incompatible ways.** Sec. 3.1 has no input;
   Sec. 4.2 encodes noise into the circuit. We implemented and tested both.

---

## 3. What we did, stage by stage

All tuning used a **validation split** cut from the training rows (31,258 train / 7,815 validation;
flag `--tune`). The real test split was used only in final stages. Every run writes its config,
metrics and training history to `runs/`.

### Stage 1: first-pass tuning (`scripts/tune.sh`, 13 runs, 40 epochs)

Hand-picked grid over categorical output × loss, then the circuit-input reading × shots. Findings:

- With shot noise as the only randomness, the generator **mode-collapses**. Whole categorical
  columns came out as a single value.
- At this budget the Gaussian control beat every quantum variant on F1 gap and JSD, and trained
  2.5–6× faster.

### Stage 2: first comparison (`scripts/final.sh`, test split, 150 epochs, 2 seeds)

QTabGAN (both circuit-input readings), a frozen circuit, the Gaussian control and CTGAN.

- **The Gaussian control beat both quantum readings on correlation structure.** Diff. Corr was
  1.69 vs 3.05 / 3.99.
- **Every quantum variant was worse than the no-learning floor on correlations** (floor 2.71).
- **Training the circuit helped a little:** the trained circuit beat the frozen one.
- **CTGAN was best on almost everything** (Diff. Corr 0.67).

### Stage 3: random search (`scripts/search.sh`, `scripts/search_final.sh`)

20 random configs per arm, sharing the classical settings across arms, then each arm's winner and
the most literal reading of the paper retrained on the test split (2 seeds).

| model | F1 gap | avg JSD | Diff. Corr |
|---|---|---|---|
| best Gaussian | 0.160 | 0.048 | 0.82 |
| best QTabGAN | 0.213 | 0.376 | 2.23 |
| literal paper settings | 0.237 | 0.580 | 5.15 |

Searching helped QTabGAN (it now beat the floor on correlations), but it helped the control more.
The literal paper reading was the worst model tested.

### Stage 4: pre-registered systematic search (the main result)

Because 20 random configs might simply under-search the quantum arm, we wrote a plan **before any
run** (`runs/optuna/PLAN.md`; later changes are logged there with date and reason) and followed it.

**Rules:**
- Tuning and selection use the validation split only.
- The test split is touched only in the confirmation step.
- The Gaussian control gets the same budget, sampler, pruner and selection rule, and every shared
  setting is searchable in both arms.
- Every run saves its config and metrics; long jobs are resumable.

**Objective** (lower is better). A single score that weights utility, marginals and correlations
equally, using fixed scales:

```
score = ( F1 gap / 0.2365  +  avg JSD / 0.3299  +  Diff. Corr / 2.906 ) / 3
```

**Steps:**

1. **Screening** (`scripts/screen.sh`, 112 runs, 40 epochs, 2 seeds per config). One setting
   changed at a time from the stage-3 winners, to find which settings matter and drop clearly
   harmful levels from the search space.
   - No single change improved QTabGAN beyond seed noise.
   - Four improved the Gaussian control: 5 discriminator steps, linear categorical output, a wider
     discriminator and a wider mapper.
2. **Optuna search, quality objective** (`scripts/optuna_search.py`).
   - 40 trials per arm, 60 epochs each.
   - TPE sampler.
   - Successive-halving pruning, identical in both arms (first check at epoch 20, reduction
     factor 3). Pruned trials count towards the 40.
3. **Optuna search, paper objective.** Same budget and space, but the objective is the distance to
   the paper's gaps under the paper-style kNN evaluation:
   `|accuracy gap − 0.002| + |F1 gap − 0.008|`.
4. **Confirmation** (`scripts/confirm.py`).
   - Configs: the top 2 trials per arm by quality and the closest trial per arm by paper distance
     (6 configs).
   - Each retrained on the real split for 150 epochs, 5 seeds, full evaluation on the **test**
     split.
   - Paired comparisons by seed index.
5. **Training length** (`scripts/training_length.py`). Each arm's best confirmed config retrained
   for 300 epochs on the validation split (3 seeds), evaluated every 25 epochs with a new
   `--eval-every` flag. This is descriptive only, because the "best" pick used test results.

Steps 2–4 run end to end with `bash scripts/optuna_pipeline.sh`, which resumes where it stopped.
On one CPU machine with 6 parallel jobs, the full pipeline took about 17 hours.

---

## 4. Results

### Search (validation split)

| study | completed | pruned | best value |
|---|---|---|---|
| quality, QTabGAN | 21 | 19 | 0.192 |
| quality, Gaussian | 21 | 19 | 0.217 |
| paper, QTabGAN | 20 | 20 | 0.0043 |
| paper, Gaussian | 17 | 23 | 0.0022 |

### Confirmation (test split, 150 epochs, 5 seeds, mean ± sd, lower is better)

| metric | QTabGAN #1 | QTabGAN #2 | QTabGAN paper | Gaussian #1 | Gaussian #2 | Gaussian paper |
|---|---|---|---|---|---|---|
| accuracy gap (pp) | 3.60 ± 0.81 | 4.32 ± 0.75 | 4.58 ± 1.42 | 4.17 ± 0.61 | 4.25 ± 0.66 | 3.72 ± 0.57 |
| F1 gap | 0.115 ± 0.023 | 0.102 ± 0.015 | 0.106 ± 0.020 | 0.128 ± 0.011 | 0.123 ± 0.015 | 0.122 ± 0.030 |
| avg JSD | 0.027 ± 0.002 | 0.028 ± 0.002 | 0.097 ± 0.007 | 0.025 ± 0.002 | 0.034 ± 0.001 | 0.048 ± 0.004 |
| Diff. Corr | 0.40 ± 0.03 | 0.43 ± 0.05 | 1.23 ± 0.19 | 0.44 ± 0.04 | 0.35 ± 0.04 | 0.48 ± 0.03 |
| plan score | 0.235 ± 0.037 | 0.222 ± 0.021 | 0.389 ± 0.031 | 0.255 ± 0.018 | 0.248 ± 0.020 | 0.276 ± 0.046 |
| train time (min) | 259 | 142 | 180 | 36 | 13 | 107 |

Full table with AUC, WD and paired tests: `runs/confirm/summary.md`.

### How much the systematic search improved QTabGAN

| QTabGAN version | plan score | Diff. Corr | avg JSD |
|---|---|---|---|
| literal paper settings | 1.51 | 5.15 | 0.580 |
| random-search winner (stage 3) | 0.94 | 2.23 | 0.376 |
| **Optuna winner (stage 4)** | **0.22** | **0.43** | **0.028** |
| CTGAN, for comparison (2 seeds) | 0.33 | 0.67 | 0.106 |

### Answer to (A): can the paper's Adult result be reproduced?

**Yes, under a guessed evaluation, but the closest match comes from the Gaussian control, not
QTabGAN.** Under the paper-style kNN evaluation (our real-data accuracy 0.7781 vs the paper's
0.7760):

| config (5 test seeds) | accuracy gap (pp) | weighted-F1 gap |
|---|---|---|
| **paper's reported numbers** | **0.2** | **~0.008** |
| Gaussian, paper objective | −0.36 ± 0.38 | 0.0069 ± 0.0029 |
| Gaussian, quality #1 | 0.01 ± 0.71 | 0.0118 ± 0.0039 |
| QTabGAN, quality #1 | 0.76 ± 0.26 | 0.0140 ± 0.0023 |
| QTabGAN, paper objective | 2.05 ± 2.05 | 0.0233 ± 0.0113 |

Caveats:
- **The kNN evaluation is our inference**, not a documented protocol.
- **Small gaps under it are cheap.** kNN on unscaled features is a weak classifier. Under the
  CTAB-GAN+ protocol the same configs still trail real data by 3.6–4.6 accuracy points.

Matching the paper's decimals therefore says little about data quality, and nothing about it
requires the quantum circuit.

### Answer to (B): does the quantum noise source help?

**No evidence that it does.** With equal tuning budgets, QTabGAN and the Gaussian control are
statistically tied.

| pair (same search rank, paired by seed) | mean score difference (QTabGAN − Gaussian) | QTabGAN wins | p |
|---|---|---|---|
| quality #1 vs #1 | −0.021 | 3 of 5 | 0.38 |
| quality #2 vs #2 | −0.026 | 3 of 5 | 0.22 |

- **No significant difference on any individual metric.** After Holm correction over all 18
  comparisons, no QTabGAN-vs-Gaussian difference between the quality configs is significant.
- **The direction is mixed.** QTabGAN tends to have a lower F1 gap and slightly lower JSD; the
  Gaussian control has the lowest correlation error.
- **The one significant result:** QTabGAN's paper-objective config is worse on correlations than
  every Gaussian config.
- **Cost:** QTabGAN takes 1.7–11× as long to train.

### Other findings

- **Training past 150 epochs** lowers category-share error by about 18% more in both arms; F1 gap
  and correlation error stay flat (`figures/fig2_training_length.png`).
- **The best QTabGAN really does use its circuit.** It encodes noise into the circuit and its angles
  keep changing throughout training (norm about 2 → 5 rad).
- **The literal paper settings are unstable.** Their losses become spiky after about epoch 90, and
  some classifiers trained on their data score at or below chance AUC.
- **Both models miss Adult's dominant 40-hours-per-week value** (47% of real rows).
- **Synthetic numeric columns cannot leave the paper's 1st–99th percentile clip range.** No
  synthetic age above 74.

---

## 5. Figures

All in `figures/`, built from saved runs with `python scripts/make_figures.py` (PNG at 200 dpi plus
PDF). Captions, source runs and caveats are in `figures/FIGURES.md`.

| figure | shows |
|---|---|
| fig1_training_curves | losses and circuit-angle norm over 300 epochs |
| fig2_training_length | validation metrics every 25 epochs, does training longer help |
| fig3_final_comparison | headline test-split comparison with floor and ceiling |
| fig4_real_vs_synthetic | real vs synthetic histograms and category shares |
| fig5_correlation_heatmaps | association matrices and their errors |
| figA_search_progress | Optuna search, best-so-far per arm, pruned trials |
| figB_screening | effect of each setting in screening |
| figC_classifier_utility | per-classifier accuracy / F1 / AUC (train on synthetic, test on real) |
| figD_training_curves_all | losses of every GAN variant |
| figE_scorecard | all 6 metrics × 4 models, ranked |

---

## 6. Limits

- **One dataset** (Adult) and 5 seeds per confirmed config. CTGAN and the literal-paper runs have
  only 2 seeds.
- **Simulation only:** a noiseless simulator with shot sampling, no quantum hardware.
- **Our own budget choices:** 40 trials per arm, 60 search epochs, and a search space narrowed by
  screening. A larger search could change details.
- **Unknown settings:** 14 settings are unstated and 5 ambiguous in the paper. Our conclusions are
  about *our* faithful implementation under a fair search, not proof about the authors' code. If
  they release code or settings, `scripts/run.py` accepts them as flags and the same evaluation
  applies.

## 7. How to reproduce

```
bash scripts/get_data.sh                       # Adult.csv from the CTAB-GAN+ repo
python scripts/run.py --source quantum ...     # one run (see --help for every setting)
bash scripts/screen.sh                         # step 1: screening
bash scripts/optuna_pipeline.sh                # steps 2-4: Optuna studies + confirmation (resumable)
bash scripts/training_length.sh                # step 5: training-length experiment
python scripts/make_figures.py                 # all figures
python scripts/summarize.py final runs/confirm runs/reference/result.json
```

Generated synthetic tables (`fake.csv`) are not committed (see `.gitignore`), so figures 4 and 5 need
the confirmation runs to be regenerated. Every other figure rebuilds from the committed results.
