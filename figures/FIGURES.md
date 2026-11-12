# Figures

All figures are built by `python scripts/make_figures.py` from saved runs in `runs/`; nothing is
retrained. Each is saved as PNG (200 dpi) and PDF. Colours are fixed per model throughout:
QTabGAN purple, Gaussian control orange, literal-paper QTabGAN light purple, CTGAN grey, real data
dark grey/black. Every metric is an error (lower is better) unless the axis says otherwise.

"Best" config per arm (used by fig1, fig2, fig3, fig4, fig5, figC, figD, figE) = the step-5 config
with the lowest mean plan score over its 5 test-split seeds (`scripts/training_length.py`):
`quantum_score2` (Optuna trial 4) and `gaussian_score2` (trial 36). That pick uses test results, so
these figures are descriptive; the pre-registered comparisons are in REPRODUCIBILITY.md Section 9.

`figures/preliminary/` holds earlier versions built from 2-seed runs (`runs/search_final`), kept
for the record; the figures in this folder supersede them.

---

## Main figures

### fig1_training_curves: training dynamics, best configs, 300 epochs
Generator loss, discriminator loss and the norm of the circuit's trainable angles over 300 epochs
(mean over 3 seeds, band = min–max). Both arms use WGAN-GP, so the loss scales are comparable, but
a GAN loss is not a quality measure. The circuit angles keep moving all the way through training
(norm about 1.7 to 6 rad), so the circuit is being trained, not sitting at its initial values.
- Runs: `runs/training_length/{quantum,gaussian}_s{0,1,2}/history.json`.
- Caveats: validation split (`--tune`). These runs also sample and evaluate every 25 epochs; that
  draws from the same random-number stream, so a run is not bit-identical to the same seed without
  evaluation.

### fig2_training_length: does training longer help?
Validation metrics every 25 epochs up to 300 epochs, with the no-learning floor and a marker at
150 epochs (the length used for every test-split run). Past 150 epochs the category-share error
(JSD) keeps falling by about 18% in both models, while the ML-utility gap and correlation error stay
flat within seed noise. So the 150-epoch results slightly understate the achievable JSD; they don't
understate the other two.
- Runs: `runs/training_length/*_s{0,1,2}/history.json`; floor from `runs/reference/result.json`.
- Caveats: 3 seeds. The curves are on the validation split, but the floor line was computed on the
  test split (it moves very little between splits). This evaluation uses 3 classifiers
  (lr/dt/rf), the step-5 numbers use 5. "Change" in the title = mean of the last 3 points vs the 3
  points around epoch 150, >10% in the same direction for both models.

### fig3_final_comparison: headline comparison (test split)
F1 gap, category-share error and correlation error for the best QTabGAN, the best Gaussian control
(5 seeds each), the literal-paper QTabGAN and CTGAN (2 seeds each), with floor and ceiling lines.
QTabGAN is better on category-share error, the Gaussian control is better on correlation error, and
the F1 gap is a tie (paired t-test over 5 seeds, Holm-corrected over these 3 metrics). Both searched
models are far better than the literal paper reading, whose bars are cut off and labelled.
- Runs: `runs/confirm/{quantum_score2,gaussian_score2}_s0-4`, `runs/search_final/paper_literal_s0-1`,
  `runs/final/ctgan_s0-1`, `runs/reference/result.json`.
- Caveats: the p-values here are corrected over 3 tests; corrected over all 18 comparisons of step
  5 (`runs/confirm/summary.md`), none of the best-vs-best differences is significant. The
  "best" pick uses test results (see top). Literal paper and CTGAN have only 2 seeds and come from
  earlier stages run with the same evaluation.

### fig4_real_vs_synthetic: what the synthetic columns look like
Real vs synthetic distributions for age and hours-per-week (histograms) and education and
occupation (category shares, sorted by real share). Each panel title gives the total variation
distance from the real column. Both models get the categorical shares within about 0.02, and both
miss the single dominant 40-hours-per-week value (47% of real rows), which is their largest error.
- Runs: `runs/confirm/{quantum_score2,gaussian_score2}_s0/fake.csv` vs the full real Adult table.
- Caveats: one seed per model; differences of 0.02 or less are treated as ties. No synthetic age
  exceeds 74 and hours stay within 8–80. This is the paper's preprocessing, which clips
  numeric columns to their 1st–99th percentiles (Sec. 4.1, `qtabgan/data.py`), so the generator
  cannot produce the tails and puts their mass at the clip value.

### fig5_correlation_heatmaps: relationships between columns
Association matrices (Pearson / Cramér's V / correlation ratio, the same measure as Diff. Corr)
for real data and each model (top), and the absolute difference from real (bottom). Both models
recover the main structure (age, marital status, relationship, gender, income). The largest errors
are capital-gain's link with income and native-country's links.
- Runs: as fig4.
- Caveats: seed 0 only. The "1.1x" in the title is one seed (0.43 vs 0.39); across 5 seeds the
  means are 0.43 vs 0.35 (fig3).

## Search figures

### figA_search_progress: Optuna search (step 3)
Validation score of every trial in the order run, with best-so-far lines. QTabGAN ends lower (0.19,
trial 28) than the Gaussian control (0.22, trial 14). Both arms had 21 completed and 19 pruned
trials.
- Runs: `runs/optuna/studies.db` (studies `score_quantum`, `score_gaussian`).
- Caveats: pruned trials are plotted at their epoch-20 score, completed ones at epoch 60, so the
  two marker types aren't directly comparable. The y-axis is cut at 1.15× the 90th percentile, so a
  few very poor trials are off the top. On the test split (step 5) the lead shrinks: plan
  score 0.222 vs 0.248 for the best configs, not significant.

### figB_screening: one-at-a-time screening (step 2)
For each setting, the level that moved the 2-seed validation score most relative to the starting
config, with ±2× seed-noise lines per arm. No single change improved QTabGAN beyond noise. Four
improved the Gaussian control: d_steps 5, linear categorical output, wider discriminator, wider
mapper. Large harms (softmax outputs, no circuit input) were removed from the Optuna search space.
- Runs: `runs/screen/{quantum,gaussian}/*/result.json`.
- Caveats: 40 epochs, 2 seeds per level, one setting at a time (interactions are not tested here;
  that is what step 3 is for).

## Extra "general ML" figures (4 models)

### figC_classifier_utility: train on synthetic, test on real
Accuracy, F1 and ROC AUC of 5 classifiers trained on each model's synthetic data and scored on the
same real test rows, next to the same classifier trained on real data. QTabGAN and CTGAN come
equally close on average (mean F1 gap 0.10). The literal paper reading is furthest, with some
classifiers near or below chance AUC.
- Runs: as fig3. Caveat: the accuracy axis starts at 50%.

### figD_training_curves_all: losses of every GAN variant, test-split runs
One column per model (rows: G loss, D loss, circuit-angle norm). The literal paper reading (BCE
loss) becomes spiky after about epoch 90 and its angle norm is about 30–37 rad. The searched models
settle smoothly. CTGAN does not log per-epoch losses.
- Runs: `runs/confirm/*_score2_s0-4/history.json`, `runs/search_final/paper_literal_s0-1`.
- Caveat: BCE and WGAN-GP losses are on different scales; each column has its own axis.

### figE_scorecard: all 6 metrics × 4 models
Mean ± sd for accuracy, F1 and AUC gaps, JSD, WD and Diff. Corr, coloured by rank within each
column, with floor and ceiling rows. The Gaussian control is best on 3 metrics, QTabGAN on 2 and
CTGAN on 1. Many of these differences are within seed noise (compare the ± values).
- Runs: as fig3.
- Caveat: colour is rank only, not significance. Note that the no-learning floor beats every model
  on JSD and WD, because those metrics only look at single columns.
