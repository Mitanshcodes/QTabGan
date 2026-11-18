# Teammate-style figures (Adult)

These figures follow the layout of our teammate's Loan-data poster, applied to our own study on the Adult census
dataset. Each one compares the **best QTabGAN** we found with the **Gaussian control** (the same network fed plain
random noise instead of the quantum circuit) and **CTGAN** (a standard classical generator). Where it makes sense,
they also show a **real-data reference**.

The question throughout: **does the quantum circuit help, compared with ordinary random noise?**

**How to read the numbers**
- **The task:** predict whether a person earns more than $50K a year. That group is the minority, 24% of rows.
- **"Synthetic data is useful"** means: a classifier trained only on the synthetic rows still predicts well on
  *real* people it has never seen.
- **The classifier:** unless a figure says otherwise, it is a random forest with 100 trees, tested on the real test
  set (9,769 rows the generators never saw).
- **The models:** both GANs use the settings our Optuna search picked (fig8). They were trained for 150 epochs with
  5 random seeds each.
- **Seed counts:** CTGAN and the two older QTabGAN versions are being rerun to 5 seeds. Until those finish, a "1
  seed" label marks numbers that come from one run.

Rebuild everything with `python scripts/make_teammate_figures.py` (the runs come from
`bash scripts/teammate_style.sh`).

---

### poster.png: one-page summary
The question, how the model works (diagram), the two main result figures, the conclusion and the limitations.
The dark banner gives the headline: with equal tuning, QTabGAN and the Gaussian control reach the same test
macro F1 (0.756 vs 0.756).

### fig1_training_300epochs.png: what happens during training
Six panels, each with one line per model. The line is the average over 3 seeds; the shaded band shows the best and
worst seed.
- **Top left, losses:** the generator and discriminator losses. A GAN's loss only shows that training is stable; it
  is not a quality score, so neither "up" nor "down" means better.
- **Top middle, circuit angles:** how far QTabGAN's 64 circuit parameters move. They keep moving through all 300
  epochs, so the circuit is really being trained.
- **Top right, cost:** cumulative training time. QTabGAN is about 5× slower.
- **Bottom row, quality over time:** the three quality errors. For all three, lower is better, and the dashed line
  is a "generator" that learns nothing.
  - **Bottom left:** utility gap.
  - **Bottom middle:** how well category shares match.
  - **Bottom right:** how well the relationships between columns match.

**What it shows:** most of the improvement happens in the first ~100 epochs. QTabGAN ends with better category
shares, and the Gaussian control with better column relationships.

**Note:** these runs use the validation split, not the test set.

*Coming when the runs finish:* a version with the circuit's gradient size, accuracy / F1 / AUROC / AP per epoch,
Brier score and diversity checks, logged every 10 epochs.

### fig2_roc_pr.png: ROC and precision-recall curves
A random forest is trained on each model's synthetic data, and one on the real training rows. Each is then asked to
flag the >50K earners in the real test set.
- **Left, ROC curve:**
  - **Axes:** the x-axis is the share of ≤50K people wrongly flagged; the y-axis is the share of >50K people
    correctly flagged.
  - **Reading it:** curves closer to the top-left corner are better. The dotted diagonal is random guessing.
- **Right, precision-recall curve:**
  - **Axes:** the x-axis is the share of >50K people found; the y-axis is how many flagged people really earn
    >50K.
  - **Reading it:** higher is better. The dotted line is the 24% you would get by guessing.
- **Legend:** AUROC and AP summarise each curve (1.0 = perfect).

**What it shows:** all three generators produce data that ranks people almost as well as real data (AUROC 0.87–0.88
vs 0.91). The gap is larger on precision-recall (AP about 0.69–0.70 vs 0.79). QTabGAN is not better than the Gaussian
control.

### fig3_cdfs.png: do the numeric columns look real?
Each panel shows one numeric column (age, hours per week, capital gain, census weight). A curve shows, for every
value x, the share of rows at or below x. If the synthetic curve lies on the black real-data curve, the column's
distribution was copied well.
- **KS number in the legend:** the biggest vertical gap between a model's curve and the real one (0 = identical).

**What it shows:**
- **Age:** all three models match it closely.
- **Hours per week:** QTabGAN and the Gaussian control smooth over the big jump at exactly 40 hours (the real data
  has a huge spike there); CTGAN keeps it.
- **Capital gain:** mostly zeros; every model gets this roughly right but not exactly.

### fig4_forest.png: summary across seeds
Every model in one plot. The square is the average, the bar is ±1 standard deviation, and the small dots are the
individual seeds.
- **Left, test macro F1 (higher is better):** how well a forest trained on that data predicts real incomes, averaged
  over both classes.
- **Right, our overall score (lower is better):** one number combining utility, category match and correlation
  match. The dashed line is the "learns nothing" generator.

**What it shows:**
- **Search matters more than the circuit.** The settings taken literally from the paper, and our first attempt, are
  far worse.
- **The two searched models are tied:** QTabGAN 0.756 vs Gaussian 0.756 F1.

### fig5_results_table.png / results_table.md: the numbers
Accuracy, macro F1, AUROC, average precision and recall of >50K for each data source. The table also has the
real-data reference and the "always guess ≤50K" baseline, which is right 76% of the time but finds nobody earning
>50K.

**What it shows:**
- **Real data is still best on accuracy.**
- **QTabGAN data finds the most >50K earners** (recall 0.73), but at lower accuracy. It trades precision for recall,
  so this is not an overall win.
- **The paper-literal settings are worse than always guessing ≤50K on accuracy.**

### fig6_diversity.png: do the models produce varied rows?
- **Left, variation ratio:** within each income group, how spread-out the synthetic numeric columns are compared with
  the real ones (100% = same spread).
- **Right, category coverage:** the share of all real category values (every job, education level, country...) that
  appear at least once in the synthetic data.

**What it shows:**
- **The literal-paper and first-pass settings collapse.** Their rows are nearly identical (under 20% of real
  variation), and most categories never appear.
- **The searched models keep 84–86% of real variation.**
- **Category coverage:** CTGAN generates every category, QTabGAN ~90%, the Gaussian control ~77%.

### fig7_optuna.png: the hyperparameter search
- **Left, search history:** each dot is one finished trial (lower score = better); the step line is the best found
  so far.
- **Middle and right, parameter importance:** how much each setting mattered, from an fANOVA analysis of the 21
  completed trials per model.

**What it shows:**
- **QTabGAN's search reached a lower validation score** (0.192 vs 0.217). On the test set that lead becomes small
  and not statistically significant.
- **What mattered most:** batch size and discriminator steps for QTabGAN; learning rate and discriminator steps for
  the Gaussian control.
- **The quantum-only settings mattered little** (noise encoding, circuit learning rate, angle initialisation).
- **Caveat:** with 21 trials the importances are a rough guide only.

### fig8_settings.png: settings used, ours vs the paper
The exact settings of the two models used in every figure, next to what the paper says. Red cells are settings the
paper does not give (11 of 17), which is why we had to search for them.
