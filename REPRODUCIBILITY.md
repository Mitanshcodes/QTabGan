# Reproducing QTabGAN without the authors' code

QTabGAN: Kumari, Achutha, Sivaraman, *A Hybrid Quantum-Classical GAN for Tabular Data Synthesis*,
arXiv:2602.12704 (v1 Feb 2026, v2 Mar 2026). No code has been released, so this document records
what can be recovered exactly, what can't, what we chose instead, and the problems we found.

## TL;DR

- **The evaluation half is fully recoverable.** QTabGAN's classical baseline numbers (CTAB-GAN+
  5.23% / 0.090 / JSD 0.039 / Diff. Corr 2.03; CTAB-GAN 8.90% / 0.107) are copied verbatim from the
  CTAB-GAN+ paper's Table III (arXiv:2204.00401). CTAB-GAN+'s evaluation code, data file and split
  are public, so we re-implemented that protocol step by step.
- **The model half is not.** Of the 29 settings in our implementation, the paper states 10, is
  ambiguous about 5 and silent on 14 (layers, shots, optimizer, learning rates, epochs, batch size,
  loss, network widths, how categories are output, how noise enters the circuit, seeds, simulator).
- **Several things suggest QTabGAN's numbers are not comparable to the baselines they are compared
  against** (Section 4). The most concrete: their real-data accuracy on Adult (77.6%) is impossible
  under the CTAB-GAN+ protocol, where real data scores 80.5–86.2%.
- **Result (Adult, 2 seeds):** our faithful QTabGAN does not beat a classical control built the same
  way (same networks, Gaussian noise instead of the circuit). With our first-pass settings every
  quantum variant was worse on correlation structure than a generator that learns no correlations
  at all (Section 6). A 20-config random search per arm (Section 7) made the best QTabGAN beat that
  floor, but the searched classical control still won on every metric (correlation error 0.82 vs
  2.23). CTGAN beats both.
- **Update (Section 9, pre-registered Optuna search, 5 test seeds):** with an equal 40-trial
  search per arm, QTabGAN and the Gaussian control are statistically tied, and both now beat CTGAN
  on JSD and correlation error. The paper's Adult gaps can be matched under a guessed kNN
  evaluation, but best by the Gaussian control.
- So "reproducing" here means: a faithful implementation, unknowns tuned fairly on a validation
  split, the exact evaluation protocol, and controlled comparisons. It does **not** mean matching
  their decimals, which nobody can do without their code.

## 1. Reproduced exactly: the evaluation protocol (from CTAB-GAN+'s public code)

| item | value | source |
|---|---|---|
| data file | `Real_Datasets/Adult.csv`, 48,842 rows, 14 columns, categoricals label-encoded | CTAB-GAN+ repo |
| split | `train_test_split(test_size=0.2, stratify=income, random_state=42)`: 39,073 train / 9,769 test | `model/pipeline/data_preparation.py` and `model/eval/evaluation.py` |
| generator trains on | the 39,073 train rows only | same |
| synthetic rows sampled | 48,842 (= len(full csv)) | `ctabgan.py: generate_samples` |
| classifiers | lr (max_iter 500), dt, rf, mlp (max_iter 100), svm (SVC), all random_state=42 | `Experiment_Script_Adult.ipynb`, `evaluation.py` |
| scaling | MinMax fit on real train; synthetic gets its own MinMax fit on all synthetic X | `evaluation.py` |
| synthetic training rows | 80% of the synthetic table (same stratified split call) | `evaluation.py` |
| ML-utility number | real score minus synthetic score, per classifier, then averaged | `evaluation.py` + paper text |
| F1 | mean of per-class F1 (binary) | `evaluation.py` |
| "Avg JSD" | `scipy.spatial.distance.jensenshannon(base=2)` = JS **distance** | `evaluation.py` |
| "Diff. Corr" | Frobenius norm of (assoc(real) − assoc(fake)); dython: Pearson / Cramér's V / correlation ratio | `evaluation.py` |
| stat similarity compares | the full real csv (incl. test rows) vs the synthetic table | `evaluation.py` |

Our only change: SVC is fit with `probability=False` and AUC uses `decision_function`. For a binary
target Platt scaling is monotone, so accuracy, F1 and AUC are unchanged; it is ~5x faster.

CTAB-GAN+'s *paper* and *code* differ in small ways (paper: Theil's U, linear SVM, max_depth 28,
MLP 128 units; code: Cramér's V via dython's default, RBF SVC, default depth, MLP default 100). We
follow the code, because code is what produced numbers.

**Real-data scores under this protocol (Adult test set):** lr 80.47%, dt 81.29%, rf 86.24%,
mlp 83.64%, svm 81.50% (mean 82.63%). Majority-class baseline: 76.07%.

## 2. Every setting: paper vs. ours

`paper` = stated, `ambiguous` = stated but readable more than one way, `unstated` = our choice.
Generated from `qtabgan/config.py` (`python -m qtabgan.config`). Defaults shown; the final runs
override some of them (Section 3).

| setting | default | source | note |
|---|---|---|---|
| `n_qubits` | `8` | paper | Sec. 5.4: "its 8-qubit circuit" -> 256-dim probability vector. |
| `n_layers` | `4` | unstated | Paper only says "L layers". |
| `n_shots` | `1024` | ambiguous | Paper: "N > 2^n" shots per sample, no value given. |
| `init_hadamard` | `True` | paper | Hadamard on every qubit before the variational layers. |
| `rotations` | `ry_rz` | paper | One RY and one RZ per qubit per layer. |
| `entanglement` | `ring` | paper | CNOT(i, (i+1) mod n) after each layer. |
| `angle_init` | `uniform_0_2pi` | paper | Angles "chosen randomly in [0, 2pi)". |
| `noise_encoding` | `none` | ambiguous | Sec. 3 describes no circuit input (diversity only from finite shots); Sec. 4.2 says the generator "samples random noise and encodes it into the quantum circuit" without saying how. Options: none | angle (RY(z_i), z_i ~ U[0, pi/2) per qubit, after the Hadamards). |
| `prob_scale` | `n_outcomes` | unstated | How the probability vector is scaled before the mapper. n_outcomes multiplies by 2^n so entries average 1 (comparable to the one-hot label); none keeps raw probabilities. |
| `gradient` | `backprop_straight_through` | ambiguous | Paper uses parameter-shift. On a simulator we backprop through the exact circuit probabilities (same derivative, computed faster) and pass the shot-noisy value forward (straight-through). |
| `backend` | `torch` | unstated | Simulator. torch = our batched statevector sim (verified equal to PennyLane in tests, ~10x faster); pennylane = PennyLane default.qubit reference. |
| `source` | `quantum` | paper | quantum = QTabGAN. Controls (ours): gaussian = classical N(0, I) noise of the same width; frozen_quantum = same circuit with angles never trained. |
| `mapper_dims` | `(256, 256)` | unstated | Paper: feed-forward, ReLU, linear output; widths not given. |
| `disc_dims` | `(256, 256)` | unstated | Paper: feed-forward, ReLU, linear scalar output; widths not given. |
| `cat_output` | `softmax` | unstated | Paper: mapper output is linear in R^d; no handling of one-hot blocks stated. Options: linear (literal) | softmax | gumbel (per categorical block). |
| `condition_on` | `target` | paper | One-hot class label concatenated to mapper and discriminator input. |
| `loss` | `bce` | ambiguous | Paper gives the standard minimax (log) GAN objective but names no loss. bce = that objective with the usual non-saturating generator loss. wgan_gp offered as an alternative. |
| `epochs` | `150` | unstated | Not stated. 150 matches what CTAB-GAN+ used for Adult. |
| `batch_size` | `500` | unstated | Not stated. |
| `lr_classical` | `0.0002` | unstated | Adam lr for mapper + discriminator. Optimizer not stated. |
| `lr_quantum` | `0.01` | unstated | Separate Adam lr for circuit angles (our choice). |
| `betas` | `(0.5, 0.9)` | unstated | Adam betas. |
| `d_steps` | `1` | unstated | Discriminator updates per generator update. |
| `clip_percentiles` | `(1, 99)` | paper | Numeric features clipped to 1st/99th percentile. |
| `numeric_range` | `zscore_then_minmax_0_pi` | paper | z-score, then min-max into [0, pi]. |
| `input_noise_var` | `1e-05` | ambiguous | Paper: add N(0, 10^-5); read as variance, applied once. |
| `categorical_encoding` | `one_hot` | paper | One-hot for categorical features. |
| `seed` | `0` | unstated | Paper does not report seeds or number of runs. |
| `device` | `cpu` | unstated | Paper: "standard simulators", none named. |

## 3. How we filled the gaps

All tuning used a **validation split carved out of the 39,073 train rows** (the same CTAB-GAN+
split function applied again: 31,258 for training, 7,815 held out). The real test set was never
used for any decision. Every tuning run: 40 epochs, seed 0, quick evaluation (lr/dt/rf).

**Stage 1: shared settings** (category output × loss), quantum generator with no circuit input,
1,024 shots. **Stage 2: quantum settings** (noise encoding × shots) on top of the stage-1 winner,
plus a classical Gaussian-noise control with the same shared settings.

**Selection rule:** average rank across three metric families: utility (F1 gap), marginals (avg
JSD), joint structure (Diff. Corr). Lower is better. *Honest note:* we fixed this rule after
seeing stage 1 (we had first sorted by F1 alone, which picked a run whose categorical columns had
mostly collapsed to one value) and before running stage 2. It ranks over all 13 runs; `s1_gumbel_wgan_gp`
and `s2_none_shots1024` are the same config and gave identical numbers (determinism check).

| run | avg rank | acc | f1 | auc | avg_jsd | avg_wd | diff_corr | js_div (paper def) | avg abs dPearson (paper def) | train s |
|---|---|---|---|---|---|---|---|---|---|---|
| s2_gaussian_control | 3.67 | 6.72 | 0.167 | 0.215 | 0.116 | 0.039 | 3.85 | 0.0173 | 0.152 | 85 |
| s2_angle_shots300 | 4.33 | 4.95 | 0.178 | 0.113 | 0.448 | 0.119 | 2.60 | 0.2368 | 0.082 | 445 |
| s2_none_shots300 | 4.33 | 6.01 | 0.223 | 0.165 | 0.340 | 0.129 | 3.63 | 0.1536 | 0.134 | 213 |
| s2_angle_shots8192 | 5.00 | 13.51 | 0.238 | 0.200 | 0.427 | 0.065 | 3.16 | 0.2356 | 0.110 | 501 |
| s2_angle_shots1024 | 5.33 | 5.40 | 0.254 | 0.205 | 0.392 | 0.057 | 3.32 | 0.1958 | 0.115 | 464 |
| s2_none_shots1024 | 6.17 | 5.96 | 0.265 | 0.187 | 0.301 | 0.081 | 3.83 | 0.1174 | 0.138 | 236 |
| s1_gumbel_wgan_gp | 6.17 | 5.96 | 0.265 | 0.187 | 0.301 | 0.081 | 3.83 | 0.1174 | 0.138 | 230 |
| s2_none_shots8192 | 7.33 | 4.41 | 0.228 | 0.183 | 0.350 | 0.073 | 4.81 | 0.1546 | 0.181 | 256 |
| s1_gumbel_bce | 8.33 | 5.90 | 0.284 | 0.275 | 0.441 | 0.157 | 3.35 | 0.2499 | 0.121 | 208 |
| s1_linear_bce | 9.00 | 8.60 | 0.129 | 0.098 | 0.616 | 0.106 | 4.94 | 0.4840 | 0.320 | 135 |
| s1_softmax_wgan_gp | 10.33 | 7.18 | 0.257 | 0.193 | 0.591 | 0.089 | 4.60 | 0.4515 | 0.339 | 154 |
| s1_softmax_bce | 10.33 | 5.48 | 0.270 | 0.226 | 0.575 | 0.176 | 4.19 | 0.4288 | 0.171 | 141 |
| s1_linear_wgan_gp | 10.67 | 18.49 | 0.279 | 0.175 | 0.466 | 0.146 | 4.43 | 0.2826 | 0.243 | 143 |

**What stage 1 showed:** with shot noise as the only randomness, the generator **mode-collapses**.
In the best-F1 run (linear output, BCE), education, occupation and race each collapsed to a single
value and the other categorical columns to one or two. In the softmax+BCE run,
hours-per-week was exactly 80 for every row and age had std 1.4 (real: 13.7). Gumbel output
"fixes" the categorical columns only because Gumbel sampling injects its own classical noise at the
output.

**What stage 2 showed:** the two readings of the paper's circuit input (no input, Sec. 3; noise
encoded into the circuit, Sec. 4.2) tie exactly under the selection rule (both best with 300
shots), so **both go into the final experiment** rather than picking one arbitrarily. At 40 epochs
the classical Gaussian control beats every quantum configuration on F1 gap and JSD, and trains
2.5–6x faster; the quantum versions are better only on Diff. Corr.

**Final settings:** Gumbel category output (tau 0.2), WGAN-GP loss (lambda 10), 300 shots, 4
layers, 8 qubits, Adam lr 2e-4 (classical) / 1e-2 (angles), betas (0.5, 0.9), batch 500,
150 epochs (CTAB-GAN+'s budget for Adult), seeds 0 and 1 (seed 2 was stopped partway for time).

## 4. Problems found in the paper

1. **Their real-data baseline doesn't fit the protocol their comparison numbers come from.** For
   Adult they report real accuracy 0.7760 / F1 0.7619. Under CTAB-GAN+'s protocol, no classifier
   scores below 80.47% on real data (mean 82.63%), and 77.6% is barely above the 76.07% majority
   baseline. So their ML-utility evaluation differs from CTAB-GAN+'s, yet the resulting gaps are
   compared against CTAB-GAN+'s Table III. The paper never names its downstream classifier(s).
   A probe (`scripts/probe_real_accuracy.py`, output in `runs/probe_real_accuracy.md`) trained 8
   common classifiers on real Adult data with default settings and 3 scalings. The closest matches
   to 0.776 / 0.762 are k-nearest-neighbours on unscaled data (0.778 accuracy, 0.758 weighted F1)
   and logistic regression on unscaled data (0.790, 0.762 weighted F1). This suggests, but does not
   prove, a single untuned classifier on unscaled features with weighted F1.
2. **The similarity metrics are defined differently from the numbers they're compared with.**
   CTAB-GAN+'s "Avg JSD" is the JS *distance* and its "Diff. Corr" is the *Frobenius norm* of the
   association-matrix difference. QTabGAN's Sec. 5.2.2 defines JSD as the JS *divergence* (= distance²)
   and Diff. Corr as the *average* absolute Pearson difference. These live on different scales.
   On Adult, a "generator" that samples every column independently (no learning at all) scores
   Diff. Corr 2.71 under CTAB-GAN+'s definition but 0.076 under QTabGAN's. QTabGAN reports 0.35:
   near-perfect under the first reading, ~5x worse than the no-learning floor under the second.
   (Their figure is a 5-dataset average, so this is suggestive, not proof.) Likewise their JSD 0.028
   would be excellent as a distance but poor as a divergence.
3. **The baselines were not re-run.** The CTAB-GAN+, CTAB-GAN, CTGAN, TableGAN, MedGAN and CW-GAN
   numbers match CTAB-GAN+'s Table III (arXiv v1) exactly: 2021-era runs, averaged over Adult,
   Covertype, Credit, Intrusion and Loan. IT-GAN and TVAE are not in that version (possibly from the
   2024 journal version; the paper doesn't say). Our own CTGAN run on Adult (Section 6) is far better
   than the CTGAN numbers quoted (accuracy gap 21.5%, F1 0.274, Diff. Corr 2.73), so the quoted
   baselines are a weak yardstick.
4. **Two incompatible descriptions of the circuit input.** Sec. 3.1 describes a circuit with no
   input (angles initialised randomly, then trained). Sec. 4.2 says the generator "samples random
   noise and encodes it into the quantum circuit". We implement and test both.
5. **Regression conditioning** (King, Insurance) is not described: the method conditions on a class
   label, which regression targets don't have.

*Corrections to our own first read.* Two things we initially flagged are not problems.
(a) Avg JSD 0.028 / Diff. Corr 0.35 in Sec. 5.3.2 vs 0.05 / 0.40 in Table 2 is not a contradiction:
the table compares QTabGAN with TabularQGAN on the subset of features TabularQGAN can model, as the
paper says. (b) "The generator seeks to maximize the discriminator's loss" (Sec. 4.2) is just the
minimax game of Sec. 2.5, not a conflict.

## 5. Reference points: what the metrics mean on Adult

Same evaluation, no training. **Floor**: every column (incl. the label) sampled independently from
its training marginal; all relationships destroyed. **Ceiling**: the generator's own training rows
used as the synthetic table.

| metric (lower is better) | floor: independent marginals | ceiling: copy of train rows |
|---|---|---|
| accuracy gap (pts) | 9.34 | 0.18 |
| F1 gap | 0.276 | 0.001 |
| AUC gap | 0.337 | 0.001 |
| Avg JSD (CTAB-GAN+ def = JS distance) | 0.0061 | 0.0022 |
| Avg WD | 0.0007 | 0.0004 |
| Diff. Corr (CTAB-GAN+ def = Frobenius norm) | 2.710 | 0.029 |
| JS divergence (QTabGAN paper def) | 0.00006 | 0.00001 |
| avg abs Pearson diff (QTabGAN paper def) | 0.0757 | 0.0018 |

Two consequences for reading any tabular-GAN paper:
- **Avg JSD and Avg WD only check one column at a time**, so the no-learning floor nails them
  (JSD 0.006, better than CTAB-GAN+'s reported 0.039). Low JSD is not evidence of a good generator.
- **Accuracy gaps are compressed by class imbalance.** The floor's accuracy gap is only 9.3 points
  but its F1 gap is 0.276. F1 gap and Diff. Corr are the informative columns.

## 6. Final results

Seeds 0 and 1 for every arm (seed 2 was stopped partway for time; `SEEDS=2 bash scripts/final.sh`
adds it). All arms share the tuned settings (Gumbel category output, WGAN-GP, 300 shots, 150 epochs)
and the exact CTAB-GAN+ evaluation. Values are mean +- sd over the 2 seeds; **lower is better**.
`quantum` = QTabGAN with shot noise as the only randomness (paper Sec. 3); `quantum_angle` = noise
encoded into the circuit (Sec. 4.2); `frozen_quantum` = same circuit, angles never trained;
`gaussian` = the same mapper and discriminator fed classical N(0, I) noise (159,595 vs 159,659
parameters: the only difference is QTabGAN's 64 circuit angles); `ctgan` = vanilla CTGAN.

| metric | ctgan [2] | frozen_quantum [2] | gaussian [2] | quantum_angle [2] | quantum [2] | floor: marginals | ceiling: real copy |
|---|---|---|---|---|---|---|---|
| acc | 4.40 +- 0.82 | 7.98 +- 2.84 | 4.29 +- 0.09 | 9.86 +- 5.49 | 11.91 +- 2.54 | 9.34 | 0.18 |
| f1 | 0.101 +- 0.061 | 0.227 +- 0.054 | 0.125 +- 0.046 | 0.202 +- 0.090 | 0.069 +- 0.012 | 0.276 | 0.001 |
| auc | 0.057 +- 0.002 | 0.248 +- 0.113 | 0.107 +- 0.015 | 0.184 +- 0.002 | 0.091 +- 0.016 | 0.337 | 0.001 |
| avg_jsd | 0.106 +- 0.003 | 0.361 +- 0.128 | 0.170 +- 0.010 | 0.367 +- 0.027 | 0.417 +- 0.013 | 0.006 | 0.002 |
| avg_wd | 0.012 +- 0.001 | 0.072 +- 0.019 | 0.036 +- 0.007 | 0.098 +- 0.027 | 0.088 +- 0.039 | 0.001 | 0.000 |
| diff_corr | 0.67 +- 0.06 | 4.57 +- 1.14 | 1.69 +- 0.19 | 3.99 +- 0.54 | 3.05 +- 0.01 | 2.71 | 0.03 |
| js_div (paper def) | 0.0133 +- 0.0003 | 0.1799 +- 0.1090 | 0.0437 +- 0.0094 | 0.1709 +- 0.0209 | 0.2093 +- 0.0191 | 0.0001 | 0.0000 |
| avg abs dPearson (paper def) | 0.036 +- 0.001 | 0.207 +- 0.069 | 0.070 +- 0.028 | 0.179 +- 0.076 | 0.146 +- 0.018 | 0.076 | 0.002 |

What holds in **both** seeds (with n = 2 we claim nothing else; full per-run numbers and paired
tests are in `runs/final/summary.md`):

- **QTabGAN does not beat the classical control built the same way.** Both quantum readings are
  worse than `gaussian` on correlation structure (Diff. Corr 3.05 and 3.99 vs 1.69) and on category
  distributions (JSD 0.42 and 0.37 vs 0.17). The shot-noise version has a better F1 gap (0.069 vs
  0.125) but a much worse accuracy gap (11.9 vs 4.3), consistent with a near-collapsed generator
  producing easily separable classes.
- **Every quantum arm is worse on correlations than a generator that learns none** (floor 2.71).
  In our implementation this cuts against the paper's attribution of its correlation results to
  "the use of entangling quantum circuits". The classical control and CTGAN both beat the floor.
- **Training the circuit does matter a little:** trained beats frozen on F1 gap and Diff. Corr in
  both seeds (not on JSD). So the circuit isn't inert; it just loses to plain Gaussian noise.
- **CTGAN is best on almost everything** (Diff. Corr 0.67, JSD 0.106) and far better than the 2021
  CTGAN numbers QTabGAN quotes as its baseline.
- **Cost:** on 2 CPU cores, 17–18 min (quantum), 34–35 min (quantum_angle), 11 min (gaussian),
  14–15 min (CTGAN) per 150-epoch run.

**What this does and doesn't show.** We could not reproduce QTabGAN's advantage under 12 tuning
configurations and the paper's two possible circuit-input readings. That is evidence, not proof:
14 settings are unstated and 5 ambiguous, and the authors' choices might behave differently. If
they share code or settings, `scripts/run.py` takes them as flags and the same evaluation applies.

## 7. Random search over the unknowns (both arms, same budget)

To turn "we guessed the unknowns" into "we searched them", `scripts/search.sh` ran 20 random
configurations for QTabGAN and 20 for the Gaussian control (60 epochs, validation split, test set
untouched). Config i shares network sizes, classical learning rate, loss and batch size across both
arms; the quantum arm also varies layers (2/4/6), shots (300/1024), circuit learning rate
(1e-3/1e-2/5e-2), circuit input (none/angle), noise range (pi/2 or pi) and starting angles
(uniform [0, 2pi) or near zero, the latter from our teammates' version). Full table:
`runs/search/summary.md`.

**Paired result (20 pairs, same shared settings):** the circuit beat Gaussian noise in 3/20 pairs
on F1 gap, 2/20 on category JSD and 3/20 on correlation error (Wilcoxon p = 0.001, < 0.001, 0.002).
Two Gaussian configs were sampled twice (identical settings, identical scores), so the Gaussian arm
effectively had 18 distinct configs: a small handicap that makes the comparison conservative.

**One exception in the search, which did not hold up:** QTabGAN config 12 had the lowest correlation
error of all 40 runs (1.64 vs the best Gaussian 1.69), while losing on JSD and F1. Retrained on the
real split for 150 epochs it scored 2.56 and 1.89 (two seeds).

**Final stage** (`scripts/search_final.sh`): the best config of each arm by the same average-rank
rule, plus the most literal reading of the paper (no circuit input, linear outputs, standard GAN
loss), retrained on the real CTAB-GAN+ split, 150 epochs, seeds 0 and 1, full evaluation:

| metric | best_gaussian [2] | best_quantum [2] | paper_literal [2] | floor: marginals | ceiling: real copy |
|---|---|---|---|---|---|
| acc | 5.34 +- 1.53 | 9.72 +- 2.52 | 7.77 +- 0.96 | 9.34 | 0.18 |
| f1 | 0.160 +- 0.036 | 0.213 +- 0.011 | 0.237 +- 0.011 | 0.276 | 0.001 |
| auc | 0.090 +- 0.010 | 0.241 +- 0.051 | 0.231 +- 0.055 | 0.337 | 0.001 |
| avg_jsd | 0.048 +- 0.005 | 0.376 +- 0.003 | 0.580 +- 0.192 | 0.006 | 0.002 |
| avg_wd | 0.037 +- 0.025 | 0.036 +- 0.009 | 0.124 +- 0.059 | 0.001 | 0.000 |
| diff_corr | 0.82 +- 0.09 | 2.23 +- 0.47 | 5.15 +- 0.18 | 2.71 | 0.03 |
| js_div (paper def) | 0.0040 +- 0.0021 | 0.1756 +- 0.0047 | 0.4277 +- 0.2435 | 0.0001 | 0.0000 |
| avg abs dPearson (paper def) | 0.036 +- 0.002 | 0.087 +- 0.001 | 0.359 +- 0.056 | 0.076 | 0.002 |

What holds in both seeds:
- **The searched-best Gaussian control beats the searched-best QTabGAN on every metric** except
  avg WD (a tie): correlation error 0.88 / 0.75 vs 2.56 / 1.89, category JSD 0.045 / 0.052 vs
  0.378 / 0.374, F1 gap 0.135 / 0.185 vs 0.206 / 0.221.
- **Searching helped QTabGAN:** its best config now beats the no-learning floor on correlations
  (2.56 and 1.89 vs 2.71; the untuned versions in Section 6 did not). It just helped the classical
  control more (1.69 to 0.82).
- **The literal paper reading is the worst model tested:** correlation error 5.03 / 5.28, about
  twice the floor.
- **The best QTabGAN configs barely use the circuit:** they encode classical noise into it, start
  the angles near zero and train them with a very small learning rate, so the circuit acts as a
  fixed transform of classical noise.

**Bottom line of the search:** under a 20-config budget per arm, chosen and judged the same way,
the quantum circuit does not beat Gaussian noise in the same network. This is still one dataset
and 2 final seeds.

## 8. Questions for the authors

See `AUTHOR_EMAIL.md`.

## 9. Systematic search for the unstated settings

Sections 6–7 guessed, then randomly searched, the unknowns. This section does it with a plan
written before any run (`runs/optuna/PLAN.md`, changes logged there with date and reason) and
two questions:

- **(A)** Is there any reasonable config that reproduces the paper's Adult numbers (real accuracy
  0.7760, synthetic 0.7740, gap 0.002, weighted-F1 gap about 0.008)?
- **(B)** Under an equal-budget search, does the quantum noise source beat the same network fed
  Gaussian noise?

Rules: tuning and selection use the validation split only (`--tune`). The test split is touched
only in step 5. The Gaussian control gets the same budget, sampler, pruner and selection rule, and
every shared setting is searchable in both arms. The CTAB-GAN+ evaluation is unchanged; new
evaluation modes are separate functions. Every run saves its config and metrics.

### Steps

1. **Screening** (`scripts/screen.sh`, 112 runs, 40 epochs, 2 seeds): one setting changed at a time
   from the Section 7 winners. No single change improved QTabGAN beyond seed noise. Four improved the
   Gaussian control (5 discriminator steps, linear categorical output, wider discriminator, wider
   mapper). Clearly harmful levels (softmax outputs, no circuit input, ...) were dropped from the
   search space (`runs/optuna/space.json`). Figure: `figures/figB_screening.png`.
2. **Optuna, quality objective** (`scripts/optuna_search.py`): 40 trials per arm, 60 epochs. TPE
   sampler; successive-halving pruner identical in both arms (min resource 20 epochs, factor 3).
   Objective = mean of F1 gap, avg JSD and Diff. Corr, each divided by a fixed scale (lower is
   better).
3. **Optuna, paper objective**: same budget and space. The objective is the distance to the paper's
   gaps under `paper_guess_utility`: one kNN (k=5) on unscaled features, which is the simple
   classifier that reproduces the paper's real-data accuracy (`runs/probe_real_accuracy.md`).
4. **Confirmation** (`scripts/confirm.py`): the top 2 trials per arm by quality and the closest
   trial per arm by paper distance (6 configs). Retrained on the real CTAB-GAN+ split, 150 epochs,
   5 seeds each, full evaluation on the test split.

### Search (validation split)

| study | completed | pruned | best value | best trial |
|---|---|---|---|---|
| quality, QTabGAN | 21 | 19 | 0.192 | 28 |
| quality, Gaussian | 21 | 19 | 0.217 | 14 |
| paper, QTabGAN | 20 | 20 | 0.0043 | 15 |
| paper, Gaussian | 17 | 23 | 0.0022 | 11 |

Pruning hit both quality studies equally. It hit the Gaussian paper study slightly harder (23 vs 20).
Tables: `runs/optuna/report_score.md`, `runs/optuna/report_paper.md`. Figure:
`figures/figA_search_progress.png`.

### Confirmation (test split, 150 epochs, 5 seeds, mean ± sd)

From `runs/confirm/summary.md`. ML-utility rows are real minus synthetic over lr/dt/rf/mlp/svm.
Every row is lower-is-better.

| metric | Q score #1 | Q score #2 | Q paper | G score #1 | G score #2 | G paper | floor | ceiling |
|---|---|---|---|---|---|---|---|---|
| acc gap (%) | 3.60 ± 0.81 | 4.32 ± 0.75 | 4.58 ± 1.42 | 4.17 ± 0.61 | 4.25 ± 0.66 | 3.72 ± 0.57 | 9.34 | 0.18 |
| F1 gap | 0.115 ± 0.023 | 0.102 ± 0.015 | 0.106 ± 0.020 | 0.128 ± 0.011 | 0.123 ± 0.015 | 0.122 ± 0.030 | 0.276 | 0.001 |
| AUC gap | 0.041 ± 0.009 | 0.044 ± 0.004 | 0.065 ± 0.017 | 0.047 ± 0.004 | 0.056 ± 0.005 | 0.045 ± 0.010 | 0.337 | 0.001 |
| avg JSD | 0.027 ± 0.002 | 0.028 ± 0.002 | 0.097 ± 0.007 | 0.025 ± 0.002 | 0.034 ± 0.001 | 0.048 ± 0.004 | 0.006 | 0.002 |
| avg WD | 0.012 ± 0.005 | 0.014 ± 0.005 | 0.023 ± 0.004 | 0.015 ± 0.002 | 0.011 ± 0.002 | 0.021 ± 0.008 | 0.001 | 0.000 |
| Diff. Corr | 0.40 ± 0.03 | 0.43 ± 0.05 | 1.23 ± 0.19 | 0.44 ± 0.04 | 0.35 ± 0.04 | 0.48 ± 0.03 | 2.71 | 0.03 |
| plan score | 0.235 ± 0.037 | 0.222 ± 0.021 | 0.389 ± 0.031 | 0.255 ± 0.018 | 0.248 ± 0.020 | 0.276 ± 0.046 | 0.706 | 0.007 |
| train time (min) | 259 | 142 | 180 | 36 | 13 | 107 | | |

Train times are for 6 runs in parallel on one CPU machine, so read them only relative to each other.

Paper-style evaluation on the same runs (kNN on unscaled features). Our real-data accuracy is
0.7781; the paper reports 0.7760.

| config | accuracy gap (pp) | weighted-F1 gap | distance to the paper's gaps |
|---|---|---|---|
| paper's reported numbers | 0.2 | ~0.008 | 0 |
| G paper | −0.36 ± 0.38 | 0.0069 ± 0.0029 | 0.0077 ± 0.0053 |
| G score #1 | 0.01 ± 0.71 | 0.0118 ± 0.0039 | 0.0097 ± 0.0046 |
| Q score #1 | 0.76 ± 0.26 | 0.0140 ± 0.0023 | 0.0115 ± 0.0047 |
| G score #2 | 0.89 ± 1.46 | 0.0161 ± 0.0074 | 0.0186 ± 0.0189 |
| Q score #2 | 1.24 ± 0.86 | 0.0158 ± 0.0066 | 0.0182 ± 0.0148 |
| Q paper | 2.05 ± 2.05 | 0.0233 ± 0.0113 | 0.0338 ± 0.0318 |

### Answers

**(A) Yes, under a guessed evaluation, but the Gaussian control does it best.** Under the
paper-style kNN evaluation, several configs land at the paper's numbers. The Gaussian config
searched for this purpose gives an accuracy gap of −0.36 ± 0.38 pp and an F1 gap of 0.007 ± 0.003
over 5 test seeds (paper: 0.2 pp and about 0.008). The QTabGAN config searched the same way is the
furthest of all six, with a large seed-to-seed spread (0.008–0.081). The difference between the
two paper configs is not significant (paired t-test p = 0.17).

Two things limit this answer:
- **The evaluation is a guess.** It is the simple classifier that reproduces the paper's real-data
  accuracy, not a documented protocol.
- **Small gaps here are cheap.** kNN on unscaled features mostly sees fnlwgt and the capital
  columns, so it is a weak classifier. Under the CTAB-GAN+ protocol the same paper-matching
  configs still trail real data by 3.6–4.6 accuracy points.

So matching the paper's decimals says little about data quality, and nothing in it requires the
quantum circuit.

**(B) No evidence that the quantum noise source helps.** Comparing configs of equal search rank
(paired by seed, plan score):

| pair | mean difference (Q − G) | QTabGAN wins | p (uncorrected) |
|---|---|---|---|
| score #1 vs #1 | −0.021 | 3 of 5 seeds | 0.38 |
| score #2 vs #2 | −0.026 | 3 of 5 seeds | 0.22 |

On individual metrics, after the plan's Holm correction over all 18 step-5 comparisons, no
quality-config QTabGAN-vs-Gaussian difference is significant. The only significant results:
QTabGAN's paper-objective config has worse correlation error than every Gaussian config (Holm
p ≤ 0.018).

The direction is mixed. QTabGAN tends to have a lower F1 gap; the Gaussian control's best config
has the lowest correlation error. Correcting over only the 3 headline metrics for the two "best"
configs (fig3; the pick itself used test results):
- QTabGAN is better on JSD (0.028 vs 0.034).
- Gaussian is better on Diff. Corr (0.35 vs 0.43).
- F1 is a tie.

Meanwhile QTabGAN takes 1.7–11× as long to train as the matching Gaussian config.

### What changed from Sections 6–7

- **The systematic search improved QTabGAN a lot.** Plan score: literal paper reading 1.51,
  Section 7 winner 0.94, now 0.22. Diff. Corr went from 5.15 to 0.43 and JSD from 0.58 to 0.028.
- **The searched QTabGAN and Gaussian control now beat CTGAN** (2 seeds) on JSD and correlation
  error, and match it on F1 gap (`figures/figE_scorecard.png`).
- **Section 7's large Gaussian advantage was mostly an under-searched quantum arm.** With an equal
  40-trial budget and pruning, the two arms are statistically tied.
- **Section 7's remark that the best QTabGAN configs barely use the circuit does not hold here.**
  The step-5 winners use angle encoding, and the circuit angles keep moving through training
  (norm about 2 to 5 rad over 150 epochs; `figures/figD_training_curves_all.png`).

### Training length (descriptive, validation split)

Each arm's best step-5 config was retrained for 300 epochs, evaluated every 25 (3 seeds;
`runs/training_length/`, `figures/fig2_training_length.png`). Past 150 epochs, JSD falls about
another 18% in both arms; F1 gap and Diff. Corr stay flat. Training longer would not change
answer B.

### Limits

- **Scope:** one dataset (Adult) and 5 seeds per config.
- **Simulation only:** a noiseless state-vector simulator with shot sampling; no hardware.
- **Seed and pairing:** the CTGAN and literal-paper comparisons have 2 seeds. Seed pairing
  matches seed indices, not identical random streams.
- **Choices made by us, not the paper:** 40 trials per arm, 60 epochs during the search, and the
  search space itself, which was narrowed by screening.
- **Clipping:** the paper's 1st–99th percentile clipping caps synthetic numeric columns (age ≤ 74,
  hours 8–80). This affects both arms equally.

Figures with captions and caveats: `figures/FIGURES.md`.
