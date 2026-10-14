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
  way (same networks, Gaussian noise instead of the circuit), and every quantum variant is worse on
  correlation structure than a generator that learns no correlations at all. CTGAN beats both.
  Section 6 has the numbers and caveats.
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

| run | avg rank | acc | f1 | auc | avg_jsd | avg_wd | diff_corr | js_div (paper def) | avg|dPearson| (paper def) | train s |
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
| avg|dPearson| (paper def) | 0.036 +- 0.001 | 0.207 +- 0.069 | 0.070 +- 0.028 | 0.179 +- 0.076 | 0.146 +- 0.018 | 0.076 | 0.002 |

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

## 7. Questions for the authors

See `AUTHOR_EMAIL.md`.
