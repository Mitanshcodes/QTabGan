# Plan: systematic search for the settings QTabGAN leaves unstated

Written 2026-11-11, **before any run of this plan**. Rules: tuning and selection use the validation
split only (`scripts/run.py --tune`); the real test split is touched only in step 5. Both arms
(QTabGAN `quantum`, control `gaussian`) get the same budget, the same pruner and the same selection
rule. Changes after this point are logged at the bottom with date and reason.

Machine: Apple M4 (4 performance + 6 efficiency cores), 16 GB, CPU only. 6 jobs in parallel, 1
torch thread each (measured: about 3.2x the throughput of one job).

## Starting points (from `runs/search_final/commands.txt`, the 20-config random search winners)

- **quantum base (Q\*)**: `--mapper-dims 512,512 --disc-dims 256,256 --lr-classical 0.0005 --loss bce
  --batch-size 256 --cat-output gumbel --n-layers 4 --n-shots 300 --lr-quantum 0.001
  --noise-encoding angle --noise-scale 1.0 --angle-init small_normal`
- **gaussian base (G\*)**: `--mapper-dims 256,256 --disc-dims 512,512 --lr-classical 0.0005
  --loss wgan_gp --batch-size 256 --cat-output gumbel`
- Everything else: `qtabgan/config.py` defaults (8 qubits, betas 0.5/0.9, d_steps 1, ...).

## Objective (used for screening effects and for the step-3 studies)

Three metrics, all "lower is better", on the validation split with quick evaluation (lr/dt/rf):
F1 gap (`ml_utility_mean_diff.f1`), avg JSD (`stat_sim.avg_jsd`), Diff. Corr (`stat_sim.diff_corr`).
Each is divided by a **fixed scale = its median over the 40 runs in `runs/search/`** (both arms
pooled, 60 epochs, same split and evaluation), so each term is ~1 for a typical run and none
dominates:

    score = ( f1_gap / 0.2365  +  avg_jsd / 0.3299  +  diff_corr / 2.906 ) / 3      (lower is better)

Fixed scales instead of ranks, because a rank depends on the other trials and Optuna needs a
per-trial number. The scales are pooled over both arms so the same score is used for both.

## Step 2: one-at-a-time screening

Each variant changes one setting of the base and keeps the rest. 40 epochs, `--tune --eval quick`,
seeds 0 and 1. The base itself is run with both seeds too.

Settings not screened: those the paper states (Hadamard, RY+RZ, ring CNOT, clipping, [0, pi]
scaling, one-hot, label conditioning), implementation choices that don't change the model
(`gradient`, `backend`), `epochs` (fixed by this plan), `seed`. Gumbel tau (0.2) and the
gradient-penalty weight (10) are hard-coded and not screened.

| setting | base Q\* | base G\* | levels tried | arms |
|---|---|---|---|---|
| n_qubits (= noise width for gaussian) | 8 | 8 | 4, 6 | both |
| n_layers | 4 | - | 1, 2, 6, 8 | quantum |
| n_shots | 300 | - | 100, 1024, exact (0) | quantum |
| lr_quantum | 1e-3 | - | 1e-4, 1e-2 | quantum |
| noise_encoding | angle | - | none | quantum |
| noise_scale | 1.0 | - | 0.5, 2.0 | quantum |
| angle_init | small_normal | - | uniform_0_2pi | quantum |
| prob_scale | n_outcomes | - | none | quantum |
| mapper_dims | 512,512 | 256,256 | 128,128 / 1024,1024 (+ the other base's value) | both |
| disc_dims | 256,256 | 512,512 | 128,128 / 1024,1024 (+ the other base's value) | both |
| lr_classical | 5e-4 | 5e-4 | 2e-4, 1e-3 | both |
| loss | bce | wgan_gp | the other one | both |
| batch_size | 256 | 256 | 128, 500 | both |
| cat_output | gumbel | gumbel | softmax, linear | both |
| betas | 0.5,0.9 | 0.5,0.9 | 0.5,0.999 | both |
| d_steps | 1 | 1 | 2, 5 | both |
| input_noise_var | 1e-5 | 1e-5 | 0, 1e-3 | both |

Budget: 35 quantum configs and 21 gaussian configs (base included), x 2 seeds = 112 runs
(`runs/screen/commands.txt`). Estimated ~1.5-2 h (the quantum base, with noise encoded into the
circuit, costs ~6.6 s/epoch alone).

**Ranking table** (`scripts/screen.py summarize`): per setting and arm, for each metric the largest
change of the 2-seed mean versus the base's 2-seed mean, and the same for `score`.
**Seed noise** sigma = per-run standard deviation of `score`, estimated from the seed-0 vs seed-1
difference of every screened config in that arm (sigma = rms(diff) / sqrt 2). The difference of
two 2-seed means then has standard error sigma.

**Selection rule:** a setting "matters" if its largest |change in score| exceeds 2 sigma in
either arm. Settings that matter go into step 3; shared settings that matter in either arm are
searched in **both** arms (rule 2). `noise_encoding`, `angle_init` and `noise_scale` are always
searched for the quantum arm. Settings that don't matter stay at each arm's base value.

## Step 3: Optuna, one study per arm (`scripts/optuna_search.py`)

- TPE sampler (seed 0, `n_startup_trials=10`, `constant_liar=True` for parallel workers),
  **40 trials per arm**. Trial 0 of each arm is its base config (enqueued), same for both.
- Each trial: validation split, seed 0, up to 60 epochs, objective = `score` above at epoch 60.
- **Pruning: successive halving**, `SuccessiveHalvingPruner(min_resource=20, reduction_factor=3,
  min_early_stopping_rate=0)`, identical for both arms. No evaluation before epoch 20; the rungs are
  at epochs 20 and 60. At epoch 20 a trial continues only if its `score` is in the top third of
  the trials of that study that reached epoch 20. GAN metrics early in training are noisy and don't
  improve steadily, so this is deliberately the mildest halving schedule.
- Pruned trials count towards the 40. **Pruned-trial counts per arm are recorded and reported**,
  because QTabGAN configs train more slowly and could be hit harder.
- Search space: the settings selected in step 2 with these ranges (filled in from step 2's result,
  before the studies start): n_qubits {4, 6, 8}; n_layers int 1-8; n_shots {100, 300, 1024, exact};
  lr_quantum log-uniform [1e-4, 1e-1]; noise_encoding {none, angle}; noise_scale [0.25, 2.0];
  angle_init {uniform_0_2pi, small_normal}; prob_scale {n_outcomes, none}; mapper_dims and
  disc_dims {128, 256, 512, 1024} (2 layers); lr_classical log-uniform [5e-5, 2e-3]; loss
  {bce, wgan_gp}; batch_size {128, 256, 500}; cat_output {gumbel, softmax, linear}; betas
  {(0.5, 0.9), (0.5, 0.999)}; d_steps {1, 2, 5}; input_noise_var {0, 1e-5, 1e-3}.
- Studies stored in SQLite (`runs/optuna/studies.db`); every trial writes its config, per-check
  metrics and history to `runs/optuna/<study>/trial_NNN/`. Restarting the script resumes the study;
  trials left RUNNING by a crash are marked FAIL and do not count towards the 40.

## Step 4: matching the paper's numbers (question A)

New evaluation function `paper_guess_utility` (added to `qtabgan/evaluate.py` as a separate
function; the CTAB-GAN+ functions are not touched): ONE classifier, **kNN (k=5) on unscaled
features** (closest to the paper's real-data numbers in `runs/probe_real_accuracy.md`); trained on
the real train part / on the synthetic table (same 80% stratified split call as CTAB-GAN+), scored on
the same held-out real rows; reports accuracy and weighted F1. Unscaled logistic regression is also
reported, not optimised.

Objective (lower is better), on gaps so that the classifier's own real-data accuracy cancels:

    paper_dist = | (acc_real - acc_syn) - 0.002 |  +  | (f1w_real - f1w_syn) - 0.008 |

(paper: real 0.7760, synthetic 0.7740, F1 gap ~0.008). One study per arm, same search space,
sampler, pruner and **40 trials** as step 3.

## Step 5: confirmation on the test split

Top 2 completed trials of each arm from step 3 by `score`, plus the closest trial of each arm from
step 4 (6 configs). Each: real CTAB-GAN+ split, 150 epochs, `--eval full` (plus `paper_guess`
recorded), seeds 0-4. Summary with `scripts/summarize.py`, paired quantum-vs-gaussian by seed index.
Claims only for what holds across seeds, with spread.

## Changes after this point

- **2026-11-11, before any screening result existed: implementation details of step 3/4, no rule
  changed.** (1) Parallel workers each get their own TPE seed (0-5); identical seeds would draw
  identical startup configs. (2) Parallel workers can finish a few trials beyond 40; selection uses
  the first 40 finished (COMPLETE or PRUNED) trials by trial number, identically for both arms.
  (3) A trial whose objective is NaN/inf is marked FAIL, does not count towards the 40, and failures
  are reported per arm. (4) Quantum-only settings that have no effect for a given trial (e.g.
  `noise_scale` when `noise_encoding=none`) are still sampled; this costs TPE some efficiency equally
  in every quantum trial. (5) `scripts/screen.sh` passes each command to xargs as one argument
  (macOS `xargs -I` truncates at 255 bytes, which broke the first launch before any run started).
