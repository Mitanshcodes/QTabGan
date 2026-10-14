# qtabgan-repro

An independent re-implementation of **QTabGAN** (Kumari, Achutha, Sivaraman, arXiv:2602.12704), a
hybrid quantum-classical GAN for tabular data. It's evaluated with the exact public protocol behind
the baseline numbers the paper compares against (CTAB-GAN+), and compared with controls.

No official code exists. **Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) first**: it lists what the
paper specifies, what it doesn't, what we chose, and the problems we found in the paper.

## What's here

```
qtabgan/config.py     every setting, tagged paper / ambiguous / unstated (python -m qtabgan.config)
qtabgan/data.py       Adult loading, the CTAB-GAN+ split, QTabGAN preprocessing (clip, z-score, [0, pi], one-hot)
qtabgan/sim.py        batched statevector simulator for the QTabGAN circuit (tested equal to PennyLane)
qtabgan/model.py      quantum source (VQC + finite shots), classical mapper, discriminator, training loop
qtabgan/evaluate.py   CTAB-GAN+ evaluation, step for step; plus the metrics as QTabGAN's paper writes them
scripts/run.py        train + sample + evaluate one configuration (any setting overridable by flag)
scripts/tune.sh       the two tuning stages (validation split only)
scripts/final.sh      the final experiment: 5 arms x 3 seeds
scripts/run_ctgan.py  vanilla CTGAN on the same split and evaluation
scripts/reference_points.py  no-training floor and ceiling for every metric
scripts/summarize.py  tables + paired tests
tests/test_core.py    simulator vs PennyLane (values and gradients), shots, preprocessing, split, training
runs/                 logs, configs and metrics of every run reported (synthetic CSVs omitted: regenerate)
AUTHOR_EMAIL.md       draft questions for the authors
```

## Quick start

```bash
pip install -r requirements.txt
bash scripts/get_data.sh              # fetches CTAB-GAN+'s Adult.csv (48,842 x 14)
python -m pytest -q tests             # ~5 s
python scripts/run.py --epochs 5 --eval quick --tune --out runs/try   # ~1 min on a laptop CPU
```

Reproduce the reported experiment (about 2.5 h on 2 CPU cores):

```bash
bash scripts/tune.sh stage1
bash scripts/tune.sh stage2 --cat-output gumbel --loss wgan_gp
python scripts/reference_points.py --out runs/reference
bash scripts/final.sh
python scripts/summarize.py final runs/final runs/reference/result.json
```

Switching settings is a flag, e.g. `--source gaussian`, `--noise-encoding angle`,
`--n-layers 2`, `--n-shots 0` (exact probabilities), `--backend pennylane` (reference simulator;
the path to running on hardware), `--disc-dims 64,64` (weaker discriminator).

## Results (Adult, 150 epochs, 2 seeds)

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

## Relation to the earlier CTGAN fork

The earlier patch spliced a quantum noise source into `sdv-dev/CTGAN`. That tests "CTGAN with a
quantum noise source", not QTabGAN: CTGAN uses different preprocessing (Gaussian-mixture
normalisation), loss (WGAN-GP with PacGAN) and conditioning (training-by-sampling over every
discrete column). Keep the fork for the vanilla CTGAN baseline (`scripts/run_ctgan.py` uses it);
use this repo for QTabGAN.
