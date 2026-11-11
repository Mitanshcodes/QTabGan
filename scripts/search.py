"""Bounded random search over the settings the paper leaves open, for QTabGAN AND the Gaussian
control, with the same budget for both. Turns "we guessed the unknowns" into "we searched them".

Design
  - N configs per arm (default 20), sampled once with a fixed seed.
  - Config i uses the SAME shared settings (network sizes, learning rate, loss, batch size) in both
    arms, so the pairs differ only in what feeds the mapper. The quantum arm also samples the
    quantum-only settings (layers, shots, circuit learning rate, noise encoding and range, angle init).
  - Every run: validation split of the CTAB-GAN+ train set (test set untouched), 60 epochs, seed 0,
    quick evaluation (lr/dt/rf + similarity metrics).
  - Selection inside each arm: average rank over F1 gap, avg JSD and Diff. Corr (as in tuning).

Usage
  python scripts/search.py plan  [--n 20] > runs/search/commands.txt   # print the commands
  bash   scripts/search.sh                                            # run them, 2 at a time, resumable
  python scripts/search.py summarize                                   # tables + best config per arm
"""

import argparse
import glob
import json
import os
import random

import numpy as np
from scipy import stats

SHARED = {
    'mapper-dims': ['256,256', '512,512'],
    'disc-dims': ['256,256', '512,512'],
    'lr-classical': [1e-4, 2e-4, 5e-4],
    'loss': ['wgan_gp', 'bce'],
    'batch-size': [256, 500],
}
QUANTUM = {
    'n-layers': [2, 4, 6],
    'n-shots': [300, 1024],
    'lr-quantum': [1e-3, 1e-2, 5e-2],
    'noise-encoding': ['none', 'angle'],
    'noise-scale': [0.5, 1.0],
    'angle-init': ['uniform_0_2pi', 'small_normal'],
}
FIXED = '--tune --eval quick --epochs 60 --seed 0 --cat-output gumbel'


def plan(n, seed=2026):
    rng = random.Random(seed)
    cmds = []
    for i in range(n):
        shared = {k: rng.choice(v) for k, v in SHARED.items()}
        quantum = {k: rng.choice(v) for k, v in QUANTUM.items()}
        s = ' '.join(f'--{k} {v}' for k, v in shared.items())
        q = ' '.join(f'--{k} {v}' for k, v in quantum.items())
        cmds.append(f'python3 scripts/run.py {FIXED} {s} {q} --source quantum --out runs/search/quantum_{i:02d}')
        cmds.append(f'python3 scripts/run.py {FIXED} {s} --source gaussian --out runs/search/gaussian_{i:02d}')
    return cmds


def summarize(root='runs/search'):
    keys = [('f1', 'ml_utility_mean_diff', 'f1'), ('avg_jsd', 'stat_sim', 'avg_jsd'),
            ('diff_corr', 'stat_sim', 'diff_corr')]
    best = {}
    for arm in ('quantum', 'gaussian'):
        runs = []
        for p in sorted(glob.glob(f'{root}/{arm}_*/result.json')):
            r = json.load(open(p))
            runs.append((os.path.basename(os.path.dirname(p)), r))
        if not runs:
            continue
        ranks = np.mean([stats.rankdata([r[s][k] for _, r in runs]) for _, s, k in keys], axis=0)
        order = np.argsort(ranks)
        print(f'\n### {arm} ({len(runs)} runs, best first)\n')
        qcols = ['n_layers', 'n_shots', 'lr_quantum', 'noise_encoding', 'noise_scale', 'angle_init'] if arm == 'quantum' else []
        cols = ['mapper_dims', 'disc_dims', 'lr_classical', 'loss', 'batch_size'] + qcols
        print('| run | avg rank | f1 | avg_jsd | diff_corr | min | ' + ' | '.join(cols) + ' |')
        print('|---' * (6 + len(cols)) + '|')
        for i in order:
            name, r = runs[i]
            c = r['config']
            print(f"| {name} | {ranks[i]:.1f} | {r['ml_utility_mean_diff']['f1']:.3f} | "
                  f"{r['stat_sim']['avg_jsd']:.3f} | {r['stat_sim']['diff_corr']:.2f} | "
                  f"{r['train_seconds'] / 60:.0f} | " + ' | '.join(str(c[k]) for k in cols) + ' |')
        best[arm] = [(runs[i][0], runs[i][1]) for i in order[:3]]
        # best on each single metric, so no metric choice hides a quantum win
        for label, s, k in keys:
            j = int(np.argmin([r[s][k] for _, r in runs]))
            print(f'best {label}: {runs[j][0]} = {runs[j][1][s][k]:.3f}')
    # paired comparison: same shared settings, quantum vs gaussian
    pairs = []
    for p in sorted(glob.glob(f'{root}/quantum_*/result.json')):
        g = p.replace('quantum_', 'gaussian_')
        if os.path.exists(g):
            pairs.append((json.load(open(p)), json.load(open(g))))
    if pairs:
        print(f'\n### paired: same shared settings ({len(pairs)} pairs), quantum minus gaussian (negative = quantum better)\n')
        for label, s, k in keys:
            d = np.array([q[s][k] - g[s][k] for q, g in pairs])
            wins = int((d < 0).sum())
            t = stats.wilcoxon(d) if len(d) >= 6 and np.any(d != 0) else None
            print(f'{label:10s} median diff {np.median(d):+.3f}  quantum better in {wins}/{len(d)} pairs'
                  + (f'  (Wilcoxon p={t.pvalue:.3f})' if t else ''))
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['plan', 'summarize'])
    ap.add_argument('--n', type=int, default=20)
    a = ap.parse_args()
    if a.cmd == 'plan':
        print('\n'.join(plan(a.n)))
    else:
        summarize()
