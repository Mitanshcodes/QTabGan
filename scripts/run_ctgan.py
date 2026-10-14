"""Vanilla CTGAN (pip/sdv-dev `ctgan`) trained on the same split and scored with the same evaluation.

Usage: python scripts/run_ctgan.py --seed 0 --epochs 150 --out runs/final/ctgan_s0
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ctgan import CTGAN  # noqa: E402

from qtabgan.data import ctabgan_split, load  # noqa: E402
from qtabgan.evaluate import ml_utility, qtabgan_paper_metrics, stat_sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='adult')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    df, spec = load(a.dataset)
    train_df, _ = ctabgan_split(df, spec['target'])

    model = CTGAN(epochs=a.epochs, enable_gpu=False)
    model.set_random_state(a.seed)
    t0 = time.time()
    model.fit(train_df, spec['categorical'])
    secs = time.time() - t0
    fake = model.sample(len(df))
    for c in spec['integer']:
        fake[c] = np.round(fake[c]).astype(int)
    fake = fake[df.columns]
    fake.to_csv(f'{a.out}/fake.csv', index=False)

    diff, real_s, fake_s = ml_utility(df, [fake], return_raw=True)
    gen_params = sum(p.numel() for p in model._generator.parameters())
    result = {
        'model': 'ctgan', 'seed': a.seed, 'epochs': a.epochs, 'train_seconds': secs,
        'params': {'generator_total': gen_params},
        'ml_utility_diff': diff.to_dict(), 'ml_utility_mean_diff': diff.mean().to_dict(),
        'ml_utility_real': real_s.to_dict(), 'ml_utility_fake': fake_s.to_dict(),
        'stat_sim': stat_sim(df, fake, spec['categorical']),
        'paper_metrics': qtabgan_paper_metrics(df, fake, spec['categorical']),
    }
    print(diff.round(4).to_string())
    print('mean diff', result['ml_utility_mean_diff'], 'stat', result['stat_sim'], flush=True)
    with open(f'{a.out}/result.json', 'w') as fh:
        json.dump(result, fh, indent=1)


if __name__ == '__main__':
    main()
