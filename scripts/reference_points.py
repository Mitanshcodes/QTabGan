"""Two no-training reference points that calibrate what the metrics mean on this dataset.

floor   'marginals': every column (incl. the label) sampled independently from its real-train
        marginal. All relationships between columns are destroyed, so any generator worth
        publishing must beat this.
ceiling 'real_copy': the generator's own training rows used as the "synthetic" table. Best
        possible fidelity (and zero privacy); the gap to this is the room for improvement.

Usage: python scripts/reference_points.py --out runs/reference
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402

from qtabgan.data import ctabgan_split, load  # noqa: E402
from qtabgan.evaluate import ml_utility, qtabgan_paper_metrics, stat_sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='adult')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, spec = load(a.dataset)
    train_df, _ = ctabgan_split(df, spec['target'])
    rng = np.random.default_rng(0)
    fakes = {
        'marginals': train_df.apply(lambda c: rng.choice(c.to_numpy(), size=len(df), replace=True)),
        'real_copy': train_df,
    }
    out = {}
    for name, fake in fakes.items():
        diff, real_s, fake_s = ml_utility(df, [fake], return_raw=True)
        out[name] = {
            'ml_utility_diff': diff.to_dict(), 'ml_utility_mean_diff': diff.mean().to_dict(),
            'ml_utility_real': real_s.to_dict(), 'ml_utility_fake': fake_s.to_dict(),
            'stat_sim': stat_sim(df, fake, spec['categorical']),
            'paper_metrics': qtabgan_paper_metrics(df, fake, spec['categorical']),
        }
        print(name, {k: round(v, 4) for k, v in out[name]['ml_utility_mean_diff'].items()},
              {k: round(v, 4) for k, v in out[name]['stat_sim'].items()},
              {k: round(v, 4) for k, v in out[name]['paper_metrics'].items()}, flush=True)
    print('real-data scores per classifier:\n', real_s.round(4).to_string())
    with open(f'{a.out}/result.json', 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
