"""Training-length experiment ("does training longer help?") and the shared "best config" rule.

Rule (fixed 2026-11-11, before any step-5 result existed): per arm, the "best" config is the step-5
confirmation config (runs/confirm/<arm>_<objective><rank>_s<seed>) with the lowest MEAN plan score over
its seeds, where score = (f1 / 0.2365 + avg_jsd / 0.3299 + diff_corr / 2.906) / 3 is the same fixed-scale
objective as runs/optuna/PLAN.md, computed from each run's full evaluation on the test split. This is a
post-hoc pick for a descriptive experiment: nothing in it feeds back into any claim of steps 3-5.

The experiment retrains each arm's best config on the VALIDATION split (--tune), seeds 0-2, 300 epochs,
evaluating every 25 epochs (scripts/run.py --eval-every 25).

Usage: python scripts/training_length.py plan > runs/training_length/commands.txt
"""

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

SCALES = {'f1': 0.2365, 'avg_jsd': 0.3299, 'diff_corr': 2.906}  # = optuna_search.SCALES
SEEDS, EPOCHS, EVERY = range(3), 300, 25


def score(r):
    m = {'f1': r['ml_utility_mean_diff']['f1'], 'avg_jsd': r['stat_sim']['avg_jsd'],
         'diff_corr': r['stat_sim']['diff_corr']}
    return float(np.mean([m[k] / SCALES[k] for k in SCALES]))


def confirm_runs(root='runs/confirm'):
    """{config name: [result.json dicts]} for every finished step-5 run."""
    out = {}
    for p in sorted(glob.glob(f'{root}/*_s[0-9]*/result.json')):
        name = re.sub(r'_s\d+$', '', os.path.basename(os.path.dirname(p)))
        out.setdefault(name, []).append(json.load(open(p)))
    return out


def best_configs(root='runs/confirm'):
    """{arm: (config name, mean score)} by the rule in the module docstring."""
    runs = confirm_runs(root)
    best = {}
    for arm in ('quantum', 'gaussian'):
        cands = {n: np.mean([score(r) for r in rs]) for n, rs in runs.items() if n.startswith(arm + '_')}
        if cands:
            n = min(cands, key=cands.get)
            best[arm] = (n, float(cands[n]))
    return best


if __name__ == '__main__':
    from confirm import flags  # noqa: E402  (imports optuna_search -> torch; only needed here)
    sel = {s['name']: s for s in json.load(open('runs/confirm/selected.json'))}
    best = best_configs()
    os.makedirs('runs/training_length', exist_ok=True)
    json.dump({arm: {'config': n, 'mean_test_score': v, **sel[n]} for arm, (n, v) in best.items()},
              open('runs/training_length/selected.json', 'w'), indent=1)
    for s in SEEDS:
        for arm, (n, _) in best.items():
            print(f'python3 scripts/run.py --tune --epochs {EPOCHS} --eval quick --eval-every {EVERY} '
                  f'{flags(arm, sel[n]["params"])} --seed {s} --out runs/training_length/{arm}_s{s}')
