"""Aggregate result.json files into markdown tables.

Usage
    python scripts/summarize.py tune runs/tune            # one row per tuning run
    python scripts/summarize.py final runs/final runs/reference/result.json
"""

import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROW_KEYS = [('acc', 'ml_utility_mean_diff', 'acc', 2), ('f1', 'ml_utility_mean_diff', 'f1', 3),
            ('auc', 'ml_utility_mean_diff', 'auc', 3), ('avg_jsd', 'stat_sim', 'avg_jsd', 3),
            ('avg_wd', 'stat_sim', 'avg_wd', 3), ('diff_corr', 'stat_sim', 'diff_corr', 2),
            ('js_div (paper def)', 'paper_metrics', 'avg_js_divergence', 4),
            ('avg|dPearson| (paper def)', 'paper_metrics', 'avg_abs_pearson_diff', 3)]


def _get(r, sec, key):
    return r[sec][key]


def tune(root):
    rows = []
    for p in sorted(glob.glob(f'{root}/*/result.json')):
        r = json.load(open(p))
        rows.append((os.path.basename(os.path.dirname(p)), r))
    # Selection rule (fixed before stage 2): average rank over the three metric families
    # (utility = F1 gap, marginals = avg JSD, joint structure = Diff. Corr). Lower is better.
    crit = [('ml_utility_mean_diff', 'f1'), ('stat_sim', 'avg_jsd'), ('stat_sim', 'diff_corr')]
    ranks = np.zeros(len(rows))
    for sec, key in crit:
        ranks += stats.rankdata([r[sec][key] for _, r in rows])
    ranks /= len(crit)
    order = np.argsort(ranks)
    print('| run | avg rank | ' + ' | '.join(k for k, *_ in ROW_KEYS) + ' | train s |')
    print('|---' * (len(ROW_KEYS) + 3) + '|')
    for i in order:
        name, r = rows[i]
        vals = [f'{_get(r, s, k):.{d}f}' for _, s, k, d in ROW_KEYS]
        print(f'| {name} | {ranks[i]:.2f} | ' + ' | '.join(vals) + f" | {r['train_seconds']:.0f} |")


def final(root, ref_path=None):
    arms = {}
    for p in sorted(glob.glob(f'{root}/*/result.json')):
        name = os.path.basename(os.path.dirname(p))
        arm = re.sub(r'_s\d+$', '', name)
        arms.setdefault(arm, []).append(json.load(open(p)))
    print('Mean +- sd over seeds (n per arm in brackets). ML-utility rows are real - synthetic, '
          'averaged over lr/dt/rf/mlp/svm; lower is better everywhere.\n')
    print('| metric | ' + ' | '.join(f'{a} [{len(v)}]' for a, v in arms.items()) +
          (' | floor: marginals | ceiling: real copy |' if ref_path else ' |'))
    print('|---' * (len(arms) + 1 + (2 if ref_path else 0)) + '|')
    ref = json.load(open(ref_path)) if ref_path else None
    for label, sec, key, d in ROW_KEYS:
        cells = []
        for runs in arms.values():
            x = np.array([_get(r, sec, key) for r in runs])
            cells.append(f'{x.mean():.{d}f} +- {x.std(ddof=1) if len(x) > 1 else 0:.{d}f}')
        if ref:
            cells += [f"{ref[k][sec][key]:.{d}f}" for k in ('marginals', 'real_copy')]
        print(f'| {label} | ' + ' | '.join(cells) + ' |')
    # Paired comparisons: each trained-quantum arm vs each control (same seed index), Holm-corrected
    qarms = [a for a in arms if a in ('quantum', 'quantum_angle')]
    controls = [a for a in arms if a not in qarms]
    tests = []
    for qa in qarms:
        for arm in controls:
            if len(arms[arm]) != len(arms[qa]) or len(arms[qa]) < 2:
                continue
            for label, sec, key in [('f1', 'ml_utility_mean_diff', 'f1'), ('diff_corr', 'stat_sim', 'diff_corr')]:
                q = np.array([_get(r, sec, key) for r in arms[qa]])
                c = np.array([_get(r, sec, key) for r in arms[arm]])
                t, pv = stats.ttest_rel(q, c)
                tests.append((f'{qa} - {arm}', label, q.mean() - c.mean(), pv))
    if tests:
        n = min(len(v) for v in arms.values())
        print(f'\nPaired t-tests (same seed index), Holm-corrected over all comparisons below. '
              f'Negative delta = quantum arm better. n={n} seeds per arm: a pilot with very low power'
              f'{" (df=1, p-values are close to meaningless)" if n == 2 else ""}.')
        m = len(tests)
        order = np.argsort([t[3] for t in tests])
        adj, running = [0.0] * m, 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * tests[i][3]))
            adj[i] = running
        for (arm, label, delta, pv), pa in zip(tests, adj):
            print(f'  {arm:32s} {label:10s} delta {delta:+.4f}   p={pv:.3f}  p_holm={pa:.3f}')


if __name__ == '__main__':
    if sys.argv[1] == 'tune':
        tune(sys.argv[2])
    else:
        final(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
