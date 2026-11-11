"""Step 2 of runs/optuna/PLAN.md: one-at-a-time screening around each arm's best config.

Usage
  python scripts/screen.py plan > runs/screen/commands.txt    # print the commands
  bash   scripts/screen.sh                                     # run them (resumable)
  python scripts/screen.py summarize > runs/screen/summary.md  # ranking table + selection
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = 'runs/screen'
EPOCHS = 40
SEEDS = (0, 1)
# Fixed objective scales: medians over the 40 runs of runs/search (see PLAN.md).
SCALES = {'f1': 0.2365, 'avg_jsd': 0.3299, 'diff_corr': 2.906}

BASE = {
    'quantum': {'mapper-dims': '512,512', 'disc-dims': '256,256', 'lr-classical': '0.0005', 'loss': 'bce',
                'batch-size': '256', 'cat-output': 'gumbel', 'n-layers': '4', 'n-shots': '300',
                'lr-quantum': '0.001', 'noise-encoding': 'angle', 'noise-scale': '1.0',
                'angle-init': 'small_normal', 'source': 'quantum'},
    'gaussian': {'mapper-dims': '256,256', 'disc-dims': '512,512', 'lr-classical': '0.0005',
                 'loss': 'wgan_gp', 'batch-size': '256', 'cat-output': 'gumbel', 'source': 'gaussian'},
}
SHARED = {
    'n-qubits': ['4', '6', '8'],
    'mapper-dims': ['128,128', '256,256', '512,512', '1024,1024'],
    'disc-dims': ['128,128', '256,256', '512,512', '1024,1024'],
    'lr-classical': ['0.0002', '0.0005', '0.001'],
    'loss': ['bce', 'wgan_gp'],
    'batch-size': ['128', '256', '500'],
    'cat-output': ['gumbel', 'softmax', 'linear'],
    'betas': ['0.5,0.9', '0.5,0.999'],
    'd-steps': ['1', '2', '5'],
    'input-noise-var': ['0.0', '1e-05', '0.001'],
}
QUANTUM = {
    'n-layers': ['1', '2', '4', '6', '8'],
    'n-shots': ['100', '300', '1024', '0'],
    'lr-quantum': ['0.0001', '0.001', '0.01'],
    'noise-encoding': ['none', 'angle'],
    'noise-scale': ['0.5', '1.0', '2.0'],
    'angle-init': ['uniform_0_2pi', 'small_normal'],
    'prob-scale': ['n_outcomes', 'none'],
}
DEFAULTS = {'n-qubits': '8', 'betas': '0.5,0.9', 'd-steps': '1', 'input-noise-var': '1e-05',
            'prob-scale': 'n_outcomes'}


def _slug(v):
    return re.sub(r'[^0-9a-zA-Z.]+', '-', v).strip('-')


def variants(arm):
    """[(name, setting or None, level, args dict)] - base first."""
    base = BASE[arm]
    out = [('base', None, None, dict(base))]
    space = dict(SHARED, **(QUANTUM if arm == 'quantum' else {}))
    for k, levels in space.items():
        cur = base.get(k, DEFAULTS.get(k))
        for v in levels:
            if v == cur:
                continue
            args = dict(base)
            args[k] = v
            out.append((f'{k}={_slug(v)}', k, v, args))
    return out


def plan():
    cmds = []
    for seed in SEEDS:
        for arm in ('quantum', 'gaussian'):
            for name, _, _, args in variants(arm):
                a = ' '.join(f'--{k} {v}' for k, v in args.items())
                cmds.append(f'python3 scripts/run.py --tune --eval quick --epochs {EPOCHS} --seed {seed} {a} '
                            f'--out {ROOT}/{arm}/{name}/s{seed}')
    return cmds


def metrics(r):
    m = {'f1': r['ml_utility_mean_diff']['f1'], 'avg_jsd': r['stat_sim']['avg_jsd'],
         'diff_corr': r['stat_sim']['diff_corr']}
    m['score'] = float(np.mean([m[k] / SCALES[k] for k in SCALES]))
    return m


def summarize():
    keys = ['score', 'f1', 'avg_jsd', 'diff_corr']
    selected = set()
    for arm in ('quantum', 'gaussian'):
        res = defaultdict(dict)  # name -> seed -> metrics
        for p in glob.glob(f'{ROOT}/{arm}/*/s*/result.json'):
            name, s = p.split('/')[-3], int(p.split('/')[-2][1:])
            res[name][s] = metrics(json.load(open(p)))
        if 'base' not in res:
            continue
        complete = {n: v for n, v in res.items() if len(v) == len(SEEDS)}
        mean = {n: {k: np.mean([v[s][k] for s in SEEDS]) for k in keys} for n, v in complete.items()}
        diffs = [v[SEEDS[0]]['score'] - v[SEEDS[1]]['score'] for v in complete.values()]
        sigma = float(np.sqrt(np.mean(np.square(diffs))) / np.sqrt(2))
        b = mean['base']
        print(f'\n### {arm}: {len(complete)} configs complete (x{len(SEEDS)} seeds), '
              f'{sum(len(v) for v in res.values())} runs\n')
        print(f"base (2-seed mean): score {b['score']:.3f}  f1 {b['f1']:.3f}  avg_jsd {b['avg_jsd']:.3f}  "
              f"diff_corr {b['diff_corr']:.2f}.  Seed noise sigma(score) = {sigma:.3f} "
              f'(from {len(diffs)} seed pairs); a setting matters if max |d score| > 2 sigma = {2 * sigma:.3f}.\n')
        # per setting: the level with the largest |change in score|; also max |change| per metric
        by_setting = defaultdict(list)
        for name in mean:
            if name != 'base':
                by_setting[name.split('=')[0]].append(name)
        rows = []
        for st, names in by_setting.items():
            eff = {k: max(abs(mean[n][k] - b[k]) for n in names) for k in keys}
            best = min(names, key=lambda n: mean[n]['score'])
            worst = max(names, key=lambda n: abs(mean[n]['score'] - b['score']))
            rows.append((eff['score'], st, eff, best, worst))
        rows.sort(reverse=True)
        print('| setting | max abs d score | max abs d F1 gap | max abs d avg JSD | max abs d Diff.Corr | '
              'level with best score (d score) | matters? |')
        print('|---|---|---|---|---|---|---|')
        for e, st, eff, best, worst in rows:
            m = e > 2 * sigma
            if m:
                selected.add(st)
            d = mean[best]['score'] - b['score']
            print(f"| {st} | {eff['score']:.3f} | {eff['f1']:.3f} | {eff['avg_jsd']:.3f} | {eff['diff_corr']:.2f} | "
                  f"{best.split('=')[1]} ({d:+.3f}) | {'yes' if m else 'no'} |")
        print('\n<details><summary>all configs (2-seed means)</summary>\n')
        print('| config | score | f1 | avg_jsd | diff_corr | score s0 | score s1 |')
        print('|---|---|---|---|---|---|---|')
        for n in sorted(mean, key=lambda n: mean[n]['score']):
            print(f"| {n} | {mean[n]['score']:.3f} | {mean[n]['f1']:.3f} | {mean[n]['avg_jsd']:.3f} | "
                  f"{mean[n]['diff_corr']:.2f} | {complete[n][0]['score']:.3f} | {complete[n][1]['score']:.3f} |")
        print('\n</details>')
        missing = sorted(set(n for n, _, _, _ in variants(arm)) - set(complete))
        if missing:
            print(f'\nincomplete: {missing}')
    print(f"\n**Selected for step 3** (matters in either arm): {', '.join(sorted(selected)) or 'none'}"
          '; plus noise-encoding, angle-init, noise-scale for the quantum arm (always).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['plan', 'summarize'])
    a = ap.parse_args()
    if a.cmd == 'plan':
        print('\n'.join(plan()))
    else:
        summarize()
