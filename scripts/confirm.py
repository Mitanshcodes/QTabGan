"""Step 5 of runs/optuna/PLAN.md: pick configs from the Optuna studies and print confirmation commands.

Per arm: top 2 COMPLETE trials of the score study + the closest COMPLETE trial of the paper study,
all within the first 40 finished trials (scripts/optuna_search.budget_trials). Each config is retrained
on the REAL CTAB-GAN+ split, 150 epochs, full evaluation + paper_guess, seeds 0-4.

Usage: python scripts/confirm.py plan > runs/confirm/commands.txt   (run with scripts/optuna_pipeline.sh)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import optuna  # noqa: E402

from optuna_search import BASE, budget_trials, storage, study_name  # noqa: E402

SEEDS, N = range(5), 40


def picks():
    out = []
    for arm in ('quantum', 'gaussian'):
        for objective, k in (('score', 2), ('paper', 1)):
            st = optuna.load_study(study_name=study_name(objective, arm), storage=storage())
            done = [t for t in budget_trials(st, N) if t.state == optuna.trial.TrialState.COMPLETE]
            for rank, t in enumerate(sorted(done, key=lambda t: t.value)[:k], 1):
                out.append((f'{arm}_{objective}{rank}', arm, objective, t.number, t.value, t.params))
    return out


def flags(arm, params):
    kw = dict(BASE[arm], **params)
    return ' '.join(f"--{k.replace('_', '-')} {v}" for k, v in kw.items()) + f' --source {arm}'


if __name__ == '__main__':
    sel = picks()
    os.makedirs('runs/confirm', exist_ok=True)
    json.dump([dict(name=n, arm=a, objective=o, trial=t, value=v, params=p) for n, a, o, t, v, p in sel],
              open('runs/confirm/selected.json', 'w'), indent=1)
    for s in SEEDS:
        for name, arm, _, _, _, params in sel:
            print(f'python3 scripts/run.py --epochs 150 --eval full --paper-guess {flags(arm, params)} '
                  f'--seed {s} --out runs/confirm/{name}_s{s}')
