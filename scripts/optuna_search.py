"""Steps 3 and 4 of runs/optuna/PLAN.md: one Optuna study per arm, same budget and pruner for both.

    objective=score   CTAB-GAN+ metrics on the validation split, fixed-scale average (step 3)
    objective=paper   distance to the paper's Adult numbers under evaluate.paper_guess_utility (step 4)

Resumable: studies live in SQLite (runs/optuna/studies.db); rerunning the same command continues the
study until it has --trials COMPLETE+PRUNED trials. Trials left RUNNING by a crash are marked FAIL (via
heartbeat) and don't count. Every trial writes runs/optuna/<study>/trial_NNN/{config,checks,history}.json.

Usage
  python scripts/optuna_search.py run --arm quantum --objective score --space runs/optuna/space.json --workers 6
  python scripts/optuna_search.py report --objective score      # tables + pruned counts per arm
"""

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402

from qtabgan.config import Config  # noqa: E402
from qtabgan.data import Preprocessor, ctabgan_split, load  # noqa: E402
from qtabgan.evaluate import ml_utility, paper_guess_utility, stat_sim  # noqa: E402
from qtabgan.model import QTabGAN  # noqa: E402

# Environment overrides exist only for smoke tests; the plan's values are the defaults.
ROOT = os.environ.get('OPTUNA_ROOT', 'runs/optuna')
DB = f'sqlite:///{ROOT}/studies.db'
EPOCHS = int(os.environ.get('OPTUNA_EPOCHS', 60))
MIN_RESOURCE = int(os.environ.get('OPTUNA_MIN_RESOURCE', 20))
REDUCTION = 3
SCALES = {'f1': 0.2365, 'avg_jsd': 0.3299, 'diff_corr': 2.906}  # PLAN.md: medians over runs/search

BASE = {  # runs/search_final/commands.txt winners (same as scripts/screen.py)
    'quantum': dict(mapper_dims='512,512', disc_dims='256,256', lr_classical=5e-4, loss='bce', batch_size=256,
                    cat_output='gumbel', n_layers=4, n_shots=300, lr_quantum=1e-3, noise_encoding='angle',
                    noise_scale=1.0, angle_init='small_normal'),
    'gaussian': dict(mapper_dims='256,256', disc_dims='512,512', lr_classical=5e-4, loss='wgan_gp', batch_size=256,
                     cat_output='gumbel'),
}
QUANTUM_ONLY = {'n_layers', 'n_shots', 'lr_quantum', 'noise_encoding', 'noise_scale', 'angle_init', 'prob_scale'}
ALWAYS_QUANTUM = {'noise_encoding', 'angle_init', 'noise_scale'}
DIMS = ['128,128', '256,256', '512,512', '1024,1024']


def suggest(trial, name):
    """PLAN.md step 3 ranges."""
    if name in ('lr_classical', 'lr_quantum'):
        lo, hi = (5e-5, 2e-3) if name == 'lr_classical' else (1e-4, 1e-1)
        return trial.suggest_float(name, lo, hi, log=True)
    if name == 'noise_scale':
        return trial.suggest_float(name, 0.25, 2.0)
    if name == 'n_layers':
        return trial.suggest_int(name, 1, 8)
    choices = {
        'n_qubits': [4, 6, 8], 'n_shots': [100, 300, 1024, 0], 'noise_encoding': ['none', 'angle'],
        'angle_init': ['uniform_0_2pi', 'small_normal'], 'prob_scale': ['n_outcomes', 'none'],
        'mapper_dims': DIMS, 'disc_dims': DIMS, 'loss': ['bce', 'wgan_gp'], 'batch_size': [128, 256, 500],
        'cat_output': ['gumbel', 'softmax', 'linear'], 'betas': ['0.5,0.9', '0.5,0.999'], 'd_steps': [1, 2, 5],
        'input_noise_var': [0.0, 1e-5, 1e-3],
    }[name]
    return trial.suggest_categorical(name, choices)


def searched(arm, space):
    names = [s for s in space if arm == 'quantum' or s not in QUANTUM_ONLY]
    if arm == 'quantum':
        names += [s for s in sorted(ALWAYS_QUANTUM) if s not in names]
    return names


def make_config(arm, params, seed=0, epochs=EPOCHS):
    kw = dict(BASE[arm], **params)
    for k in ('mapper_dims', 'disc_dims', 'betas'):
        if isinstance(kw.get(k), str):
            kw[k] = tuple(float(v) if '.' in v else int(v) for v in kw[k].split(','))
    if kw.get('n_shots') == 0:
        kw['n_shots'] = None
    return Config(source=arm, seed=seed, epochs=epochs, **kw)


def evaluate(objective, df, fake, cat_cols):
    if objective == 'score':
        f1 = ml_utility(df, [fake], classifiers=('lr', 'dt', 'rf')).mean()['f1']
        ss = stat_sim(df, fake, cat_cols)
        m = {'f1': float(f1), 'avg_jsd': ss['avg_jsd'], 'diff_corr': ss['diff_corr']}
        m['value'] = float(np.mean([m[k] / SCALES[k] for k in SCALES]))
    else:
        m = paper_guess_utility(df, fake, 'knn')
        m['lr'] = paper_guess_utility(df, fake, 'lr')
        m['value'] = m['paper_dist']
    return m


def base_value(arm, name):
    v = BASE[arm][name] if name in BASE[arm] else getattr(Config(), name)
    return ','.join(str(x) for x in v) if isinstance(v, tuple) else v


def budget_trials(study, n):
    """The first n finished (COMPLETE or PRUNED) trials by number. Parallel workers can overshoot the
    budget by a few trials; those extra trials are excluded from selection, identically for both arms."""
    fin = [t for t in sorted(study.trials, key=lambda t: t.number)
           if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    return fin[:n]


def study_name(objective, arm):
    return f'{objective}_{arm}'


def storage():
    return optuna.storages.RDBStorage(DB, heartbeat_interval=60, grace_period=300,
                                      engine_kwargs={'connect_args': {'timeout': 60}})


def worker(arm, objective, names, n_trials, wid):
    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df, spec = load('adult')
    df, _ = ctabgan_split(df, spec['target'])  # validation setting: the real test split is never loaded
    train_df, _ = ctabgan_split(df, spec['target'])
    sampler = optuna.samplers.TPESampler(seed=wid, n_startup_trials=10, constant_liar=True)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=MIN_RESOURCE, reduction_factor=REDUCTION,
                                                    min_early_stopping_rate=0)
    study = optuna.load_study(study_name=study_name(objective, arm), storage=storage(), sampler=sampler,
                              pruner=pruner)

    def run_trial(trial):
        params = {n: suggest(trial, n) for n in names}
        cfg = make_config(arm, params)
        out = f'{ROOT}/{study.study_name}/trial_{trial.number:03d}'
        os.makedirs(out, exist_ok=True)
        json.dump({'params': params, 'config': dataclasses.asdict(cfg)}, open(f'{out}/config.json', 'w'),
                  indent=1, default=str)
        prep = Preprocessor(spec['categorical'], spec['integer'], spec['target'], clip=cfg.clip_percentiles,
                            noise_var=cfg.input_noise_var).fit(train_df)
        X, y = prep.transform(train_df, add_noise=True, rng=np.random.default_rng(cfg.seed))
        model = QTabGAN(cfg, prep)
        checks = []

        def check(m, epoch):
            if epoch not in (MIN_RESOURCE, EPOCHS):
                return
            r = evaluate(objective, df, m.sample(len(df)), spec['categorical'])
            r['epoch'] = epoch
            checks.append(r)
            json.dump(checks, open(f'{out}/checks.json', 'w'), indent=1)
            if not math.isfinite(r['value']):
                raise ValueError(f'non-finite objective at epoch {epoch}')  # -> FAIL, reported
            if epoch < EPOCHS:
                trial.report(r['value'], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        try:
            model.fit(X, y, log_every=0, callback=check)
        finally:
            json.dump(model.history, open(f'{out}/history.json', 'w'))
        return checks[-1]['value']

    cb = optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,
                                                          optuna.trial.TrialState.PRUNED,
                                                          optuna.trial.TrialState.RUNNING))
    study.optimize(run_trial, callbacks=[cb], catch=(RuntimeError, ValueError))


def run(a):
    os.makedirs(ROOT, exist_ok=True)
    space = json.load(open(a.space))['selected']
    names = searched(a.arm, space)
    st = storage()
    study = optuna.create_study(study_name=study_name(a.objective, a.arm), storage=st, direction='minimize',
                                load_if_exists=True)
    # stale RUNNING trials from a crash -> FAIL (they don't count towards the budget)
    optuna.storages.fail_stale_trials(study)
    if len(study.trials) == 0:
        study.enqueue_trial({n: base_value(a.arm, n) for n in names})
        study.set_user_attr('searched', names)
        study.set_user_attr('pruner', f'SuccessiveHalving(min_resource={MIN_RESOURCE}, reduction_factor={REDUCTION})')
    print(f'{study.study_name}: searching {names}', flush=True)
    procs = [mp.get_context('spawn').Process(target=worker, args=(a.arm, a.objective, names, a.trials, w))
             for w in range(a.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    report_one(optuna.load_study(study_name=study.study_name, storage=st), n=a.trials)


def report_one(study, top=5, n=40):
    from collections import Counter
    allc = Counter(t.state.name for t in study.trials)
    budget = budget_trials(study, n)
    c = Counter(t.state.name for t in budget)
    done = [t for t in budget if t.state == optuna.trial.TrialState.COMPLETE]
    print(f'\n### {study.study_name}: all trials ' + '  '.join(f'{k} {v}' for k, v in sorted(allc.items())))
    print(f'budget (first {n} finished): COMPLETE {c.get("COMPLETE", 0)}, PRUNED at epoch {MIN_RESOURCE} '
          f'{c.get("PRUNED", 0)} ({c.get("PRUNED", 0) / max(len(budget), 1):.0%})')
    print('\n| trial | value | ' + ' | '.join(study.user_attrs.get('searched', [])) + ' |')
    print('|---|---|' + '---|' * len(study.user_attrs.get('searched', [])))
    for t in sorted(done, key=lambda t: t.value)[:top]:
        print(f'| {t.number} | {t.value:.4f} | ' + ' | '.join(str(t.params.get(k)) for k in study.user_attrs.get('searched', [])) + ' |')


def report(a):
    for arm in ('quantum', 'gaussian'):
        try:
            report_one(optuna.load_study(study_name=study_name(a.objective, arm), storage=storage()), a.top, a.trials)
        except KeyError:
            print(f'(no study {study_name(a.objective, arm)})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('--arm', choices=['quantum', 'gaussian'])
    ap.add_argument('--objective', choices=['score', 'paper'], default='score')
    ap.add_argument('--space', default=f'{ROOT}/space.json')
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--top', type=int, default=5)
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else report(a)
