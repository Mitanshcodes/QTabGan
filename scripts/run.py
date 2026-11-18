"""Train QTabGAN (or a control) on one dataset and evaluate it.

Examples
    # final run, evaluated on the CTAB-GAN+ test split
    python scripts/run.py --source quantum --seed 0 --out runs/quantum_s0

    # tuning run: generator trains on 80% of the CTAB-GAN+ *train* split and is evaluated on the
    # remaining 20% of it; the real test set is never touched
    python scripts/run.py --tune --epochs 30 --eval quick --out runs/tune/x

Any Config field can be overridden: --n-layers 2 --cat-output linear --loss wgan_gp ...
"""

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qtabgan.config import Config  # noqa: E402
from qtabgan.data import Preprocessor, ctabgan_split, load  # noqa: E402
from qtabgan.diagnostics import classifier_metrics, diversity_metrics  # noqa: E402
from qtabgan.evaluate import ml_utility, paper_guess_utility, qtabgan_paper_metrics, stat_sim  # noqa: E402
from qtabgan.model import QTabGAN  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='adult')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tune', action='store_true', help='evaluate on a validation split of train')
    ap.add_argument('--eval', choices=['full', 'quick', 'none'], default='full')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--paper-guess', action='store_true',
                    help='also record evaluate.paper_guess_utility (kNN and LR, unscaled; a guess at the paper\'s eval)')
    ap.add_argument('--eval-every', type=int, default=0,
                    help='every N epochs: sample a synthetic table and add F1 gap (lr+dt+rf), avg JSD and '
                         'Diff. Corr against the evaluation table to that epoch\'s history.json record')
    for f in dataclasses.fields(Config):
        flag = '--' + f.name.replace('_', '-')
        if isinstance(f.default, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ('1', 'true', 'yes'), default=f.default)
        elif isinstance(f.default, tuple):
            ap.add_argument(flag, type=lambda s: tuple(float(v) if '.' in v else int(v) for v in s.split(',')),
                            default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    return ap.parse_args()


def make_eval_callback(every, df, cat_cols, holdout=None, spec=None):
    """Training-length curve. Uses the existing evaluate.py functions unchanged. Sampling draws from the
    global torch RNG, so a run with --eval-every is not bit-identical to the same seed without it.
    With `holdout` (real rows the generator never trains on) it also records qtabgan.diagnostics
    metrics: a 100-tree random forest trained on the synthetic table and scored on `holdout`, plus
    diversity checks of the synthetic table against `df`."""
    if not every:
        return None

    def cb(model, epoch):
        if epoch % every:
            return
        t = time.time()
        fake = model.sample(len(df))
        ss = stat_sim(df, fake, cat_cols)
        model.history[-1].update({
            'eval_f1': float(ml_utility(df, [fake], classifiers=('lr', 'dt', 'rf')).mean()['f1']),
            'eval_avg_jsd': ss['avg_jsd'], 'eval_diff_corr': ss['diff_corr']})
        if holdout is not None:
            tgt = spec['target']
            model.history[-1].update({f'eval_{k}': v for k, v in
                                      classifier_metrics(fake, holdout, tgt).items()})
            model.history[-1].update({f'eval_{k}': v for k, v in diversity_metrics(
                df, fake, spec['integer'], spec['categorical'], tgt).items()})
        model.history[-1]['eval_seconds'] = time.time() - t
        with open(f'{model.out}/history.json', 'w') as fh:  # partial curve survives a crash
            json.dump(model.history, fh)
        print(f"eval @ {epoch}: f1 {model.history[-1]['eval_f1']:.4f}  jsd {ss['avg_jsd']:.4f}  "
              f"diff_corr {ss['diff_corr']:.3f}", flush=True)
    return cb


def main():
    a = parse()
    torch.set_num_threads(a.threads)
    cfg = Config(**{f.name: getattr(a, f.name) for f in dataclasses.fields(Config)})
    if cfg.n_shots == 0:
        cfg.n_shots = None
    os.makedirs(a.out, exist_ok=True)

    df, spec = load(a.dataset)
    train_df, test_df = ctabgan_split(df, spec['target'])
    if a.tune:  # nested split: "real" for evaluation = the train portion only
        df = train_df
        train_df, test_df = ctabgan_split(df, spec['target'])
    print(f'real rows for eval: {len(df)}  generator train rows: {len(train_df)}  held out: {len(test_df)}')

    prep = Preprocessor(spec['categorical'], spec['integer'], spec['target'],
                        clip=cfg.clip_percentiles, noise_var=cfg.input_noise_var).fit(train_df)
    X, y = prep.transform(train_df, add_noise=True, rng=np.random.default_rng(cfg.seed))

    model = QTabGAN(cfg, prep)
    model.out = a.out
    params = model.n_params()
    print('params:', params, flush=True)
    t0 = time.time()
    model.fit(X, y, callback=make_eval_callback(a.eval_every, df, spec['categorical'],
                                                holdout=test_df, spec=spec))
    train_secs = time.time() - t0

    fake = model.sample(len(df))
    fake.to_csv(f'{a.out}/fake.csv', index=False)
    result = {'config': dataclasses.asdict(cfg), 'params': params, 'train_seconds': train_secs,
              'tune': a.tune, 'n_real_eval_rows': len(df), 'n_fake_rows': len(fake)}

    if a.eval != 'none':
        clfs = ('lr', 'dt', 'rf') if a.eval == 'quick' else ('lr', 'dt', 'rf', 'mlp', 'svm')
        diff, real_s, fake_s = ml_utility(df, [fake], classifiers=clfs, return_raw=True)
        result['ml_utility_diff'] = diff.to_dict()
        result['ml_utility_real'] = real_s.to_dict()
        result['ml_utility_fake'] = fake_s.to_dict()
        result['ml_utility_mean_diff'] = diff.mean().to_dict()
        result['stat_sim'] = stat_sim(df, fake, spec['categorical'])
        result['paper_metrics'] = qtabgan_paper_metrics(df, fake, spec['categorical'])
        if a.paper_guess:
            result['paper_guess'] = {c: paper_guess_utility(df, fake, c) for c in ('knn', 'lr')}
            print('paper_guess:', {c: round(v['paper_dist'], 4) for c, v in result['paper_guess'].items()})
        print('ML utility diff (real - synthetic):\n', diff.round(4).to_string())
        print('mean diff:', {k: round(v, 4) for k, v in result['ml_utility_mean_diff'].items()})
        print('stat_sim (CTAB-GAN+ defs):', {k: round(v, 4) for k, v in result['stat_sim'].items()})
        print('paper-as-written defs:', {k: round(v, 4) for k, v in result['paper_metrics'].items()})

    with open(f'{a.out}/result.json', 'w') as fh:
        json.dump(result, fh, indent=1, default=str)
    with open(f'{a.out}/history.json', 'w') as fh:
        json.dump(model.history, fh)
    print(f'done in {time.time() - t0:.0f}s -> {a.out}')


if __name__ == '__main__':
    main()
