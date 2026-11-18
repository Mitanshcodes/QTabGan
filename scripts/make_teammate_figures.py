"""Teammate-poster-style figures for our Adult study -> figures/teammate_style/ (PNG 200 dpi + PDF).

Inputs (nothing is retrained here):
  runs/confirm/<best>_s0-4/fake.csv           best QTabGAN / best Gaussian (step 5, test split)
  runs/teammate_style/{ctgan,paper_literal,first_pass}_s0-4/   same protocol, scripts/teammate_style.sh
  runs/teammate_style/diag_{quantum,gaussian}_s0-2/history.json  validation-split training diagnostics
  runs/optuna/studies.db                       step-3 Optuna studies
Random-forest evaluations are cached in runs/teammate_style/rf_eval.json (delete it to recompute).

Positive class = income >50K (label 1, the minority class, ~24% of rows). Every classifier here is a
random forest with 100 trees, trained on a synthetic table (or on the real training rows for the
reference) and tested on the real CTAB-GAN+ test split (9,769 rows).

Usage: python scripts/make_teammate_figures.py [fig1 fig2 ... poster]
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import training_length as tl  # noqa: E402

OUT = 'figures/teammate_style'
TS = 'runs/teammate_style'
CACHE = f'{TS}/rf_eval.json'
SEEDS = range(5)

# Categorical colours: the three models shown together everywhere passed the dataviz palette
# validator (all-pairs CVD and normal-vision separation). The two extra QTabGAN variants only appear
# in figures where every row is labelled by name.
C = {'quantum': '#7B2CBF', 'gaussian': '#eb6834', 'ctgan': '#1baf7a', 'literal': '#2a78d6',
     'first_pass': '#e87ba4', 'real': '#222222'}
LABEL = {'quantum': 'QTabGAN (best searched)', 'gaussian': 'Gaussian control (best searched)',
         'ctgan': 'CTGAN (classical baseline)', 'literal': 'QTabGAN, literal paper settings',
         'first_pass': 'QTabGAN, first-pass settings', 'real': 'Real data (reference)'}
SHORT = {'quantum': 'QTabGAN', 'gaussian': 'Gaussian control', 'ctgan': 'CTGAN', 'literal': 'Literal paper',
         'first_pass': 'First pass', 'real': 'Real data'}
CORE = ('quantum', 'gaussian', 'ctgan')
ALL = ('literal', 'first_pass', 'quantum', 'gaussian', 'ctgan')
NUMERIC_SHOWN = ('age', 'hours-per-week', 'capital-gain', 'fnlwgt')
UNITS = {'age': 'years', 'hours-per-week': 'hours per week', 'capital-gain': 'US dollars',
         'fnlwgt': 'census sampling weight'}

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.linewidth': 0.6, 'font.family': 'DejaVu Sans', 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


# ---------------------------------------------------------------- helpers
def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=200)
    plt.close(fig)
    print(f'wrote {OUT}/{name}.png/.pdf')


def headline(fig, title, subtitle=None):
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    else:
        fig.canvas.draw()
    h = fig.get_figheight()
    fig.text(0.01, 1 + (0.75 if subtitle else 0.3) / h, title, fontweight='bold', ha='left', va='bottom',
             fontsize=15)
    if subtitle:
        fig.text(0.01, 1 + 0.2 / h, subtitle, ha='left', va='bottom', fontsize=11.5, color='#444444')


def band(ax, x, ys, color, label=None, ls='-', marker=None, lw=2):
    ys = np.asarray(ys, float)
    ax.fill_between(x, np.nanmin(ys, 0), np.nanmax(ys, 0), color=color, alpha=0.18, lw=0)
    ax.plot(x, np.nanmean(ys, 0), color=color, lw=lw, ls=ls, label=label, marker=marker, ms=4)


def run_dirs():
    best = {arm: n for arm, (n, _) in tl.best_configs('runs/confirm').items()}
    return {'quantum': [f'runs/confirm/{best["quantum"]}_s{s}' for s in SEEDS],
            'gaussian': [f'runs/confirm/{best["gaussian"]}_s{s}' for s in SEEDS],
            'ctgan': [f'{TS}/ctgan_s{s}' for s in SEEDS],
            'literal': [f'{TS}/paper_literal_s{s}' for s in SEEDS],
            'first_pass': [f'{TS}/first_pass_s{s}' for s in SEEDS]}


def data():
    from qtabgan.data import ctabgan_split, load
    df, spec = load('adult')
    train, test = ctabgan_split(df, spec['target'])
    return df, spec, train, test


def fmt(v, d=3):
    mu, sd = mean_sd(v)
    return f'{mu:.{d}f} ± {sd:.{d}f}' if len(v) > 1 else f'{mu:.{d}f} (1 seed)'


def mean_sd(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


# ---------------------------------------------------------------- cached evaluation
GRID = np.linspace(0, 1, 201)


def rf_eval():
    """Per model and seed: classifier metrics, ROC/PR curves on a fixed grid, KS per numeric column,
    diversity metrics. Cached; a model whose runs are missing is skipped (and recomputed later)."""
    from scipy.stats import ks_2samp
    from sklearn.metrics import precision_recall_curve, roc_curve

    from qtabgan.diagnostics import classifier_metrics, diversity_metrics, rf_fit_predict
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    df, spec, train, test = data()
    tgt, y = spec['target'], test[spec['target']].to_numpy().astype(int)

    def one(table, seed):
        p = rf_fit_predict(table, test, tgt, seed)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        # interpolated precision: best precision at recall >= r (the usual PR-curve envelope).
        # sklearn returns recall in decreasing order, so "recall >= r" = this index and all earlier ones.
        env = np.maximum.accumulate(prec)
        rec_r, env_r = rec[::-1], env[::-1]
        m = classifier_metrics(table, test, tgt, seed)
        return {'metrics': m, 'roc_tpr': np.interp(GRID, fpr, tpr).tolist(),
                'pr_precision': np.interp(GRID, rec_r, env_r).tolist()}

    changed = False
    if 'real' not in cache:  # reference: same forest trained on the real training rows, 5 forest seeds
        cache['real'] = [dict(one(train, s), seed=s) for s in SEEDS]
        changed = True
    import pandas as pd
    for model, dirs in run_dirs().items():
        have = {r['seed'] for r in cache.get(model, [])}
        todo = [(s, d) for s, d in zip(SEEDS, dirs) if s not in have and os.path.exists(f'{d}/fake.csv')]
        if not todo:
            continue
        rows = cache.get(model, [])
        for s, d in todo:  # partial runs are fine: each finished seed is evaluated once and cached
            fake = pd.read_csv(f'{d}/fake.csv')
            r = one(fake, 0)  # forest seed fixed; the spread comes from the generator seeds
            r['seed'] = s
            r['ks'] = {c: float(ks_2samp(df[c], fake[c]).statistic) for c in spec['integer']}
            r['diversity'] = diversity_metrics(df, fake, spec['integer'], spec['categorical'], tgt)
            rows.append(r)
            print(f'  evaluated {model} seed {s}', flush=True)
        cache[model] = sorted(rows, key=lambda r: r['seed'])
        changed = True
    if changed:
        json.dump(cache, open(CACHE, 'w'))
    return cache


def plan_scores(model):
    out = []
    for d in run_dirs()[model]:
        p = f'{d}/result.json'
        if os.path.exists(p):
            out.append(tl.score(json.load(open(p))))
    return out


# ---------------------------------------------------------------- fig1: training diagnostics
def diag_histories(arm):
    return [json.load(open(p)) for p in sorted(glob.glob(f'{TS}/diag_{arm}_s*/history.json'))]


def series(hs, key, eval_only=False):
    rows = [[e for e in h if (key in e) or not eval_only] for h in hs]
    n = min(len(r) for r in rows)
    x = [e['epoch'] for e in rows[0][:n]]
    return x, [[e.get(key, np.nan) for e in r[:n]] for r in rows]


def fig1():
    H = {a: diag_histories(a) for a in ('quantum', 'gaussian')}
    if not all(H.values()):
        raise SystemExit('fig1 needs runs/teammate_style/diag_* (scripts/teammate_style.sh)')
    ns = {a: len(h) for a, h in H.items()}
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    # (a) losses
    a = ax[0, 0]
    for arm in H:
        for key, ls in (('g_loss', '-'), ('d_loss', '--')):
            x, ys = series(H[arm], key)
            band(a, x, ys, C[arm], ls=ls)
    a.set(title='Generator and discriminator loss', xlabel='Epoch', ylabel='WGAN-GP loss (no "better" direction)')
    a.legend(handles=[Line2D([], [], color='#555', ls='-', label='Generator'),
                      Line2D([], [], color='#555', ls='--', label='Discriminator')], loc='upper right',
             frameon=False)
    # (b) circuit gradient norm
    a = ax[0, 1]
    x, ys = series(H['quantum'], 'angle_grad_norm')
    band(a, x, ys, C['quantum'])
    a.set(title='Gradient reaching the circuit (QTabGAN only)', xlabel='Epoch',
          ylabel='L2 norm of d(G loss)/d(circuit angles)')
    a.set_ylim(bottom=0)
    g = np.nanmean(ys, 0)
    # (c) validation accuracy / macro F1 / AUROC
    a = ax[0, 2]
    styles = {'eval_accuracy': ('-', 'Accuracy'), 'eval_macro_f1': ('--', 'Macro F1'), 'eval_auroc': (':', 'AUROC')}
    for arm in H:
        for key, (ls, _) in styles.items():
            x, ys = series(H[arm], key, eval_only=True)
            band(a, x, ys, C[arm], ls=ls, marker='o')
    maj = np.nanmean(series(H['quantum'], 'eval_majority_accuracy', True)[1])
    a.axhline(maj, color='#777', ls=(0, (6, 3)), lw=1.4)
    a.set(title='Validation utility (forest trained on synthetic data)', xlabel='Epoch',
          ylabel='Score on held-out real rows (higher is better)')
    a.legend(handles=[Line2D([], [], color='#555', ls=ls, label=lab) for ls, lab in styles.values()] +
             [Line2D([], [], color='#777', ls=(0, (6, 3)), label=f'Majority-class accuracy ({maj:.2f})')],
             loc='lower right', frameon=False)
    # (d) Brier
    a = ax[1, 0]
    for arm in H:
        x, ys = series(H[arm], 'eval_brier', True)
        band(a, x, ys, C[arm], marker='o')
    a.set(title='Probability error (Brier score)', xlabel='Epoch',
          ylabel='Mean squared error of P(>50K) (lower is better)')
    # (e) feature-mean error
    a = ax[1, 1]
    for arm in H:
        x, ys = series(H[arm], 'eval_feature_mean_error', True)
        band(a, x, ys, C[arm], marker='o')
    a.set(title='Numeric feature-mean error', xlabel='Epoch',
          ylabel='|synthetic mean - real mean| / real SD\n(averaged over 5 numeric columns, lower is better)')
    a.set_ylim(bottom=0)
    # (f) diversity
    a = ax[1, 2]
    for arm in H:
        for key, ls in (('eval_std_ratio', '-'), ('eval_category_coverage', '--')):
            x, ys = series(H[arm], key, True)
            band(a, x, ys, C[arm], ls=ls, marker='o')
    a.axhline(1, color='#777', ls=(0, (6, 3)), lw=1.4)
    a.set(title='Diversity checks', xlabel='Epoch', ylabel='Ratio (1 = matches real data)')
    a.set_ylim(0, 1.1)
    a.legend(handles=[Line2D([], [], color='#555', ls='-', label='Within-class SD ratio (synthetic / real)'),
                      Line2D([], [], color='#555', ls='--', label='Category coverage')], loc='lower right',
             frameon=False)
    fig.legend(handles=[Line2D([], [], color=C[a_], lw=3, label=f'{LABEL[a_]} ({ns[a_]} seeds)') for a_ in H],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    # finding
    fin = {arm: {k: np.nanmean(np.nanmean(series(H[arm], f'eval_{k}', True)[1], 0)[-5:]) for k in
                 ('macro_f1', 'auroc', 'std_ratio')} for arm in H}
    lead = 'QTabGAN' if fin['quantum']['macro_f1'] > fin['gaussian']['macro_f1'] else 'The Gaussian control'
    title = (f'Both models learn steadily and the gradient reaching the circuit keeps growing; validation macro F1 over '
             f'the last 50 epochs {fin["quantum"]["macro_f1"]:.2f} (QTabGAN) vs {fin["gaussian"]["macro_f1"]:.2f} (Gaussian)')
    headline(fig, title, 'Validation split, 150 epochs, evaluated every 10 epochs. Line = mean over seeds, '
             'band = min-max. Utility panels: random forest (100 trees) trained on 39k synthetic rows, '
             'scored on 7,815 held-out real rows; positive class = income >50K.')
    del lead, g
    save(fig, 'fig1_training_diagnostics')


def fig1_300():
    """Training diagnostics from the finished 300-epoch runs (runs/training_length, validation split, 3 seeds,
    evaluated every 25 epochs with the CTAB-GAN+ metrics). Available before the diag_* runs finish."""
    H = {a: [json.load(open(p)) for p in sorted(glob.glob(f'runs/training_length/{a}_s*/history.json'))]
         for a in ('quantum', 'gaussian')}
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    a = ax[0, 0]
    for arm in H:
        for key, ls in (('g_loss', '-'), ('d_loss', '--')):
            x, ys = series(H[arm], key)
            band(a, x, ys, C[arm], ls=ls)
    a.set(title='Generator and discriminator loss', xlabel='Epoch', ylabel='WGAN-GP loss (no "better" direction)')
    a.legend(handles=[Line2D([], [], color='#555', ls='-', label='Generator'),
                      Line2D([], [], color='#555', ls='--', label='Discriminator')], loc='upper right', frameon=False)
    a = ax[0, 1]
    x, ys = series(H['quantum'], 'angle_norm')
    band(a, x, ys, C['quantum'])
    an = np.nanmean(ys, 0)
    a.set(title='Circuit parameters keep changing (QTabGAN only)', xlabel='Epoch',
          ylabel='L2 norm of the 64 circuit angles (radians)')
    a.set_ylim(bottom=0)
    a = ax[0, 2]
    for arm in H:
        x, ys = series(H[arm], 'seconds')
        band(a, x, np.asarray(ys) / 60, C[arm])
    a.set(title='Training cost', xlabel='Epoch', ylabel='Wall-clock minutes (3-6 runs sharing one CPU)')
    ref = json.load(open('runs/reference/result.json'))['marginals']
    finals = {}
    for a, (key, lab, fl) in zip(ax[1], (
            ('eval_f1', 'ML-utility gap: F1 (real - synthetic)', ref['ml_utility_mean_diff']['f1']),
            ('eval_avg_jsd', 'Category-share error (avg JSD)', ref['stat_sim']['avg_jsd']),
            ('eval_diff_corr', 'Correlation error (Diff. Corr)', ref['stat_sim']['diff_corr']))):
        for arm in H:
            x, ys = series(H[arm], key, True)
            band(a, x, ys, C[arm], marker='o')
            finals[(arm, key)] = np.nanmean(ys, 0)
        a.axhline(fl, color='#777', ls=(0, (6, 3)), lw=1.4)
        a.text(x[-1], fl, 'no-learning floor', ha='right', va='bottom', fontsize=10.5, color='#555')
        a.set(title=lab.split(':')[0].split(' (')[0], xlabel='Epoch', ylabel=f'{lab} (lower is better)')
        a.set_ylim(bottom=0)
    fig.legend(handles=[Line2D([], [], color=C[a_], lw=3, label=f'{LABEL[a_]} ({len(H[a_])} seeds)') for a_ in H],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    t = {a_: np.nanmean(series(H[a_], 'seconds')[1], 0)[-1] / 60 for a_ in H}
    headline(fig, f'Most improvement happens in the first ~100 epochs; QTabGAN ends with lower category error, the '
                  f'Gaussian control with lower correlation error and {t["quantum"] / t["gaussian"]:.0f}x less training time',
             'Validation split, 300 epochs, evaluated every 25 epochs; the ML-utility gap here uses 3 classifiers '
             '(logistic regression, decision tree, random forest). '
             'Line = mean over seeds, band = min-max. Floor = columns sampled independently.')
    del an
    save(fig, 'fig1_training_300epochs')


def fig1b():
    """Teammate's bottom-of-page plot: accuracy / macro F1 / AUROC / AP vs epoch, one panel per model."""
    H = {a: diag_histories(a) for a in ('quantum', 'gaussian')}
    met = {'eval_accuracy': ('Accuracy', '#2a78d6', 'o'), 'eval_macro_f1': ('Macro F1', '#c2185b', 's'),
           'eval_auroc': ('AUROC', '#008300', '^'), 'eval_ap': ('Average precision', '#8a6d00', 'D')}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    finals = {}
    for a, arm in zip(axes, H):
        for key, (lab, col, mk) in met.items():
            x, ys = series(H[arm], key, True)
            band(a, x, ys, col, label=lab, marker=mk)
            finals[(arm, key)] = np.nanmean(np.nanmean(ys, 0)[-5:])
        maj = np.nanmean(series(H[arm], 'eval_majority_accuracy', True)[1])
        a.axhline(maj, color='#777', ls=(0, (6, 3)), lw=1.4, label=f'Majority-class accuracy ({maj:.2f})')
        a.set(title=f'{LABEL[arm]} ({len(H[arm])} seeds)', xlabel='Epoch', ylim=(0, 1))
        a.title.set_color(C[arm])
    axes[0].set_ylabel('Validation score (higher is better)')
    axes[1].legend(loc='lower right', frameon=False)
    q, g = finals[('quantum', 'eval_ap')], finals[('gaussian', 'eval_ap')]
    headline(fig, f'Validation scores rise over the first ~50 epochs, then level off; average precision over the last 50 epochs '
                  f'{q:.2f} (QTabGAN) vs {g:.2f} (Gaussian)',
             'Random forest (100 trees) trained on each epoch\'s synthetic table, scored on held-out real rows '
             '(validation split). Line = mean over seeds, band = min-max. Chance AP = 0.24 (share of >50K).')
    save(fig, 'fig1b_validation_scores')


# ---------------------------------------------------------------- fig2: ROC + PR
def fig2():
    E = rf_eval()
    models = ['real'] + [m for m in CORE if m in E]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.6))
    for m in models:
        rows = E[m]
        tpr = np.mean([r['roc_tpr'] for r in rows], 0)
        prec = np.mean([r['pr_precision'] for r in rows], 0)
        au = fmt([r['metrics']['auroc'] for r in rows])
        ap = fmt([r['metrics']['ap'] for r in rows])
        lw, ls = (2.2, '--') if m == 'real' else (2.2, '-')
        a1.plot(GRID, tpr, color=C[m], lw=lw, ls=ls, label=f'{SHORT[m]}: AUROC {au}')
        a2.plot(GRID, prec, color=C[m], lw=lw, ls=ls, label=f'{SHORT[m]}: AP {ap}')
    base = 1 - E['real'][0]['metrics']['majority_accuracy']
    a1.plot([0, 1], [0, 1], color='#999', ls=':', lw=1.4, label='Chance (AUROC 0.5)')
    a2.axhline(base, color='#999', ls=':', lw=1.4, label=f'Chance (base rate {base:.2f})')
    a1.set(xlabel='False positive rate (share of <=50K rows flagged)', ylabel='True positive rate (recall of >50K)',
           title='ROC curve', xlim=(0, 1), ylim=(0, 1.01))
    a2.set(xlabel='Recall of >50K', ylabel='Precision of >50K predictions', title='Precision-recall curve',
           xlim=(0, 1), ylim=(0, 1.01))
    a1.legend(loc='lower right', frameon=False)
    a2.legend(loc='upper right', frameon=False)
    for a in (a1, a2):
        a.set_aspect('equal')
    au = {m: np.mean([r['metrics']['auroc'] for r in E[m]]) for m in models}
    ap = {m: np.mean([r['metrics']['ap'] for r in E[m]]) for m in models}
    best_syn = max(CORE, key=lambda m: ap.get(m, -1))
    headline(fig, f'Forests trained on synthetic data rank incomes well (AUROC {min(au[m] for m in CORE if m in au):.2f}'
                  f'-{max(au[m] for m in CORE if m in au):.2f}) but lose most on precision-recall; '
                  f'{SHORT[best_syn]} is closest to real ({ap[best_syn]:.2f} vs {ap["real"]:.2f} AP)',
             'Random forest (100 trees) trained on each synthetic table (or the real training rows), tested on the '
             'real test split (9,769 rows). Curves = mean over seeds (5 unless the legend says 1); legend = mean ± SD. '
             'Positive = income >50K.')
    save(fig, 'fig2_roc_pr')


# ---------------------------------------------------------------- fig3: empirical CDFs
def fig3():
    import pandas as pd
    E = rf_eval()
    df, spec, _, _ = data()
    dirs = run_dirs()
    fakes = {m: pd.read_csv(f'{dirs[m][0]}/fake.csv') for m in CORE if os.path.exists(f'{dirs[m][0]}/fake.csv')}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    wins = {m: 0 for m in fakes}
    for a, col in zip(axes.flat, NUMERIC_SHOWN):
        hi = 20000 if col == 'capital-gain' else np.percentile(df[col], 99.7)
        xs = np.sort(df[col].to_numpy())
        a.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=C['real'], lw=2.6, label='Real data', drawstyle='steps-post')
        ks = {}
        for m, f in fakes.items():
            v = np.sort(f[col].to_numpy())
            ks[m] = mean_sd([r['ks'][col] for r in E[m]])
            a.plot(v, np.arange(1, len(v) + 1) / len(v), color=C[m], lw=1.8, drawstyle='steps-post',
                   label=f'{SHORT[m]}  (KS {fmt([r["ks"][col] for r in E[m]])})')
        wins[min(ks, key=lambda m: ks[m][0])] += 1
        a.set(xlim=(df[col].min(), hi), ylim=(0, 1.01), title=col,
              xlabel=f'{col} ({UNITS[col]}; axis cut at ' + ('20,000; 92% of real rows are 0)' if col == 'capital-gain'
                                                            else 'the real 99.7th percentile)'),
              ylabel='Share of rows at or below x')
        a.legend(loc='lower right', frameon=False, fontsize=10.5)
    top = max(wins, key=wins.get)
    headline(fig, f'{SHORT[top]} has the smallest KS distance on {wins[top]} of 4 numeric columns; QTabGAN and the '
                  'Gaussian control smooth over the 40-hours-per-week spike',
             'Empirical CDFs of the full real table vs one synthetic table per model (seed 0). KS statistic = largest '
             'vertical gap between the curves (0 = identical), mean ± SD over the available seeds.')
    save(fig, 'fig3_cdfs')


# ---------------------------------------------------------------- fig4: forest plot
def fig4():
    E = rf_eval()
    ref = json.load(open('runs/reference/result.json'))
    order = [m for m in ALL if m in E] + ['real']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    rng = np.random.default_rng(0)
    ypos = {m: i for i, m in enumerate(reversed(order))}
    for m in order:
        y = ypos[m]
        f1 = [r['metrics']['macro_f1'] for r in E[m]]
        sc = plan_scores(m) if m != 'real' else [tl.score(ref['real_copy'])]
        for a, v in ((a1, f1), (a2, sc)):
            mu, sd = mean_sd(v)
            a.errorbar(mu, y, xerr=sd, fmt='s' if m != 'real' else 'D', color=C[m], ms=9, capsize=5, lw=2, zorder=3)
            if len(v) > 1:
                a.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=18, color=C[m], alpha=0.55, lw=0, zorder=2)
            a.text(mu, y + 0.3, f'{mu:.3f}' + (f' ± {sd:.3f}' if len(v) > 1 else ''), ha='center', va='bottom',
                   fontsize=10.5)
    maj_f1 = 0.5 * (2 * E['real'][0]['metrics']['majority_accuracy'] / (1 + E['real'][0]['metrics']['majority_accuracy']))
    a1.axvline(maj_f1, color='#999', ls=':', lw=1.4)
    a1.text(maj_f1, -0.75, f' always predict <=50K\n ({maj_f1:.2f})', fontsize=10, color='#555', va='bottom')
    a2.axvline(tl.score(ref['marginals']), color='#999', ls='--', lw=1.4)
    a2.text(tl.score(ref['marginals']), -0.75, ' no-learning floor', fontsize=10, color='#555', va='bottom')
    n = {m: len(E[m]) for m in order}
    a1.set_yticks(list(ypos.values()), [f'{LABEL[m]}\n(n={n[m]}{" forest seeds" if m == "real" else (" seed" if n[m] == 1 else " seeds")})'
                                        for m in ypos])
    a1.set(xlabel='Test macro F1 of a forest trained on this data (higher is better)', title='Downstream utility',
           ylim=(-0.9, len(order) - 0.3))
    a2.set(xlabel='Overall score: mean of scaled F1 gap, JSD, Diff. Corr (lower is better)',
           title='Overall fidelity (plan objective)')
    a2.set_xlim(left=0)
    for a in (a1, a2):
        a.grid(axis='y', visible=False)
    f = {m: np.mean([r['metrics']['macro_f1'] for r in E[m]]) for m in order}
    headline(fig, f'Searched QTabGAN ({f["quantum"]:.3f}), the Gaussian control ({f["gaussian"]:.3f}) and CTGAN '
                  f'({f.get("ctgan", np.nan):.3f}) are tied on macro F1; all far above the literal paper settings '
                  f'({f.get("literal", np.nan):.3f})',
             'Test split, 150 epochs. Square = mean, bar = ±1 SD, dots = individual seeds. Real reference: forest '
             'trained on real rows (left); real training rows used as synthetic data (right, single value).')
    save(fig, 'fig4_forest')


# ---------------------------------------------------------------- fig5: results table
def fig5():
    E = rf_eval()
    cols = [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1'), ('auroc', 'AUROC'), ('ap', 'Avg. precision'),
            ('minority_recall', 'Recall of >50K')]
    order = ['real'] + [m for m in ('quantum', 'gaussian', 'ctgan', 'first_pass', 'literal') if m in E]
    maj = E['real'][0]['metrics']['majority_accuracy']
    base = 1 - maj
    rows, md = [], []
    for m in order:
        cells = []
        for k, _ in cols:
            cells.append(fmt([r['metrics'][k] for r in E[m]]))
        rows.append([('Real training rows (reference)' if m == 'real' else LABEL[m])] + cells)
    maj_row = ['Majority class (always <=50K)', f'{maj:.3f}', f'{0.5 * 2 * maj / (1 + maj):.3f}', '0.500',
               f'{base:.3f}', '0.000']
    rows.append(maj_row)
    header = ['Training data for the forest'] + [c for _, c in cols]
    fig, ax = plt.subplots(figsize=(16, 0.55 * (len(rows) + 1)))
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=header, cellLoc='center', colLoc='center',
                 colWidths=[0.34] + [0.132] * len(cols), bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#1f2d3d')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == 0:
            cell.set_text_props(ha='left', color=C.get(order[r - 1], '#555') if r <= len(order) else '#555',
                                fontweight='bold')
            cell._loc = 'left'
        if 0 < r <= len(order) and order[r - 1] == 'real':
            cell.set_facecolor('#f2f2f2')
        if r == len(rows):
            cell.set_facecolor('#fafafa')
            cell.set_text_props(color='#666')
    f = {m: np.mean([r['metrics']['minority_recall'] for r in E[m]]) for m in order}
    acc = {m: np.mean([r['metrics']['accuracy'] for r in E[m]]) for m in order}
    lit = (f' The paper-literal settings score below always guessing <=50K ({acc["literal"]:.3f} vs {maj:.3f}).'
           if acc.get('literal', 1) < maj else '')
    headline(fig, f'Real data wins on accuracy ({acc["real"]:.3f}); QTabGAN data finds the most >50K earners '
                  f'(recall {f["quantum"]:.2f} vs real {f["real"]:.2f}) at lower accuracy ({acc["quantum"]:.3f})',
             'Random forest (100 trees), tested on the real test split (9,769 rows, 24% >50K). Mean ± SD over generator '
             'seeds (5 unless a cell says 1; real: 5 forest seeds).\nThreshold 0.5 for accuracy, F1 and recall.' + lit)
    save(fig, 'fig5_results_table')
    md.append('| ' + ' | '.join(header) + ' |')
    md.append('|' + '---|' * len(header))
    md += ['| ' + ' | '.join(r) + ' |' for r in rows]
    with open(f'{OUT}/results_table.md', 'w') as fh:
        fh.write('Random forest (100 trees) trained on each data source, tested on the real Adult test split '
                 '(9,769 rows; positive class income >50K, 24%). Mean ± SD over 5 generator seeds (real: 5 forest '
                 'seeds).\n\n' + '\n'.join(md) + '\n')
    print(f'wrote {OUT}/results_table.md')


# ---------------------------------------------------------------- fig6: diversity bars
def fig6():
    E = rf_eval()
    models = [m for m in ALL if m in E]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    rng = np.random.default_rng(0)
    for a, (key, lab) in zip(axes, (('std_ratio', 'Numeric variation ratio\n(within-class SD, synthetic / real)'),
                                    ('category_coverage', 'Category coverage\n(share of real categories generated)'))):
        for i, m in enumerate(models):
            v = np.array([r['diversity'][key] for r in E[m]]) * 100
            a.bar(i, v.mean(), color=C[m], width=0.68)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            if sd:
                a.errorbar(i, v.mean(), yerr=sd, color='#333', capsize=4, lw=1.2)
            a.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=16, color='black', zorder=3, lw=0)
            a.text(i, max(v.max(), v.mean() + sd) + 2, f'{v.mean():.0f}%' + ('\n(1 seed)' if len(v) == 1 else ''), ha='center',
                   va='bottom', fontsize=11)
        a.axhline(100, color='#222', ls='--', lw=1.4)
        a.text(-0.45, 101.5, 'real data = 100%', ha='left', va='bottom', fontsize=10.5, color='#333')
        a.set_xticks(range(len(models)), [SHORT[m].replace(' ', '\n', 1) for m in models])
        a.set(ylabel=f'{lab.splitlines()[0]} (% of real)', title=lab, ylim=(0, 125))
        a.grid(axis='x', visible=False)
    sr = {m: np.mean([r['diversity']['std_ratio'] for r in E[m]]) for m in models}
    cv = {m: np.mean([r['diversity']['category_coverage'] for r in E[m]]) for m in models}
    headline(fig, f'Searched models keep most of the real variation (QTabGAN {sr["quantum"]:.0%}, Gaussian '
                  f'{sr["gaussian"]:.0%}); the paper-literal settings collapse to {sr.get("literal", np.nan):.0%}; '
                  f'only CTGAN generates every real category ({cv.get("ctgan", np.nan):.0%})',
             'Test-split runs (bar = mean, line = SD, dots = seeds; 5 seeds unless marked). Ratio below 100% = synthetic rows '
             'vary less than real rows of the same class; coverage counts each (column, category) pair once.')
    save(fig, 'fig6_diversity')


# ---------------------------------------------------------------- fig7: Optuna
def fig7():
    import optuna
    from optuna.importance import FanovaImportanceEvaluator, get_param_importances
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = 'sqlite:///runs/optuna/studies.db'
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), gridspec_kw={'width_ratios': [1.3, 1, 1]})
    a = axes[0]
    imps = {}
    for arm in ('quantum', 'gaussian'):
        study = optuna.load_study(study_name=f'score_{arm}', storage=st)
        done = [t for t in study.trials if t.state.name in ('COMPLETE', 'PRUNED')]
        done = sorted(done, key=lambda t: t.number)[:40]  # the plan's budget: first 40 finished trials
        comp = [t for t in done if t.state.name == 'COMPLETE']
        x = [t.number for t in comp]
        v = [t.value for t in comp]
        a.scatter(x, v, s=26, color=C[arm], alpha=0.45, lw=0)
        bs = np.minimum.accumulate(v)
        a.step(x + [done[-1].number], list(bs) + [bs[-1]], where='post', color=C[arm], lw=2.4,
               label=f'{SHORT[arm]} study: best {bs[-1]:.3f} ({len(comp)} completed, {len(done) - len(comp)} pruned)')
        sub = optuna.create_study(direction='minimize')
        sub.add_trials(comp)
        imps[arm] = get_param_importances(sub, evaluator=FanovaImportanceEvaluator(seed=0))
    a.set(xlabel='Trial number (order of the search)', ylabel='Validation score (lower is better)',
          title='Optimization history (completed trials)')
    a.set_ylim(0, np.percentile([t for t in a.collections[0].get_offsets()[:, 1]] +
                                [t for t in a.collections[1].get_offsets()[:, 1]], 90) * 1.2)
    a.legend(loc='upper right', frameon=False, fontsize=10.5)
    for a, arm in zip(axes[1:], ('quantum', 'gaussian')):
        it = sorted(imps[arm].items(), key=lambda kv: kv[1])
        a.barh(range(len(it)), [v for _, v in it], color=C[arm], height=0.66)
        a.set_yticks(range(len(it)), [k for k, _ in it])
        for i, (_, v) in enumerate(it):
            a.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10.5)
        a.set(xlabel='fANOVA importance (share of score variance)', title=f'Parameter importance: {SHORT[arm]}',
              xlim=(0, max(v for _, v in it) * 1.25))
        a.grid(axis='y', visible=False)
    tq = max(imps['quantum'], key=imps['quantum'].get)
    tg = max(imps['gaussian'], key=imps['gaussian'].get)
    headline(fig, f'QTabGAN reaches a lower validation score (0.192 vs 0.217); the most important setting is '
                  f'{tq} for QTabGAN and {tg} for the Gaussian control',
             'Step-3 Optuna studies, 40 trials per arm, TPE sampler, successive-halving pruning. Importance from fANOVA '
             'on the 21 completed trials per arm (pruned trials have no final score), so it is a rough guide.')
    save(fig, 'fig7_optuna')


# ---------------------------------------------------------------- fig8: settings table
def fig8():
    sel = json.load(open('runs/training_length/selected.json'))
    q, g = sel['quantum']['params'], sel['gaussian']['params']
    rows = [
        ('Epochs', '150', '150', 'Not stated'),
        ('Batch size', q['batch_size'], g['batch_size'], 'Not stated'),
        ('Optimizer', 'Adam', 'Adam', 'Not stated'),
        ('Learning rate (mapper + discriminator)', f"{q['lr_classical']:.2e}", f"{g['lr_classical']:.2e}", 'Not stated'),
        ('Learning rate (circuit angles)', f"{q['lr_quantum']:.2e}", 'no circuit', 'Not stated'),
        ('Adam betas', q['betas'], g['betas'], 'Not stated'),
        ('Discriminator steps per generator step', q['d_steps'], g['d_steps'], 'Not stated'),
        ('Adversarial loss', 'WGAN-GP (λ = 10)', 'WGAN-GP (λ = 10)', 'Minimax GAN objective; loss not named'),
        ('Mapper hidden layers', q['mapper_dims'], g['mapper_dims'], 'Feed-forward ReLU; widths not stated'),
        ('Discriminator hidden layers', q['disc_dims'], g['disc_dims'], 'Feed-forward ReLU; widths not stated'),
        ('Categorical output', q['cat_output'], g['cat_output'], 'Linear mapper output'),
        ('Noise source', '8-qubit circuit, 4 layers, 300 shots', 'N(0, I), 256 dims', '8-qubit circuit, L layers'),
        ('Noise into the circuit', f"angle encoding, U[0, {q['noise_scale']:.2f}π)", '-',
         'Two incompatible descriptions'),
        ('Initial circuit angles', '0.01 · N(0, 1)', '-', 'Uniform in [0, 2π)'),
        ('Circuit output scaling', 'probabilities × 256 (mean 1)', '-', 'Not stated'),
        ('Numeric preprocessing', 'clip 1-99th pct, z-score, scale to [0, π]', 'same', 'Stated (we follow it)'),
        ('Seeds', '0-4', '0-4', 'Not stated'),
        ('How chosen', 'Optuna, 40 trials, validation split', 'Optuna, 40 trials, validation split', '-'),
    ]
    fig, ax = plt.subplots(figsize=(17, 0.42 * (len(rows) + 1)))
    ax.axis('off')
    t = ax.table(cellText=[[str(c) for c in r] for r in rows],
                 colLabels=['Setting', 'Our best QTabGAN', 'Our best Gaussian control', 'Paper'],
                 cellLoc='left', colLoc='left', colWidths=[0.27, 0.25, 0.22, 0.26], bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(11.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#1f2d3d')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == 3 and ('Not stated' in cell.get_text().get_text() or 'not' in cell.get_text().get_text()):
            cell.set_text_props(color='#b23b3b')
        elif r % 2 == 0:
            cell.set_facecolor('#fafafa')
    unstated = sum('not' in r[3].lower() for r in rows)
    headline(fig, f'{unstated} of the {len(rows)} settings below are not stated in the paper; ours were chosen by an '
                  'equal-budget search',
             'The two configs used in every figure in this folder (Optuna step-3 trials, confirmed on the test split). '
             'Red = the paper does not give a value.')
    save(fig, 'fig8_settings')


# ---------------------------------------------------------------- poster
def poster():
    import matplotlib.image as mpimg
    E = rf_eval()
    W, H = 24, 15
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor('#f7f8f5')
    navy, teal = '#13293d', '#0f7c80'

    def box(x, y, w, h, color):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='square,pad=0', transform=fig.transFigure,
                                          facecolor=color, edgecolor='none', zorder=0))

    def heading(x, y, s):
        box(x, y - 0.002, 0.004, 0.022, teal)
        fig.text(x + 0.008, y, s, fontsize=17, fontweight='bold', color=navy, va='bottom')

    def para(x, y, s, size=12.5, color='#222', w=None):
        fig.text(x, y, s, fontsize=size, color=color, va='top', wrap=False, linespacing=1.4)

    def image(path, rect, title=None):
        ax = fig.add_axes(rect)
        ax.imshow(mpimg.imread(path))
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=12, loc='left', color=navy, fontweight='bold')

    box(0, 0.985, 1, 0.015, teal)
    fig.text(0.02, 0.968, 'QTABGAN / REPRODUCTION + CONTROLLED COMPARISON / ADULT CENSUS DATA', fontsize=11.5,
             color=teal, fontweight='bold', va='top')
    fig.text(0.02, 0.945, 'Does the quantum circuit help? Reproducing QTabGAN with a Gaussian-noise control',
             fontsize=30, fontweight='bold', color=navy, va='top')
    fig.text(0.02, 0.895, 'Mitansh | Cold Qubits | independent reimplementation of arXiv:2602.12704',
             fontsize=13.5, color='#444', va='top')
    fig.text(0.98, 0.895, '48,842 rows · 5 seeds · 40 Optuna trials per arm · 1 dataset', fontsize=12.5,
             color='#444', va='top', ha='right')
    f1 = {m: mean_sd([r['metrics']['macro_f1'] for r in E[m]]) for m in E}
    box(0.02, 0.832, 0.96, 0.046, navy)
    fig.text(0.03, 0.861, 'With equal tuning, the quantum circuit does no better than plain Gaussian noise in the same '
             'network.', fontsize=17, fontweight='bold', color='white', va='center')
    fig.text(0.03, 0.842, f'Test macro F1: QTabGAN {f1["quantum"][0]:.3f} ± {f1["quantum"][1]:.3f}, Gaussian control '
             f'{f1["gaussian"][0]:.3f} ± {f1["gaussian"][1]:.3f}, CTGAN {fmt([r["metrics"]["macro_f1"] for r in E["ctgan"]]) if "ctgan" in E else "n/a"}, '
             f'real data {f1["real"][0]:.3f}. Paired differences are within seed noise.', fontsize=12.5,
             color='#d8e4ee', va='center')

    # column 1: question, method, diagram
    x0 = 0.02
    heading(x0, 0.795, 'Question')
    para(x0, 0.788, 'QTabGAN (2026) replaces a GAN\'s random noise with the output\nof a quantum circuit and reports '
         'near-real utility on Adult\n(accuracy gap 0.2 points). No code was released.\n\n'
         '(A) Can its Adult result be reproduced?\n(B) Is the gain due to the quantum circuit?')
    heading(x0, 0.615, 'Method')
    d = fig.add_axes([x0, 0.405, 0.30, 0.2])
    d.axis('off')
    d.set_xlim(0, 10)
    d.set_ylim(0, 6)

    def node(x, y, w, h, s, fc, tc='white'):
        d.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08,rounding_size=0.15', fc=fc, ec='none'))
        d.text(x + w / 2, y + h / 2, s, ha='center', va='center', fontsize=10.5, color=tc)

    def arrow(x1, y1, x2, y2):
        d.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color='#444', lw=1.4))
    node(0, 4.3, 3.0, 1.3, '8-qubit circuit\nRY/RZ + CNOT ring\n300 shots', C['quantum'])
    node(0, 2.6, 3.0, 1.2, 'Gaussian noise\nN(0, I), same width\n(control)', C['gaussian'])
    node(3.9, 3.4, 2.6, 1.6, 'Classical mapper\n2-layer MLP\n+ class label', '#3d4f60')
    node(7.3, 3.4, 2.7, 1.6, 'Synthetic row\n(14 columns)', teal)
    node(3.9, 0.6, 2.6, 1.4, 'Discriminator\nreal or synthetic?', navy)
    node(7.3, 0.6, 2.7, 1.4, 'Real training rows\n39,073', '#e3e8ec', '#222')
    arrow(3.0, 4.95, 3.9, 4.4)
    arrow(3.0, 3.2, 3.9, 3.9)
    arrow(6.5, 4.2, 7.3, 4.2)
    arrow(8.65, 3.4, 6.5, 1.6)
    arrow(7.3, 1.3, 6.5, 1.3)
    d.text(0.0, 0.05, 'Identical networks; only the noise source differs.', fontsize=10.5, color='#444')
    para(x0, 0.395, '• Exact CTAB-GAN+ evaluation (train on synthetic, test on real).\n'
         '• Pre-registered plan: screening (112 runs), Optuna 40 trials\n   per arm on a validation split, '
         'same budget and pruning.\n'
         '• Confirmation: 150 epochs, 5 seeds, real test split.\n'
         '• Baselines: CTGAN, literal paper settings, real data.', size=12)
    box(x0, 0.115, 0.30, 0.12, '#e6efee')
    fig.text(x0 + 0.01, 0.225, 'My contribution', fontsize=15, fontweight='bold', color=teal, va='top')
    para(x0 + 0.01, 0.198, 'Reimplemented QTabGAN and the exact evaluation, built the\nGaussian control, ran the '
         'pre-registered search and\nconfirmation, and showed the gain comes from tuning,\nnot from the circuit.',
         size=12)

    # column 2: results figures
    x1 = 0.345
    heading(x1, 0.795, 'Results across five seeds')
    image(f'{OUT}/fig4_forest.png', [x1, 0.47, 0.31, 0.31])
    image(f'{OUT}/fig2_roc_pr.png', [x1, 0.17, 0.31, 0.29])
    para(x1, 0.16, 'Random forest (100 trees) trained on each synthetic table, tested\non 9,769 real rows. '
         'Positive class = income >50K (24%).', size=11, color='#444')

    # column 3: fidelity + conclusion + limits
    x2 = 0.67
    heading(x2, 0.795, 'Fidelity and training')
    image(f'{OUT}/fig3_cdfs.png', [x2, 0.52, 0.31, 0.26])
    image(f'{OUT}/fig1_training_diagnostics.png', [x2, 0.30, 0.31, 0.21])
    heading(x2, 0.265, 'Conclusion')
    para(x2, 0.257, '(A) The paper\'s 0.2-point gap is matched only under a guessed\n    kNN evaluation, and best by the '
         'Gaussian control.\n(B) Searched QTabGAN and the Gaussian control are statistically\n    tied (p = 0.22-0.38); '
         'QTabGAN costs 1.7-11x the training time.', size=12)
    heading(x2, 0.155, 'Limitations')
    para(x2, 0.147, 'One dataset, noiseless simulator (no hardware), 5 seeds; CTGAN and\npaper-literal baselines are '
         'not tuned; 14 paper settings are unstated,\nso this tests our faithful version, not the authors\' code.',
         size=11.5)
    fig.text(0.02, 0.02, 'Code and full results: github.com/Mitanshcodes/QTabGan (REPRODUCIBILITY.md, DETAILS.md). '
             'All numbers on the real test split unless noted.', fontsize=10.5, color='#666')
    save(fig, 'poster')


FIGS = {'fig1': fig1, 'fig1_300': fig1_300, 'fig1b': fig1b, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5, 'fig6': fig6,
        'fig7': fig7, 'fig8': fig8, 'poster': poster}

if __name__ == '__main__':
    todo = sys.argv[1:] or list(FIGS)
    for name in todo:
        FIGS[name]()
