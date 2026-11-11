"""Build every figure in figures/ from files already in runs/ (no training). PNG (200 dpi) + PDF.

    python scripts/make_figures.py                 # all figures
    python scripts/make_figures.py fig3 figB       # a subset (prefix match)

Inputs (see figures/FIGURES.md for which runs feed which figure):
    runs/training_length/{quantum,gaussian}_s*/history.json     fig1, fig2   (validation split)
    runs/confirm/*_s*/{result.json,fake.csv}                     fig3-5       (test split, step 5)
    runs/search_final/paper_literal_s*, runs/final/ctgan_s*      fig3         (test split)
    runs/reference/result.json                                   floor / ceiling lines
    runs/optuna/studies.db                                       figA         (validation split)
    runs/screen/{arm}/*/s*/result.json                           figB         (validation split)
"Best" QTabGAN / Gaussian = scripts/training_length.best_configs (lowest mean step-5 score).
"""

import glob
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import training_length as tl  # noqa: E402

OUT = 'figures'
CONFIRM = os.environ.get('FIG_CONFIRM_ROOT', 'runs/confirm')
TLROOT = os.environ.get('FIG_TL_ROOT', 'runs/training_length')

# One colour per model, everywhere.
C = {'quantum': '#7B2CBF', 'gaussian': '#E8822A', 'ctgan': '#8C8C8C', 'literal': '#C9B1E6', 'real': '#222222'}
LABEL = {'quantum': 'QTabGAN (best searched config)',
         'gaussian': 'Gaussian control (same network, Gaussian noise)',
         'literal': 'QTabGAN, literal paper settings', 'ctgan': 'CTGAN (classical baseline)', 'real': 'Real data'}
SHORT = {'quantum': 'QTabGAN', 'gaussian': 'Gaussian control', 'literal': 'QTabGAN\n(literal paper)',
         'ctgan': 'CTGAN', 'real': 'Real'}
METRICS = [  # (key in result.json, axis label)
    (('ml_utility_mean_diff', 'f1'), 'ML-utility gap (F1 gap, lower is better)'),
    (('stat_sim', 'avg_jsd'), 'Category-share error (avg JSD, lower is better)'),
    (('stat_sim', 'diff_corr'), 'Correlation error (Diff. Corr, lower is better)'),
]
FLOOR_LABEL = 'No-learning floor (columns sampled independently)'
CEIL_LABEL = 'Ceiling (real training rows used as "synthetic")'

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.titlesize': 15, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6, 'font.family': 'DejaVu Sans',
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

# Adult.csv (CTAB-GAN+ copy) is label-encoded alphabetically, '?' first; verified against the UCI rows.
CODES = {
    'education': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc',
                  'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'],
    'occupation': ['? (missing)', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                   'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
                   'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support',
                   'Transport-moving'],
}


# ---------------------------------------------------------------- helpers
def get(r, path):
    for k in path:
        r = r[k]
    return r


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=200)
    plt.close(fig)
    print(f'wrote {OUT}/{name}.png/.pdf')


def headline(fig, title, subtitle=None):
    """Finding as the bold title, what-is-plotted as a smaller line under it."""
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    else:
        fig.canvas.draw()  # let constrained layout settle before placing the title above it
    h = fig.get_figheight()
    fig.text(0.01, 1 + (0.75 if subtitle else 0.3) / h, title, fontweight='bold', ha='left', va='bottom',
             fontsize=15)
    if subtitle:
        fig.text(0.01, 1 + 0.2 / h, subtitle, ha='left', va='bottom', fontsize=11.5, color='#444444')


def reference():
    return json.load(open('runs/reference/result.json'))


def best():
    b = tl.best_configs(CONFIRM)
    if set(b) != {'quantum', 'gaussian'}:
        raise SystemExit(f'step-5 results incomplete in {CONFIRM}: {b}')
    return {arm: n for arm, (n, _) in b.items()}


def results(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(pattern))]


def histories(arm):
    return [json.load(open(p)) for p in sorted(glob.glob(f'{TLROOT}/{arm}_s*/history.json'))]


def band(ax, x, ys, color, label, marker=None):
    ys = np.asarray(ys, float)
    ax.fill_between(x, np.nanmin(ys, 0), np.nanmax(ys, 0), color=color, alpha=0.2, lw=0)
    ax.plot(x, np.nanmean(ys, 0), color=color, lw=2, label=label, marker=marker, ms=5)


def real_and_fakes():
    from qtabgan.data import load
    df, spec = load('adult')  # step-5 runs evaluate against the full table (scripts/run.py, no --tune)
    names = best()
    fakes = {}
    for arm, n in names.items():
        p = f'{CONFIRM}/{n}_s0/fake.csv'
        if not os.path.exists(p):
            raise SystemExit(f'{p} missing: rerun `python3 scripts/run.py` with the flags in '
                             f'runs/confirm/commands.txt for {n}_s0 to regenerate it')
        fakes[arm] = pd.read_csv(p)
    return df, spec, fakes, names


# ---------------------------------------------------------------- fig1
def fig1():
    H = {arm: histories(arm) for arm in ('quantum', 'gaussian')}
    sel = json.load(open(f'{TLROOT}/selected.json'))
    loss = {arm: dict(sel[arm]['params']).get('loss') or tl_base_loss(arm) for arm in H}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for arm, hs in H.items():
        n = min(len(h) for h in hs)
        x = np.arange(1, n + 1)
        lab = f"{SHORT[arm]} ({'WGAN-GP' if loss[arm] == 'wgan_gp' else 'BCE'} loss, {len(hs)} seeds)"
        band(axes[0], x, [[e['g_loss'] for e in h[:n]] for h in hs], C[arm], lab)
        band(axes[1], x, [[e['d_loss'] for e in h[:n]] for h in hs], C[arm], lab)
        if arm == 'quantum' and 'angle_norm' in hs[0][0]:
            band(axes[2], x, [[e['angle_norm'] for e in h[:n]] for h in hs], C[arm], lab)
    axes[0].set(xlabel='Epoch', ylabel='Generator loss (no "better" direction)', title='Generator loss')
    axes[1].set(xlabel='Epoch', ylabel='Discriminator / critic loss (no "better" direction)',
                title='Discriminator loss')
    axes[2].set(xlabel='Epoch', ylabel='Norm of circuit angles (radians)',
                title='Circuit parameters (QTabGAN only)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    hq = H['quantum']

    def drift(arm):  # mean G loss, last fifth vs the fifth before it, relative to the loss's spread
        g = np.mean([[e['g_loss'] for e in h] for h in H[arm]], 0)
        k = max(len(g) // 5, 1)
        return abs(g[-k:].mean() - g[-2 * k:-k].mean()) / (g.std() + 1e-9)
    drifting = [SHORT[a] for a in H if drift(a) > 0.25]
    a0, a1 = np.mean([h[0]['angle_norm'] for h in hq]), np.mean([h[-1]['angle_norm'] for h in hq])
    same = loss['quantum'] == loss['gaussian']
    headline(fig, (f"Losses of both models level off by epoch {len(hq[0])}" if not drifting else
                   f"{' and '.join(drifting)} loss still drifting at epoch {len(hq[0])}") +
             f'; the circuit angles change by {100 * (a1 - a0) / a0:+.0f}% over training',
             'Mean over seeds (line) and min-max across seeds (band). '
             + ('Both use the same loss, but loss values still are not a quality measure.' if same else
                'WGAN-GP and BCE losses are on different scales: compare shapes, not levels.'))
    save(fig, 'fig1_training_curves')


def tl_base_loss(arm):
    from optuna_search import BASE
    return BASE[arm].get('loss')


# ---------------------------------------------------------------- fig2
def fig2():
    ref = reference()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    summary = {}
    for arm in ('quantum', 'gaussian'):
        hs = histories(arm)
        ev = [[e for e in h if 'eval_f1' in e] for h in hs]
        n = min(len(e) for e in ev)
        x = [e['epoch'] for e in ev[0][:n]]
        for ax, key in zip(axes, ('eval_f1', 'eval_avg_jsd', 'eval_diff_corr')):
            ys = [[e[key] for e in evs[:n]] for evs in ev]
            band(ax, x, ys, C[arm], f'{SHORT[arm]} ({len(hs)} seeds)', marker='o')
            summary[(arm, key)] = (x, np.mean(ys, 0))
    for ax, ((sec, key), lab) in zip(axes, METRICS):
        ax.axhline(ref['marginals'][sec][key], color='#444444', ls='--', lw=1.4, label=FLOOR_LABEL)
        ax.axvline(150, color='#999999', ls=':', lw=1.4, label='150 epochs (length used for the test-split runs)')
        ax.set(xlabel='Epochs trained', ylabel=lab, title=lab.split(' (')[0])
        ax.set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    # finding: where does each model reach its best mean score, and does 300 beat 150?
    parts = []
    for arm in ('quantum', 'gaussian'):
        x = summary[(arm, 'eval_f1')][0]
        s = np.mean([summary[(arm, k)][1] / tl.SCALES[m] for k, m in
                     (('eval_f1', 'f1'), ('eval_avg_jsd', 'avg_jsd'), ('eval_diff_corr', 'diff_corr'))], 0)
        i150 = x.index(150) if 150 in x else len(x) // 2
        parts.append((arm, x[int(np.argmin(s))], s[-1] - s[i150]))
    # combined-score change from 150 to the last epoch; |change| < 0.02 (well under seed noise) = no change
    verdict = {a: 'helps' if d < -0.02 else 'hurts' if d > 0.02 else 'none' for a, _, d in parts}
    if set(verdict.values()) == {'none'}:
        title = 'Training past 150 epochs neither helps nor hurts either model'
    else:
        title = '; '.join({'helps': f'Training past 150 epochs helps {SHORT[a]}',
                           'hurts': f'training past 150 epochs makes {SHORT[a]} worse',
                           'none': f'{SHORT[a]} does not change'}[v] for a, v in verdict.items())
        title = title[0].upper() + title[1:]
    headline(fig, title, 'Validation split, evaluated every 25 epochs. Line = mean over seeds, band = min-max. '
             'Mean combined score is lowest at epoch ' +
             ', '.join(f'{e} ({SHORT[a]})' for a, e, _ in parts) + '.')
    save(fig, 'fig2_training_length')


# ---------------------------------------------------------------- fig3
def fig3():
    ref, names = reference(), best()
    models = {
        'quantum': results(f'{CONFIRM}/{names["quantum"]}_s*/result.json'),
        'gaussian': results(f'{CONFIRM}/{names["gaussian"]}_s*/result.json'),
        'literal': results('runs/search_final/paper_literal_s*/result.json'),
        'ctgan': results('runs/final/ctgan_s*/result.json'),
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    rng = np.random.default_rng(0)
    wins = 0
    for ax, (path, lab) in zip(axes, METRICS):
        for i, (m, rs) in enumerate(models.items()):
            v = np.array([get(r, path) for r in rs])
            ax.bar(i, v.mean(), color=C[m], width=0.7, edgecolor='none')
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color='black', zorder=3, lw=0)
        q = np.mean([get(r, path) for r in models['quantum']])
        g = np.mean([get(r, path) for r in models['gaussian']])
        wins += g < q
        ax.axhline(get(ref['marginals'], path), color='#444444', ls='--', lw=1.4)
        ax.axhline(get(ref['real_copy'], path), color='#444444', ls=':', lw=1.6)
        fl, ce = get(ref['marginals'], path), get(ref['real_copy'], path)
        top = ax.get_ylim()[1]
        if fl < 0.03 * top:
            ax.set_ylim(top=top * 1.2)
            top *= 1.2
            ax.text(len(models) - 0.45, 0.99 * top, f'Floor ({fl:.3f}) and ceiling ({ce:.3f})\nare both near 0 here',
                    ha='right', va='top', fontsize=10.5, color='#444444')
        ax.set_xticks(range(len(models)), [f'{SHORT[m]}\n(n={len(rs)})' for m, rs in models.items()])
        ax.set(ylabel=lab, title=lab.split(' (')[0])
        ax.grid(axis='x', visible=False)
    handles = [Patch(color=C[m], label=LABEL[m]) for m in models] + [
        Line2D([], [], color='black', marker='o', ls='', ms=5, label='One seed'),
        Line2D([], [], color='#444444', ls='--', label=FLOOR_LABEL),
        Line2D([], [], color='#444444', ls=':', label=CEIL_LABEL)]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.19))
    title = {3: 'Gaussian noise beats the quantum circuit on all three metrics',
             0: 'The quantum circuit beats Gaussian noise on all three metrics'}.get(
        wins, f'Gaussian noise beats the quantum circuit on {wins} of 3 metrics')
    headline(fig, title, 'Test split, 150 epochs, full evaluation. Bar = mean over seeds, dot = one seed '
             '(n = seeds). The literal-paper and CTGAN runs come from earlier stages with 2 seeds.')
    save(fig, 'fig3_final_comparison')


# ---------------------------------------------------------------- fig4
def tvd(p, q):
    return 0.5 * np.abs(p - q).sum()


def fig4():
    df, spec, fakes, _ = real_and_fakes()
    tabs = {'real': df, **fakes}
    fig, axs = plt.subplots(2, 2, figsize=(17, 13), gridspec_kw={'height_ratios': [1, 1.9]})
    score = {'quantum': 0, 'gaussian': 0}
    for j, (col, lo, hi, w, unit) in enumerate([('age', 17, 90, 2, 'years'), ('hours-per-week', 1, 99, 3, 'hours')]):
        ax = axs[0, j]
        bins = np.arange(lo, hi + w, w)
        h = {k: np.histogram(np.clip(t[col], lo, hi), bins=bins)[0] / len(t) for k, t in tabs.items()}
        ax.fill_between(bins[:-1], h['real'] * 100, step='post', color='#BBBBBB', alpha=0.7, label=LABEL['real'])
        d = {}
        for arm in fakes:
            ax.step(bins[:-1], h[arm] * 100, where='post', color=C[arm], lw=2, label=SHORT[arm])
            d[arm] = tvd(h['real'], h[arm])
        score['quantum' if d['quantum'] < d['gaussian'] else 'gaussian'] += 1
        ax.set(xlabel=f'{col} ({unit}; values outside {lo}-{hi} clipped)', ylabel=f'Share of rows (%) per {w}-{unit[:-1]} bin',
               title=f"{col}: distance from real = {d['quantum']:.2f} QTabGAN, {d['gaussian']:.2f} Gaussian")
        ax.legend(frameon=False, loc='upper right')
    for j, col in enumerate(['education', 'occupation']):
        ax = axs[1, j]
        cats = sorted(df[col].unique())
        share = {k: t[col].value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() for k, t in tabs.items()}
        order = np.argsort(share['real'])
        y = np.arange(len(cats))
        for k, off in (('real', 0.27), ('quantum', 0), ('gaussian', -0.27)):
            ax.barh(y + off, share[k][order] * 100, height=0.27, color='#BBBBBB' if k == 'real' else C[k],
                    label=LABEL['real'] if k == 'real' else SHORT[k])
        d = {arm: tvd(share['real'], share[arm]) for arm in fakes}
        score['quantum' if d['quantum'] < d['gaussian'] else 'gaussian'] += 1
        ax.set_yticks(y, [CODES[col][c] for c in np.array(cats)[order]])
        ax.set(xlabel='Share of rows (%)', title=f"{col}: distance from real = {d['quantum']:.2f} QTabGAN, "
                                                   f"{d['gaussian']:.2f} Gaussian")
        ax.grid(axis='y', visible=False)
        ax.legend(frameon=False, loc='lower right')
    g = score['gaussian']
    title = (f'The Gaussian control matches the real distributions more closely than QTabGAN ({g} of 4 columns)'
             if g >= 3 else f'QTabGAN matches the real distributions more closely ({4 - g} of 4 columns)'
             if g <= 1 else 'QTabGAN and the Gaussian control each match 2 of 4 columns better')
    headline(fig, title, 'One synthetic table per model (seed 0, test-split run), same size as the real table. '
             '"Distance" = total variation distance between the two distributions (0 = identical, 1 = disjoint).')
    save(fig, 'fig4_real_vs_synthetic')


# ---------------------------------------------------------------- fig5
def fig5():
    from dython.nominal import associations
    df, spec, fakes, _ = real_and_fakes()
    A = {k: associations(t, nominal_columns=spec['categorical'], compute_only=True)['corr']
         for k, t in {'real': df, **fakes}.items()}
    D = {arm: (A[arm] - A['real']).abs() for arm in fakes}
    fro = {arm: float(np.linalg.norm(A[arm].to_numpy() - A['real'].to_numpy())) for arm in fakes}
    dmax = max(float(np.nanmax(d.to_numpy())) for d in D.values())
    cols = list(A['real'].columns)
    fig, axes = plt.subplots(2, 3, figsize=(19, 13.5), layout='constrained')
    for ax, k in zip(axes[0], ('real', 'quantum', 'gaussian')):
        im1 = ax.imshow(A[k].to_numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(LABEL[k] if k == 'real' else SHORT[k], color=C[k] if k != 'real' else 'black', fontweight='bold')
    for ax, arm in zip(axes[1, 1:], ('quantum', 'gaussian')):
        im2 = ax.imshow(D[arm].to_numpy(), cmap='Reds', vmin=0, vmax=dmax)
        ax.set_title(f'|{SHORT[arm]} - real|   (Diff. Corr = {fro[arm]:.2f})', color=C[arm], fontweight='bold')
    for ax in axes.flat[[0, 1, 2, 4, 5]]:
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=11)
        ax.set_yticks(range(len(cols)), cols, fontsize=11)
        ax.grid(False)
    ax = axes[1, 0]
    ax.axis('off')
    ax.text(0, 0.95, 'How to read this figure', fontweight='bold', va='top', transform=ax.transAxes)
    ax.text(0, 0.85, 'Top row: association between every pair\nof columns (numeric-numeric: Pearson r;\n'
            'category-category: Cramer\'s V;\nnumeric-category: correlation ratio).\n'
            'Red = strong positive, white = none.\n\n'
            'Bottom: absolute difference from the\nreal matrix. Whiter is better.\n'
            'Diff. Corr is the Frobenius norm of\nthe difference (lower is better).',
            va='top', transform=ax.transAxes, fontsize=11.5, linespacing=1.4)
    fig.colorbar(im1, ax=axes[0].tolist(), shrink=0.8, label='Association (-1 to 1)')
    fig.colorbar(im2, ax=axes[1, 1:].tolist(), shrink=0.8, label='|synthetic - real| (lower is better)')
    q, g = fro['quantum'], fro['gaussian']
    title = (f"QTabGAN distorts relationships between columns {q / g:.1f}x as much as the Gaussian control"
             if q > g else f"QTabGAN preserves relationships between columns better than the Gaussian control "
                           f"(Diff. Corr {q:.2f} vs {g:.2f})")
    headline(fig, title, 'Seed-0 synthetic table of each model (test-split run) vs the full real Adult table; '
             'same association measure as the Diff. Corr metric.')
    save(fig, 'fig5_correlation_heatmaps')


# ---------------------------------------------------------------- figA
def figA():
    import optuna
    from optuna_search import budget_trials, storage
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fig, ax = plt.subplots(figsize=(12, 6))
    msg, ymax = [], 0
    for arm in ('quantum', 'gaussian'):
        st = optuna.load_study(study_name=f'score_{arm}', storage=storage())
        ts = budget_trials(st, 40)
        comp = [(t.number, t.value) for t in ts if t.state.name == 'COMPLETE']
        prun = [(t.number, list(t.intermediate_values.values())[-1]) for t in ts if t.state.name == 'PRUNED']
        ax.scatter(*zip(*comp), color=C[arm], alpha=0.35, s=28, lw=0)
        if prun:
            ax.scatter(*zip(*prun), color=C[arm], alpha=0.35, marker='x', s=30)
        xs = sorted(t.number for t in ts)
        vals = dict(comp)
        run, cur, last = [], np.inf, None
        for x in xs:
            if x in vals and vals[x] < cur:
                cur, last = vals[x], x
            run.append(cur)
        ax.step(xs, run, where='post', color=C[arm], lw=2.4,
                label=f'{SHORT[arm]}: best so far ({len(comp)} completed, {len(prun)} pruned)')
        ymax = max(ymax, np.percentile([v for _, v in comp + prun], 90))
        msg.append(f'{SHORT[arm]} {cur:.2f} (found at trial {last})')
    ax.set_ylim(0, ymax * 1.15)
    ax.set(xlabel='Trial number (order of the search)', ylabel='Validation score (lower is better)')
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color='grey', marker='o', ls='', alpha=0.5, label='Completed trial (score at epoch 60)'),
          Line2D([], [], color='grey', marker='x', ls='', alpha=0.6, label='Pruned trial (score at epoch 20)')]
    ax.legend(handles=h, frameon=False, loc='upper right')
    headline(fig, 'Best validation score after 40 trials: ' + ' vs '.join(msg),
             'Optuna TPE search, 40 trials per model, 60 epochs, pruning at epoch 20. Score = mean of three '
             'scaled errors; points above the top edge are not shown.')
    save(fig, 'figA_search_progress')


# ---------------------------------------------------------------- figB
def figB():
    from collections import defaultdict
    import screen
    eff, sig, level, bestd = {}, {}, {}, {}
    for arm in ('quantum', 'gaussian'):
        res = defaultdict(dict)
        for p in glob.glob(f'{screen.ROOT}/{arm}/*/s*/result.json'):
            res[p.split('/')[-3]][int(p.split('/')[-2][1:])] = screen.metrics(json.load(open(p)))['score']
        res = {n: v for n, v in res.items() if len(v) == 2}
        sig[arm] = float(np.sqrt(np.mean([(v[0] - v[1]) ** 2 for v in res.values()])) / np.sqrt(2))
        b = np.mean(list(res['base'].values()))
        by = defaultdict(list)
        for n, v in res.items():
            if n != 'base':
                by[n.split('=')[0]].append((np.mean(list(v.values())) - b, n.split('=')[1]))
        for s, lv in by.items():
            d, l = max(lv, key=lambda t: abs(t[0]))
            eff[(s, arm)], level[(s, arm)] = d, ('exact' if s == 'n-shots' and l == '0' else l)
            bestd[(s, arm)] = min(lv)
    settings = sorted({s for s, _ in eff}, key=lambda s: max(abs(eff.get((s, a), 0)) for a in sig))
    fig, ax = plt.subplots(figsize=(13, 11))
    y = np.arange(len(settings))
    for arm, off in (('quantum', 0.2), ('gaussian', -0.2)):
        for i, s in enumerate(settings):
            if (s, arm) not in eff:
                ax.text(0.003, i + off, 'quantum-only setting', va='center', fontsize=9.5, color='#999999')
                continue
            d = eff[(s, arm)]
            ax.barh(i + off, d, height=0.38, color=C[arm])
            bd, bl = bestd[(s, arm)]
            if d > 0 and bd < 0:  # largest change was a harm, but another level helped: show it lighter
                ax.barh(i + off, bd, height=0.38, color=C[arm], alpha=0.4)
                ax.text(bd - 0.012, i + off, bl.replace('-', ' '), va='center', ha='right', fontsize=9.5,
                        color='#333333')
            ax.text(d + (0.012 if d >= 0 else -0.012), i + off, level[(s, arm)].replace('-', ' '), va='center',
                    ha='left' if d >= 0 else 'right', fontsize=9.5, color='#333333')
        for sgn in (1, -1):
            ax.axvline(sgn * 2 * sig[arm], color=C[arm], ls='--', lw=1.3)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y, [s.replace('-', '_') for s in settings])
    ax.grid(axis='y', visible=False)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.12, hi + 0.15)
    ax.set_xlabel('Change in validation score vs. the starting config (negative = better)')
    ax.legend(handles=[Patch(color=C[a], label=f'{SHORT[a]} (dashed: +/- 2 x seed noise = {2 * sig[a]:.2f})')
                       for a in ('quantum', 'gaussian')] +
              [Patch(color='grey', alpha=0.4, label='Lighter bar: best level, when the largest change was a harm')],
              frameon=False, loc='lower right')
    n_help = {a: sum(1 for (s, aa), (d, _) in bestd.items() if aa == a and d < -2 * sig[a]) for a in sig}
    headline(fig, ('No single change improves QTabGAN beyond seed noise; ' if n_help['quantum'] == 0 else
                   f"{n_help['quantum']} single changes improve QTabGAN beyond seed noise; ") +
             f"{n_help['gaussian']} settings improve the Gaussian control",
             'One-at-a-time screening, 40 epochs, 2 seeds per config. Each bar: the level (printed at its end) whose '
             '2-seed mean moved the score most. Beyond the dashed lines = larger than seed noise.')
    save(fig, 'figB_screening')


FIGS = {'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5, 'figA': figA, 'figB': figB}

if __name__ == '__main__':
    want = sys.argv[1:] or list(FIGS)
    for name, f in FIGS.items():
        if any(name.startswith(w) or w.startswith(name) for w in want):
            f()
