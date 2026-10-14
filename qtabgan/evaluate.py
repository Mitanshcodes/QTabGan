"""Evaluation.

1. ``ml_utility`` and ``stat_sim`` re-implement CTAB-GAN+'s ``model/eval/evaluation.py``
   (github.com/Team-TUD/CTAB-GAN-Plus) step for step. This is the protocol behind the baseline
   numbers QTabGAN compares against (CTAB-GAN+ 5.23% / 0.090 / JSD 0.039 / Diff.Corr 2.03 are
   CTAB-GAN+ Table III), so only numbers computed this way are comparable to that table.
   Faithful details, including quirks:
     - real data split 80/20, stratified, random_state=42; MinMax scaler fit on real train
     - synthetic data also split 80/20 (stratified, rs=42); its scaler is fit on ALL synthetic X
     - classifiers: lr, dt, rf, mlp, svm with the repo's settings; accuracy in %, F1 = mean of
       per-class F1 for binary targets
     - reported value = real score - mean over synthetic datasets (signed, per classifier)
     - "JSD" is scipy's jensenshannon(base=2), i.e. the JS *distance* (sqrt of divergence)
     - "Diff. Corr" is the Frobenius norm of (assoc(real) - assoc(fake)) using dython
       associations: Pearson / Cramer's V (dython default, what the repo code calls) / corr. ratio
     - stat_sim compares the FULL real csv (train + test) with the synthetic table
   One harmless change: SVC is fit with probability=False and AUC uses decision_function. For a
   binary target Platt scaling is monotone, so AUC is identical; predict() is unaffected; it is ~5x
   faster.

2. ``qtabgan_paper_metrics`` computes the same quantities as the QTabGAN paper *writes them*
   (Sec. 5.2.2): JS *divergence* (squared distance) and the *average* absolute Pearson difference
   over feature pairs. These are on different scales from (1); see REPRODUCIBILITY.md.
"""

import numpy as np
import pandas as pd
from dython.nominal import associations
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn import metrics, model_selection, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

CLASSIFIERS = ('lr', 'dt', 'rf', 'mlp', 'svm')


def _model(name):
    return {
        'lr': lambda: LogisticRegression(random_state=42, max_iter=500),
        'svm': lambda: svm.SVC(random_state=42),
        'dt': lambda: tree.DecisionTreeClassifier(random_state=42),
        'rf': lambda: RandomForestClassifier(random_state=42),
        'mlp': lambda: MLPClassifier(random_state=42, max_iter=100),
    }[name]()


def _scores(Xtr, ytr, Xte, yte, name):
    m = _model(name)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    acc = metrics.accuracy_score(yte, pred) * 100
    if len(np.unique(ytr)) > 2:
        auc = metrics.roc_auc_score(yte, m.predict_proba(Xte), average='weighted', multi_class='ovr')
        f1 = metrics.precision_recall_fscore_support(yte, pred, average='weighted')[2]
    else:
        s = m.decision_function(Xte) if name == 'svm' else m.predict_proba(Xte)[:, 1]
        auc = metrics.roc_auc_score(yte, s)
        f1 = metrics.precision_recall_fscore_support(yte, pred)[2].mean()
    return [acc, auc, f1]


def ml_utility(real_df, fake_dfs, classifiers=CLASSIFIERS, test_ratio=0.2, return_raw=False):
    real = real_df.to_numpy()
    d = real.shape[1]
    Xr, yr = real[:, :d - 1], real[:, -1]
    Xtr, Xte, ytr, yte = model_selection.train_test_split(
        Xr, yr, test_size=test_ratio, stratify=yr, random_state=42)
    sc = MinMaxScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    real_res = np.array([_scores(Xtr_s, ytr, Xte_s, yte, c) for c in classifiers])

    fake_res = []
    for fake_df in fake_dfs:
        fake = fake_df[real_df.columns].to_numpy()
        Xf, yf = fake[:, :d - 1], fake[:, -1]
        strat = yf if np.min(np.unique(yf, return_counts=True)[1]) >= 2 else None
        Xf_tr, _, yf_tr, _ = model_selection.train_test_split(
            Xf, yf, test_size=test_ratio, stratify=strat, random_state=42)
        scf = MinMaxScaler().fit(Xf)
        fake_res.append([_scores(scf.transform(Xf_tr), yf_tr, Xte_s, yte, c) for c in classifiers])
    fake_mean = np.array(fake_res).mean(axis=0)

    diff = pd.DataFrame(real_res - fake_mean, index=list(classifiers), columns=['acc', 'auc', 'f1'])
    if return_raw:
        real = pd.DataFrame(real_res, index=list(classifiers), columns=['acc', 'auc', 'f1'])
        fake = pd.DataFrame(fake_mean, index=list(classifiers), columns=['acc', 'auc', 'f1'])
        return diff, real, fake
    return diff


def stat_sim(real_df, fake_df, cat_cols):
    fake_df = fake_df[real_df.columns]
    rc = associations(real_df, nominal_columns=cat_cols, compute_only=True)['corr']
    fc = associations(fake_df, nominal_columns=cat_cols, compute_only=True)['corr']
    corr_dist = float(np.linalg.norm(rc.to_numpy() - fc.to_numpy()))
    jsd, wd = [], []
    for col in real_df.columns:
        if col in cat_cols:
            rp = real_df[col].value_counts(normalize=True)
            fp = fake_df[col].value_counts(normalize=True)
            cats = sorted(fp.index.tolist())
            # As in CTAB-GAN+: categories only in fake are skipped via real_pdf[i] lookup (KeyError
            # in their code); we treat them as 0 in real, which is the only well-defined reading.
            r = [rp.get(i, 0.0) for i in cats]
            f = [fp[i] for i in cats]
            for z in set(rp.index) - set(fp.index):
                r.append(rp[z])
                f.append(0.0)
            jsd.append(distance.jensenshannon(r, f, 2.0))
        else:
            sc = MinMaxScaler().fit(real_df[[col]].to_numpy(dtype=float))
            a = sc.transform(real_df[[col]].to_numpy(dtype=float)).ravel()
            b = sc.transform(fake_df[[col]].to_numpy(dtype=float)).ravel()
            wd.append(wasserstein_distance(a, b))
    return {'avg_wd': float(np.mean(wd)), 'avg_jsd': float(np.mean(jsd)), 'diff_corr': corr_dist}


def qtabgan_paper_metrics(real_df, fake_df, cat_cols):
    """Metrics exactly as QTabGAN Sec. 5.2.2 writes them (see module docstring)."""
    fake_df = fake_df[real_df.columns]
    jsd_div = []
    for col in cat_cols:
        rp = real_df[col].value_counts(normalize=True)
        fp = fake_df[col].value_counts(normalize=True)
        cats = sorted(set(rp.index) | set(fp.index))
        jsd_div.append(distance.jensenshannon([rp.get(c, 0) for c in cats],
                                              [fp.get(c, 0) for c in cats], 2.0) ** 2)
    r = real_df.astype(float).corr(method='pearson').to_numpy()
    f = fake_df.astype(float).corr(method='pearson').to_numpy()
    iu = np.triu_indices_from(r, k=1)
    diff = np.abs(r[iu] - f[iu])
    diff = diff[~np.isnan(diff)]
    return {'avg_js_divergence': float(np.mean(jsd_div)), 'avg_abs_pearson_diff': float(np.mean(diff))}
