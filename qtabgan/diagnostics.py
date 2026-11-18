"""Extra diagnostics for training curves and the teammate-style figures.

Separate from evaluate.py on purpose: the CTAB-GAN+ evaluation there is unchanged. Everything here
works on raw tables (Adult.csv is already label-encoded), with income >50K (= 1) as the positive,
minority class.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, recall_score,
                             roc_auc_score)


def rf_fit_predict(train, test, target, seed=0):
    """Random forest (100 trees) trained on `train`, returns P(positive) on `test`."""
    feats = [c for c in train.columns if c != target]
    ytr = train[target].to_numpy().astype(int)
    if len(np.unique(ytr)) < 2:  # a collapsed generator can emit a single class
        return np.full(len(test), float(ytr[0]))
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    rf.fit(train[feats].to_numpy(), ytr)
    return rf.predict_proba(test[feats].to_numpy())[:, list(rf.classes_).index(1)]


def classifier_metrics(train, test, target, seed=0):
    """Train on `train` (synthetic or real), score on real `test` rows."""
    y = test[target].to_numpy().astype(int)
    p = rf_fit_predict(train, test, target, seed)
    pred = (p >= 0.5).astype(int)
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro')),
            'auroc': float(roc_auc_score(y, p)),
            'ap': float(average_precision_score(y, p)),
            'brier': float(np.mean((p - y) ** 2)),
            'minority_recall': float(recall_score(y, pred, pos_label=1, zero_division=0)),
            'majority_accuracy': float(max(y.mean(), 1 - y.mean()))}


def diversity_metrics(real, fake, numeric, categorical, target):
    """feature_mean_error: mean over numeric columns of |mean_fake - mean_real| / sd_real.
    std_ratio: sd_fake / sd_real per numeric column within each class, averaged (1 = same spread,
    0 = every synthetic row identical). category_coverage: share of the (column, category) pairs
    present in the real table that also appear in the synthetic one (target column excluded)."""
    fme = np.mean([abs(fake[c].mean() - real[c].mean()) / real[c].std() for c in numeric])
    ratios = []
    for k in np.unique(real[target]):
        r, f = real[real[target] == k], fake[fake[target] == k]
        for c in numeric:
            sd = r[c].std()
            if sd > 0:
                ratios.append(f[c].std() / sd if len(f) > 1 else 0.0)
    cats = [c for c in categorical if c != target]
    present = sum(len(set(real[c].unique()) & set(fake[c].unique())) for c in cats)
    total = sum(real[c].nunique() for c in cats)
    return {'feature_mean_error': float(fme), 'std_ratio': float(np.nanmean(ratios)),
            'category_coverage': float(present / total)}
