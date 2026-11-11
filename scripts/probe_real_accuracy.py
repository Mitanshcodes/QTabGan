"""Which evaluation setup gives QTabGAN's reported real-data result on Adult (accuracy 0.7760,
F1 0.7619, Table 3 of arXiv:2602.12704v2)? Trains common scikit-learn classifiers on the REAL
CTAB-GAN+ train split, with default settings, with and without scaling, and scores them on the test
split. A setup that lands near 0.776 / 0.762 is a candidate for the evaluation the paper used.

Usage: python scripts/probe_real_accuracy.py > runs/probe_real_accuracy.md
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
from sklearn import metrics  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from qtabgan.data import ctabgan_split, load  # noqa: E402

warnings.filterwarnings('ignore')
TARGET_ACC, TARGET_F1 = 0.7760, 0.7619

df, spec = load('adult')
tr, te = ctabgan_split(df, spec['target'])
Xtr, ytr = tr.drop(columns=spec['target']).to_numpy(float), tr[spec['target']].to_numpy()
Xte, yte = te.drop(columns=spec['target']).to_numpy(float), te[spec['target']].to_numpy()

models = {
    'LogisticRegression': lambda: LogisticRegression(max_iter=100),
    'DecisionTree': lambda: DecisionTreeClassifier(random_state=42),
    'RandomForest': lambda: RandomForestClassifier(random_state=42),
    'GradientBoosting': lambda: GradientBoostingClassifier(random_state=42),
    'GaussianNB': lambda: GaussianNB(),
    'KNN (k=5)': lambda: KNeighborsClassifier(),
    'LinearSVC': lambda: LinearSVC(random_state=42),
    'MLP': lambda: MLPClassifier(random_state=42, max_iter=200),
}
scalings = {'none': None, 'minmax': MinMaxScaler, 'standard': StandardScaler}

rows = []
for mname, mk in models.items():
    for sname, sc in scalings.items():
        a, b = Xtr, Xte
        if sc:
            s = sc().fit(Xtr)
            a, b = s.transform(Xtr), s.transform(Xte)
        m = mk().fit(a, ytr)
        p = m.predict(b)
        acc = metrics.accuracy_score(yte, p)
        f1m = metrics.f1_score(yte, p, average='macro')
        f1w = metrics.f1_score(yte, p, average='weighted')
        f1b = metrics.f1_score(yte, p)
        dist = abs(acc - TARGET_ACC) + min(abs(f1m - TARGET_F1), abs(f1w - TARGET_F1), abs(f1b - TARGET_F1))
        rows.append((dist, mname, sname, acc, f1m, f1w, f1b))

rows.sort()
print(f'Target (QTabGAN Table 3, Adult, real data): accuracy {TARGET_ACC}, F1 {TARGET_F1}. '
      f'Majority-class accuracy: {max(yte.mean(), 1 - yte.mean()):.4f}.\n')
print('| classifier | scaling | accuracy | F1 macro | F1 weighted | F1 positive class | distance to target |')
print('|---|---|---|---|---|---|---|')
for dist, mname, sname, acc, f1m, f1w, f1b in rows:
    print(f'| {mname} | {sname} | {acc:.4f} | {f1m:.4f} | {f1w:.4f} | {f1b:.4f} | {dist:.4f} |')
