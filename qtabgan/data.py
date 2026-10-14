"""Dataset loading, the CTAB-GAN+ train/test split, and QTabGAN's preprocessing (Sec. 4.1)."""

import numpy as np
import pandas as pd
from sklearn import model_selection

# Column roles for the label-encoded Adult.csv shipped with CTAB-GAN+ (Real_Datasets/Adult.csv).
DATASETS = {
    'adult': {
        'path': 'data/Adult.csv',
        'target': 'income',
        'categorical': ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                        'race', 'gender', 'native-country', 'income'],
        'integer': ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
    },
}


def load(name, root='.'):
    spec = DATASETS[name]
    df = pd.read_csv(f'{root}/{spec["path"]}')
    return df, spec


def ctabgan_split(df, target, test_ratio=0.2):
    """Exactly the split CTAB-GAN+ uses for training and evaluation (stratified, random_state=42).

    Returns (train_df, test_df) with the original column order. The same call is made inside
    ``evaluate.ml_utility`` so the generator never sees the evaluation test rows.
    """
    values = df.to_numpy()
    X, y = values[:, :-1], values[:, -1]
    assert df.columns[-1] == target, 'target must be the last column (CTAB-GAN+ convention)'
    idx = np.arange(len(df))
    tr_idx, te_idx = model_selection.train_test_split(
        idx, test_size=test_ratio, stratify=y, random_state=42
    )
    # Sanity: identical to splitting the arrays themselves, as CTAB-GAN+ does.
    Xtr, _, _, _ = model_selection.train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)
    assert np.array_equal(Xtr, X[tr_idx])
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


class Preprocessor:
    """QTabGAN Sec. 4.1: clip numeric to [p1, p99], z-score, min-max into [0, pi], add tiny noise;
    one-hot categorical features. The target is NOT part of x: it is the conditioning label."""

    def __init__(self, categorical, integer, target, clip=(1, 99), noise_var=1e-5):
        self.target = target
        self.categorical = [c for c in categorical if c != target]
        self.integer = set(integer)
        self.clip = clip
        self.noise_var = noise_var

    def fit(self, df):
        self.columns = [c for c in df.columns if c != self.target]
        self.numeric = [c for c in self.columns if c not in self.categorical]
        self.num_stats = {}
        for c in self.numeric:
            x = df[c].to_numpy(dtype=float)
            lo, hi = np.percentile(x, self.clip)
            xc = np.clip(x, lo, hi)
            mu, sd = xc.mean(), xc.std()
            z = (xc - mu) / (sd + 1e-8)
            self.num_stats[c] = dict(lo=lo, hi=hi, mu=mu, sd=sd, zmin=z.min(), zmax=z.max())
        self.cat_levels = {c: np.sort(df[c].unique()) for c in self.categorical}
        self.classes = np.sort(df[self.target].unique())
        self.class_freq = df[self.target].value_counts(normalize=True).reindex(self.classes).to_numpy()

        # Output layout of the encoded vector: numeric block first, then one block per categorical.
        self.spans, pos = [], 0
        for c in self.numeric:
            self.spans.append(('num', c, pos, pos + 1))
            pos += 1
        for c in self.categorical:
            k = len(self.cat_levels[c])
            self.spans.append(('cat', c, pos, pos + k))
            pos += k
        self.dim = pos
        self.n_classes = len(self.classes)
        return self

    def transform(self, df, add_noise=True, rng=None):
        rng = rng or np.random.default_rng(0)
        parts = []
        for c in self.numeric:
            s = self.num_stats[c]
            x = np.clip(df[c].to_numpy(dtype=float), s['lo'], s['hi'])
            z = (x - s['mu']) / (s['sd'] + 1e-8)
            u = (z - s['zmin']) / (s['zmax'] - s['zmin'] + 1e-8) * np.pi
            if add_noise and self.noise_var > 0:
                u = u + rng.normal(0, np.sqrt(self.noise_var), size=u.shape)
            parts.append(u[:, None])
        for c in self.categorical:
            lv = self.cat_levels[c]
            onehot = (df[c].to_numpy()[:, None] == lv[None, :]).astype(float)
            parts.append(onehot)
        X = np.concatenate(parts, axis=1).astype(np.float32)
        y = np.searchsorted(self.classes, df[self.target].to_numpy())
        return X, y

    def inverse(self, X, y):
        out = {}
        for kind, c, a, b in self.spans:
            if kind == 'num':
                s = self.num_stats[c]
                z = X[:, a] / np.pi * (s['zmax'] - s['zmin']) + s['zmin']
                x = np.clip(z * s['sd'] + s['mu'], s['lo'], s['hi'])
                out[c] = np.round(x).astype(int) if c in self.integer else x
            else:
                out[c] = self.cat_levels[c][X[:, a:b].argmax(axis=1)]
        out[self.target] = self.classes[y]
        return pd.DataFrame(out)[self.columns + [self.target]]
