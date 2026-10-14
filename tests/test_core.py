"""Correctness tests. Run: python -m pytest -q tests"""

import os
import sys

import numpy as np
import pandas as pd
import pennylane as qml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qtabgan.config import Config  # noqa: E402
from qtabgan.data import DATASETS, Preprocessor, ctabgan_split  # noqa: E402
from qtabgan.model import QTabGAN, QuantumSource  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')


def _pl_probs(weights, z, n):
    dev = qml.device('default.qubit', wires=n)

    @qml.qnode(dev)
    def c():
        for q in range(n):
            qml.Hadamard(wires=q)
        if z is not None:
            for q in range(n):
                qml.RY(z[q], wires=q)
        for layer in range(weights.shape[0]):
            for q in range(n):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            for q in range(n):
                qml.CNOT(wires=[q, (q + 1) % n])
        return qml.probs(wires=range(n))

    return np.asarray(c())


def test_torch_sim_matches_pennylane():
    torch.manual_seed(0)
    for n, L in [(3, 2), (5, 3), (8, 4)]:
        cfg = Config(n_qubits=n, n_layers=L, noise_encoding='angle', n_shots=None, prob_scale='none')
        src = QuantumSource(cfg)
        z = torch.rand(4, n)
        p_torch = src.sim(src.weights, z).detach().numpy()
        w = src.weights.detach().numpy()
        for b in range(4):
            ref = _pl_probs(w, z[b].numpy(), n)
            assert np.allclose(p_torch[b], ref, atol=1e-6), (n, L, np.abs(p_torch[b] - ref).max())
        ref0 = _pl_probs(w, None, n)
        assert np.allclose(src.sim(src.weights, None)[0].detach().numpy(), ref0, atol=1e-6)


def test_torch_sim_gradients_match_pennylane():
    cfg = Config(n_qubits=4, n_layers=2, n_shots=None, prob_scale='none')
    torch.manual_seed(1)
    a = QuantumSource(cfg)
    b = QuantumSource(Config(n_qubits=4, n_layers=2, n_shots=None, prob_scale='none', backend='pennylane'))
    b.weights.data.copy_(a.weights.data)
    target = torch.rand(16)
    for src in (a, b):
        ((src(1)[0] - target) ** 2).sum().backward()
    assert torch.allclose(a.weights.grad, b.weights.grad.float(), atol=1e-5)


def test_shots_are_unbiased_and_noisy():
    cfg = Config(n_qubits=3, n_layers=1, n_shots=512, prob_scale='none')
    src = QuantumSource(cfg)
    exact = src.exact_probs(1)[0].detach()
    est = src(4000).detach()
    assert not torch.allclose(est[0], est[1])  # rows differ: shot noise is the randomness
    assert torch.allclose(est.mean(0), exact, atol=2e-3)  # ...and averages to the Born probs


def test_preprocessor_roundtrip_and_range():
    df = pd.read_csv(os.path.join(ROOT, DATASETS['adult']['path'])).sample(3000, random_state=0)
    spec = DATASETS['adult']
    p = Preprocessor(spec['categorical'], spec['integer'], spec['target']).fit(df)
    X, y = p.transform(df, add_noise=False)
    num = X[:, :len(p.numeric)]
    assert num.min() >= -1e-6 and num.max() <= np.pi + 1e-6  # paper: [0, pi]
    back = p.inverse(X, y)
    for c in p.categorical + [spec['target']]:
        assert (back[c].to_numpy() == df[c].to_numpy()).all()
    for c in p.numeric:  # equal up to the 1/99 percentile clipping
        s = p.num_stats[c]
        assert np.allclose(back[c], np.round(np.clip(df[c], s['lo'], s['hi'])), atol=1)


def test_split_matches_ctabgan():
    df = pd.read_csv(os.path.join(ROOT, DATASETS['adult']['path']))
    tr, te = ctabgan_split(df, 'income')
    assert len(tr) == 39073 and len(te) == 9769


def test_quantum_angles_get_gradients_and_move():
    df = pd.read_csv(os.path.join(ROOT, DATASETS['adult']['path'])).sample(1000, random_state=0)
    spec = DATASETS['adult']
    p = Preprocessor(spec['categorical'], spec['integer'], spec['target']).fit(df)
    X, y = p.transform(df)
    m = QTabGAN(Config(epochs=1, batch_size=200), p)
    w0 = m.G.source.weights.detach().clone()
    m.fit(X, y, log_every=0)
    assert (m.G.source.weights.detach() - w0).abs().max() > 1e-4
    fake = m.sample(500)
    assert list(fake.columns) == list(df.columns) and len(fake) == 500


def test_frozen_quantum_does_not_move():
    df = pd.read_csv(os.path.join(ROOT, DATASETS['adult']['path'])).sample(1000, random_state=0)
    spec = DATASETS['adult']
    p = Preprocessor(spec['categorical'], spec['integer'], spec['target']).fit(df)
    X, y = p.transform(df)
    m = QTabGAN(Config(epochs=1, batch_size=200, source='frozen_quantum'), p)
    w0 = m.G.source.weights.detach().clone()
    m.fit(X, y, log_every=0)
    assert torch.equal(m.G.source.weights.detach(), w0)
