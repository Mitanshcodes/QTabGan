"""QTabGAN generator / discriminator / training loop (paper Sec. 3 and 4.2).

    [noise source] -> 2^n vector --+
                                   +--> classical mapper (ReLU MLP, linear out) --> encoded row x
    one-hot class label y ---------+
    discriminator: [x ; y] --> ReLU MLP --> scalar logit

`source` selects what produces the 2^n vector:
    quantum         the variational circuit + finite-shot sampler (QTabGAN)
    frozen_quantum  the same circuit, angles never trained (does training the circuit matter?)
    gaussian        plain N(0, I) noise of the same width (standard classical conditional GAN)
Everything else is identical across sources, so differences are attributable to the source.
"""

import math
import time

import numpy as np
import pennylane as qml
import torch
from torch import nn
from torch.nn import functional as F

from qtabgan.config import Config
from qtabgan.sim import RingCircuit


# ----------------------------------------------------------------------------- quantum core
class QuantumSource(nn.Module):
    """VQC: H on all qubits -> [optional noise RY] -> L x (RY, RZ per qubit; CNOT ring) -> probs."""

    def __init__(self, cfg: Config, trainable=True):
        super().__init__()
        self.cfg = cfg
        n, L = cfg.n_qubits, cfg.n_layers
        self.n_out = 2**n
        w = torch.rand(L, n, 2) * 2 * math.pi  # paper: random in [0, 2pi)
        self.weights = nn.Parameter(w, requires_grad=trainable)
        self.shot_rng = np.random.default_rng(cfg.seed + 12345)
        self.sim = RingCircuit(n) if cfg.backend == 'torch' else None

        dev = qml.device('default.qubit', wires=n)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(weights, z):
            for q in range(n):
                qml.Hadamard(wires=q)
            if z is not None:  # per-sample angle encoding, broadcast over the batch
                for q in range(n):
                    qml.RY(z[:, q], wires=q)
            for layer in range(L):
                for q in range(n):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)
                for q in range(n):
                    qml.CNOT(wires=[q, (q + 1) % n])
            return qml.probs(wires=range(n))

        self.circuit = circuit

    def exact_probs(self, batch_size):
        z = None
        if self.cfg.noise_encoding == 'angle':
            z = torch.rand(batch_size, self.cfg.n_qubits) * (math.pi / 2)
        if self.sim is not None:
            p = self.sim(self.weights, z, batch_size)
        elif z is not None:
            p = self.circuit(self.weights, z)  # (B, 2^n), a different state per row
        else:
            p = self.circuit(self.weights, None).unsqueeze(0).expand(batch_size, -1)  # (B, 2^n)
        return p.float()

    def forward(self, batch_size):
        p = self.exact_probs(batch_size)
        if self.cfg.n_shots:
            # Finite-shot measurement: N shots per row, drawn from the Born distribution. Same
            # distribution as running the circuit N times on an ideal simulator.
            with torch.no_grad():
                pv = p.detach().double().clamp_min(0).numpy()
                pv = pv / pv.sum(axis=1, keepdims=True)
                counts = self.shot_rng.multinomial(self.cfg.n_shots, pv)
                empirical = torch.from_numpy(counts / self.cfg.n_shots).float()
            p = p + (empirical - p).detach()  # forward: shot estimate; backward: exact gradient
        if self.cfg.prob_scale == 'n_outcomes':
            p = p * self.n_out
        return p


class GaussianSource(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, batch_size):
        return torch.randn(batch_size, self.width)


# ----------------------------------------------------------------------------- networks
def mlp(dims_in, hidden):
    layers, d = [], dims_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers), d


class Generator(nn.Module):
    def __init__(self, cfg: Config, data_dim, n_classes, spans):
        super().__init__()
        self.cfg, self.spans, self.n_classes = cfg, spans, n_classes
        width = 2**cfg.n_qubits
        if cfg.source == 'quantum':
            self.source = QuantumSource(cfg, trainable=True)
        elif cfg.source == 'frozen_quantum':
            self.source = QuantumSource(cfg, trainable=False)
        elif cfg.source == 'gaussian':
            self.source = GaussianSource(width)
        else:
            raise ValueError(cfg.source)
        self.body, d = mlp(width + n_classes, cfg.mapper_dims)
        self.head = nn.Linear(d, data_dim)  # paper: final layer linear

    def forward(self, y_onehot, hard=False):
        feats = self.source(y_onehot.shape[0])
        out = self.head(self.body(torch.cat([feats, y_onehot], dim=1)))
        return self._activate(out, hard)

    def _activate(self, out, hard):
        mode = self.cfg.cat_output
        if mode == 'linear':
            return out
        pieces = []
        for kind, _, a, b in self.spans:
            block = out[:, a:b]
            if kind == 'cat':
                if mode == 'softmax':
                    block = F.softmax(block, dim=1)
                elif mode == 'gumbel':
                    block = F.gumbel_softmax(block, tau=0.2, hard=hard)
            pieces.append(block)
        return torch.cat(pieces, dim=1)


class Discriminator(nn.Module):
    def __init__(self, cfg: Config, data_dim, n_classes):
        super().__init__()
        self.body, d = mlp(data_dim + n_classes, cfg.disc_dims)
        self.head = nn.Linear(d, 1)

    def forward(self, x, y_onehot):
        return self.head(self.body(torch.cat([x, y_onehot], dim=1)))


# ----------------------------------------------------------------------------- training
def _gradient_penalty(D, real, fake, y, lam=10.0):
    a = torch.rand(real.size(0), 1)
    inter = (a * real + (1 - a) * fake).requires_grad_(True)
    out = D(inter, y)
    g = torch.autograd.grad(out.sum(), inter, create_graph=True)[0]
    return lam * ((g.norm(2, dim=1) - 1) ** 2).mean()


class QTabGAN:
    def __init__(self, cfg: Config, prep):
        self.cfg, self.prep = cfg, prep
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.G = Generator(cfg, prep.dim, prep.n_classes, prep.spans)
        self.D = Discriminator(cfg, prep.dim, prep.n_classes)
        q = [p for n, p in self.G.named_parameters() if n.startswith('source.') and p.requires_grad]
        c = [p for n, p in self.G.named_parameters() if not n.startswith('source.')]
        groups = [{'params': c, 'lr': cfg.lr_classical}]
        if q:
            groups.append({'params': q, 'lr': cfg.lr_quantum})
        self.optG = torch.optim.Adam(groups, betas=cfg.betas)
        self.optD = torch.optim.Adam(self.D.parameters(), lr=cfg.lr_classical, betas=cfg.betas)
        self.history = []

    def n_params(self):
        g = sum(p.numel() for p in self.G.parameters())
        q = sum(p.numel() for n, p in self.G.named_parameters() if n.startswith('source.'))
        d = sum(p.numel() for p in self.D.parameters())
        return {'generator_total': g, 'generator_quantum': q, 'discriminator': d}

    def fit(self, X, y, log_every=10, callback=None):
        cfg = self.cfg
        X = torch.from_numpy(X)
        Y = F.one_hot(torch.from_numpy(y).long(), self.prep.n_classes).float()
        n = len(X)
        steps = max(n // cfg.batch_size, 1)
        bce = nn.BCEWithLogitsLoss()
        t0 = time.time()
        for epoch in range(cfg.epochs):
            gl, dl = [], []
            for _ in range(steps):
                for _ in range(cfg.d_steps):
                    idx = torch.randint(0, n, (cfg.batch_size,))
                    xr, yr = X[idx], Y[idx]
                    xf = self.G(yr).detach()
                    dr, df = self.D(xr, yr), self.D(xf, yr)
                    if cfg.loss == 'bce':
                        lossD = bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df))
                    else:
                        lossD = df.mean() - dr.mean() + _gradient_penalty(self.D, xr, xf, yr)
                    self.optD.zero_grad()
                    lossD.backward()
                    self.optD.step()
                idx = torch.randint(0, n, (cfg.batch_size,))
                yr = Y[idx]
                df = self.D(self.G(yr), yr)
                lossG = bce(df, torch.ones_like(df)) if cfg.loss == 'bce' else -df.mean()
                self.optG.zero_grad()
                lossG.backward()
                self.optG.step()
                gl.append(lossG.item())
                dl.append(lossD.item())
            rec = {'epoch': epoch + 1, 'g_loss': float(np.mean(gl)), 'd_loss': float(np.mean(dl)),
                   'seconds': time.time() - t0}
            if hasattr(self.G.source, 'weights'):
                rec['angle_norm'] = float(self.G.source.weights.detach().norm())
            self.history.append(rec)
            if log_every and (epoch + 1) % log_every == 0:
                print(f"epoch {epoch+1:4d}  G {rec['g_loss']:.3f}  D {rec['d_loss']:.3f}  "
                      f"{rec['seconds']:.0f}s", flush=True)
            if callback:
                callback(self, epoch + 1)
        return self

    @torch.no_grad()
    def sample(self, n_rows, chunk=2000, rng=None):
        rng = rng or np.random.default_rng(self.cfg.seed + 1)
        y = rng.choice(self.prep.n_classes, size=n_rows, p=self.prep.class_freq)
        self.G.eval()
        xs = []
        for i in range(0, n_rows, chunk):
            yy = F.one_hot(torch.from_numpy(y[i:i + chunk]).long(), self.prep.n_classes).float()
            xs.append(self.G(yy, hard=True).numpy())
        self.G.train()
        return self.prep.inverse(np.concatenate(xs), y)
