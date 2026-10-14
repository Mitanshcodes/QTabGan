"""A small batched statevector simulator in PyTorch for exactly the QTabGAN circuit.

Same gates and wire ordering as PennyLane (wire 0 = most significant bit of the basis index), so
`probs` matches `qml.probs(wires=range(n))`. tests/test_sim_matches_pennylane.py checks this to
1e-6. It exists only for speed: PennyLane's default.qubit is ~10x slower when every row of the batch
has a different circuit (noise encoding). Use backend='pennylane' for the reference implementation
or to run on hardware.
"""

import math

import torch


def _ry(theta):  # (...,) -> (..., 2, 2), real
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def _rz(theta):  # (...,) -> (..., 2, 2), complex diagonal
    e = torch.exp(-0.5j * theta.to(torch.complex64))
    z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


def _apply_1q(state, U, q, n):
    """state (B, 2^n) complex; U (2,2) or (B,2,2); apply on wire q."""
    B = state.shape[0]
    s = state.reshape(B, 2**q, 2, 2 ** (n - q - 1))
    if U.dim() == 2:
        s = torch.einsum('ij,bljr->blir', U, s)
    else:
        s = torch.einsum('bij,bljr->blir', U, s)
    return s.reshape(B, 2**n)


def _ring_cnot_permutation(n):
    """Basis-index permutation for CNOT(0,1) CNOT(1,2) ... CNOT(n-1,0) applied in that order."""
    idx = torch.arange(2**n)
    perm = idx.clone()  # new_state[i] = old_state[perm[i]]
    for q in range(n):
        c, t = q, (q + 1) % n
        cbit, tbit = 1 << (n - 1 - c), 1 << (n - 1 - t)
        # CNOT is its own inverse: new[i] = old[i ^ tbit if i has control bit else i]
        src = torch.where((idx & cbit) != 0, idx ^ tbit, idx)
        perm = perm[src]
    return perm


class RingCircuit(torch.nn.Module):
    """probs(weights (L,n,2), z (B,n) or None) for: H^n -> [RY(z)] -> L x (RY,RZ each qubit; CNOT ring)."""

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.register_buffer('perm', _ring_cnot_permutation(n), persistent=False)

    def forward(self, weights, z=None, batch_size=1):
        n = self.n
        B = z.shape[0] if z is not None else 1
        # After H on every qubit the state is uniform; then optional per-row RY(z).
        state = torch.full((B, 2**n), 1 / math.sqrt(2**n), dtype=torch.complex64)
        if z is not None:
            for q in range(n):
                state = _apply_1q(state, _ry(z[:, q]).to(torch.complex64), q, n)
        for layer in range(weights.shape[0]):
            for q in range(n):
                U = _rz(weights[layer, q, 1]) @ _ry(weights[layer, q, 0]).to(torch.complex64)
                state = _apply_1q(state, U, q, n)
            state = state[:, self.perm]
        probs = state.real**2 + state.imag**2
        return probs if z is not None else probs.expand(batch_size, -1)
