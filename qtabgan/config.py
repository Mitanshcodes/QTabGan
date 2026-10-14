"""Every knob of the QTabGAN reimplementation, with where its value comes from.

QTabGAN (Kumari, Achutha, Sivaraman, arXiv:2602.12704, v1 and v2) does not release code and leaves
most training details unstated. Each field below is tagged:

    paper      - stated explicitly in the paper; we follow it
    ambiguous  - the paper says something, but it can be read more than one way
    unstated   - the paper says nothing; the value is our choice and should be tuned / asked about

Run ``python -m qtabgan.config`` to print this as a markdown table.
"""

from dataclasses import dataclass, field, fields


def _f(default, source, note):
    return field(default=default, metadata={'source': source, 'note': note})


@dataclass
class Config:
    # ---------------- quantum generator core ----------------
    n_qubits: int = _f(8, 'paper', 'Sec. 5.4: "its 8-qubit circuit" -> 256-dim probability vector.')
    n_layers: int = _f(4, 'unstated', 'Paper only says "L layers".')
    n_shots: int = _f(1024, 'ambiguous', 'Paper: "N > 2^n" shots per sample, no value given.')
    init_hadamard: bool = _f(True, 'paper', 'Hadamard on every qubit before the variational layers.')
    rotations: str = _f('ry_rz', 'paper', 'One RY and one RZ per qubit per layer.')
    entanglement: str = _f('ring', 'paper', 'CNOT(i, (i+1) mod n) after each layer.')
    angle_init: str = _f('uniform_0_2pi', 'paper', 'Angles "chosen randomly in [0, 2pi)".')
    noise_encoding: str = _f(
        'none',
        'ambiguous',
        'Sec. 3 describes no circuit input (diversity only from finite shots); Sec. 4.2 says the '
        'generator "samples random noise and encodes it into the quantum circuit" without saying '
        'how. Options: none | angle (RY(z_i), z_i ~ U[0, pi/2) per qubit, after the Hadamards).',
    )
    prob_scale: str = _f(
        'n_outcomes',
        'unstated',
        'How the probability vector is scaled before the mapper. n_outcomes multiplies by 2^n so '
        'entries average 1 (comparable to the one-hot label); none keeps raw probabilities.',
    )
    gradient: str = _f(
        'backprop_straight_through',
        'ambiguous',
        'Paper uses parameter-shift. On a simulator we backprop through the exact circuit '
        'probabilities (same derivative, computed faster) and pass the shot-noisy value forward '
        '(straight-through).',
    )

    backend: str = _f(
        'torch',
        'unstated',
        'Simulator. torch = our batched statevector sim (verified equal to PennyLane in tests, '
        '~10x faster); pennylane = PennyLane default.qubit reference.',
    )

    # ---------------- what feeds the mapper (for controls) ----------------
    source: str = _f(
        'quantum',
        'paper',
        'quantum = QTabGAN. Controls (ours): gaussian = classical N(0, I) noise of the same width; '
        'frozen_quantum = same circuit with angles never trained.',
    )

    # ---------------- classical mapper / discriminator ----------------
    mapper_dims: tuple = _f((256, 256), 'unstated', 'Paper: feed-forward, ReLU, linear output; widths not given.')
    disc_dims: tuple = _f((256, 256), 'unstated', 'Paper: feed-forward, ReLU, linear scalar output; widths not given.')
    cat_output: str = _f(
        'softmax',
        'unstated',
        'Paper: mapper output is linear in R^d; no handling of one-hot blocks stated. '
        'Options: linear (literal) | softmax | gumbel (per categorical block).',
    )
    condition_on: str = _f('target', 'paper', 'One-hot class label concatenated to mapper and discriminator input.')

    # ---------------- training ----------------
    loss: str = _f(
        'bce',
        'ambiguous',
        'Paper gives the standard minimax (log) GAN objective but names no loss. bce = that '
        'objective with the usual non-saturating generator loss. wgan_gp offered as an alternative.',
    )
    epochs: int = _f(150, 'unstated', 'Not stated. 150 matches what CTAB-GAN+ used for Adult.')
    batch_size: int = _f(500, 'unstated', 'Not stated.')
    lr_classical: float = _f(2e-4, 'unstated', 'Adam lr for mapper + discriminator. Optimizer not stated.')
    lr_quantum: float = _f(1e-2, 'unstated', 'Separate Adam lr for circuit angles (our choice).')
    betas: tuple = _f((0.5, 0.9), 'unstated', 'Adam betas.')
    d_steps: int = _f(1, 'unstated', 'Discriminator updates per generator update.')

    # ---------------- preprocessing ----------------
    clip_percentiles: tuple = _f((1, 99), 'paper', 'Numeric features clipped to 1st/99th percentile.')
    numeric_range: str = _f('zscore_then_minmax_0_pi', 'paper', 'z-score, then min-max into [0, pi].')
    input_noise_var: float = _f(1e-5, 'ambiguous', 'Paper: add N(0, 10^-5); read as variance, applied once.')
    categorical_encoding: str = _f('one_hot', 'paper', 'One-hot for categorical features.')

    # ---------------- run ----------------
    seed: int = _f(0, 'unstated', 'Paper does not report seeds or number of runs.')
    device: str = _f('cpu', 'unstated', 'Paper: "standard simulators", none named.')


def provenance_table():
    rows = ['| setting | default | source | note |', '|---|---|---|---|']
    for f in fields(Config):
        rows.append(f"| `{f.name}` | `{f.default}` | {f.metadata['source']} | {f.metadata['note']} |")
    return '\n'.join(rows)


if __name__ == '__main__':
    print(provenance_table())
