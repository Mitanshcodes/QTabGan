"""Final stage of the search: retrain the best config of each arm on the REAL CTAB-GAN+ split
(train 39,073 / test 9,769), 150 epochs, full evaluation, several seeds. Also the most literal
reading of the paper (no circuit input, linear outputs, standard GAN loss).

Usage
  python scripts/search_final.py plan [--seeds 0 1] > runs/search_final/commands.txt
  (run the commands, e.g. with scripts/search_final.sh)
  python scripts/summarize.py final runs/search_final runs/reference/result.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from search import summarize  # noqa: E402

FLAGS = ['mapper_dims', 'disc_dims', 'lr_classical', 'loss', 'batch_size', 'cat_output', 'n_layers',
         'n_shots', 'lr_quantum', 'noise_encoding', 'noise_scale', 'angle_init']
LITERAL = ('--source quantum --noise-encoding none --cat-output linear --loss bce --n-layers 4 '
           '--n-shots 1024')


def flags(cfg, source):
    keep = FLAGS if source == 'quantum' else FLAGS[:6]
    out = []
    for k in keep:
        v = cfg[k]
        if isinstance(v, (list, tuple)):
            v = ','.join(str(x) for x in v)
        out.append(f"--{k.replace('_', '-')} {v}")
    return ' '.join(out) + f' --source {source}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['plan'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    a = ap.parse_args()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        best = summarize()
    base = 'python3 scripts/run.py --epochs 150 --eval full'
    for s in a.seeds:
        for arm in ('quantum', 'gaussian'):
            name, r = best[arm][0]
            print(f"{base} {flags(r['config'], arm)} --seed {s} --out runs/search_final/best_{arm}_s{s}")
        print(f'{base} {LITERAL} --seed {s} --out runs/search_final/paper_literal_s{s}')
