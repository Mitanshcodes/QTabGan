#!/usr/bin/env bash
# Runs behind figures/teammate_style/ (scripts/make_teammate_figures.py). Resumable: finished runs skipped.
#  - best QTabGAN / best Gaussian step-5 configs (runs/training_length/selected.json), validation split,
#    3 seeds, 150 epochs, diagnostics every 10 epochs (scripts/run.py --eval-every 10)
#  - CTGAN, literal-paper QTabGAN and first-pass QTabGAN on the test split, seeds 0-4, saving fake.csv
#    (the earlier 2-seed runs in runs/final and runs/search_final did not keep their synthetic tables)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-9}
mkdir -p runs/teammate_style
python3 - > runs/teammate_style/commands.txt <<'PY'
import json, sys
sys.path.insert(0, 'scripts')
from confirm import flags
sel = json.load(open('runs/training_length/selected.json'))
for s in range(3):
    for arm in ('quantum', 'gaussian'):
        print(f'python3 scripts/run.py --tune --epochs 150 --eval none --eval-every 10 '
              f'{flags(arm, sel[arm]["params"])} --seed {s} --out runs/teammate_style/diag_{arm}_s{s}')
for s in range(5):
    print(f'python3 scripts/run_ctgan.py --epochs 150 --seed {s} --out runs/teammate_style/ctgan_s{s}')
    print('python3 scripts/run.py --epochs 150 --eval full --source quantum --noise-encoding none '
          '--cat-output linear --loss bce --n-layers 4 --n-shots 1024 '
          f'--seed {s} --out runs/teammate_style/paper_literal_s{s}')
    print('python3 scripts/run.py --cat-output gumbel --loss wgan_gp --n-shots 300 --epochs 150 '
          f'--noise-encoding none --seed {s} --out runs/teammate_style/first_pass_s{s}')
PY
tr '\n' '\0' < runs/teammate_style/commands.txt | xargs -0 -n 1 -P "$JOBS" bash -c \
  'cmd="$1"; out=$(echo "$cmd" | sed -E "s/.*--out ([^ ]+).*/\1/");
   if [ -f "$out/fake.csv" ] && [ -f "$out/history.json" -o -f "$out/result.json" ]; then echo "skip $out"; exit 0; fi
   mkdir -p "$out"; $cmd > "$out/log.txt" 2>&1 && echo "$(date +%H:%M) finished $out" || echo "$(date +%H:%M) FAILED $out"' _
echo "$(date +%H:%M) teammate-style runs done"
