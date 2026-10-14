#!/usr/bin/env bash
# Final runs on the real CTAB-GAN+ split (train 39,073 / test 9,769), full evaluation.
# Arms (all share the tuned settings: gumbel category output, WGAN-GP loss, 300 shots):
#   quantum         QTabGAN, Sec. 3 reading (no circuit input; shot noise is the randomness)
#   quantum_angle   QTabGAN, Sec. 4.2 reading (per-row noise encoded as RY angles)
#   frozen_quantum  same circuit as `quantum`, angles never trained
#   gaussian        same mapper/discriminator fed classical N(0, I) noise of the same width
#   ctgan           vanilla CTGAN (sdv-dev), same epochs
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-150}
JOBS=${JOBS:-2}
SEEDS=${SEEDS:-"0 1 2"}
shared="--cat-output gumbel --loss wgan_gp --n-shots 300 --epochs $EPOCHS"

cmds=()
for s in $SEEDS; do
  cmds+=("python3 scripts/run.py $shared --noise-encoding angle --seed $s --out runs/final/quantum_angle_s$s")
  cmds+=("python3 scripts/run.py $shared --noise-encoding none --seed $s --out runs/final/quantum_s$s")
  cmds+=("python3 scripts/run.py $shared --noise-encoding none --source frozen_quantum --seed $s --out runs/final/frozen_quantum_s$s")
  cmds+=("python3 scripts/run.py $shared --source gaussian --seed $s --out runs/final/gaussian_s$s")
  cmds+=("python3 scripts/run_ctgan.py --epochs $EPOCHS --seed $s --out runs/final/ctgan_s$s")
done

printf '%s\n' "${cmds[@]}" | xargs -P "$JOBS" -I{} bash -c \
  'cmd="{}"; out=$(echo "$cmd" | sed -E "s/.*--out ([^ ]+).*/\1/"); mkdir -p "$out";
   $cmd > "$out/log.txt" 2>&1 && echo "$(date +%H:%M) finished $out" || echo "$(date +%H:%M) FAILED $out"'
