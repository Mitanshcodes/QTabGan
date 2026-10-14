#!/usr/bin/env bash
# Tuning on a VALIDATION split of the CTAB-GAN+ train set (the real test set is never touched).
# Stage 1: shared settings the paper leaves open (category output handling x loss).
# Stage 2: quantum-specific settings (noise encoding x quantum learning rate), using stage-1 winner.
# Usage: bash scripts/tune.sh stage1|stage2 [extra args for stage2, e.g. "--cat-output softmax --loss bce"]
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-40}
JOBS=${JOBS:-2}
stage=$1; shift || true
extra="${*:-}"

cmds=()
if [[ $stage == stage1 ]]; then
  for cat in linear softmax gumbel; do
    for loss in bce wgan_gp; do
      cmds+=("--cat-output $cat --loss $loss --out runs/tune/s1_${cat}_${loss}")
    done
  done
elif [[ $stage == stage2 ]]; then
  # Stage 1 showed collapse when shot noise is the only randomness, so stage 2 varies where the
  # randomness comes from: noise encoded into the circuit or not, and how many shots (fewer shots =
  # noisier probability vector). A classical gaussian-noise run with the same shared settings shows
  # whether the plain GAN recipe itself works.
  for ne in none angle; do
    for shots in 300 1024 8192; do
      cmds+=("$extra --noise-encoding $ne --n-shots $shots --out runs/tune/s2_${ne}_shots${shots}")
    done
  done
  cmds+=("$extra --source gaussian --out runs/tune/s2_gaussian_control")
fi

printf '%s\n' "${cmds[@]}" | xargs -P "$JOBS" -I{} bash -c \
  'args="{}"; out=$(echo "$args" | sed -E "s/.*--out ([^ ]+).*/\1/"); mkdir -p "$out";
   python3 scripts/run.py --tune --eval quick --epochs '"$EPOCHS"' $args > "$out/log.txt" 2>&1 && echo "finished $out" || echo "FAILED $out"'
