#!/usr/bin/env bash
# Training-length experiment (scripts/training_length.py). Waits for scripts/optuna_pipeline.sh to
# finish, then runs 6 jobs (best quantum + best gaussian, seeds 0-2). Resumable: finished runs are skipped.
# Usage: bash scripts/training_length.sh
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f optuna_pipeline.sh > /dev/null; do sleep 120; done
echo "$(date +%H:%M) pipeline finished; training-length experiment"
mkdir -p runs/training_length
python3 scripts/training_length.py plan > runs/training_length/commands.txt
cat runs/training_length/selected.json
tr '\n' '\0' < runs/training_length/commands.txt | xargs -0 -n 1 -P 6 bash -c \
  'cmd="$1"; out=$(echo "$cmd" | sed -E "s/.*--out ([^ ]+).*/\1/");
   if [ -f "$out/result.json" ]; then echo "skip $out"; exit 0; fi; mkdir -p "$out";
   $cmd > "$out/log.txt" 2>&1 && echo "$(date +%H:%M) finished $out" || echo "$(date +%H:%M) FAILED $out"' _
echo "$(date +%H:%M) training-length done"
