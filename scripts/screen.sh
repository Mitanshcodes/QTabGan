#!/usr/bin/env bash
# Step 2 screening (runs/optuna/PLAN.md). Resumable: runs that already have a result.json are skipped.
# Usage: JOBS=6 bash scripts/screen.sh
# (Each command is passed as one argument: macOS xargs -I truncates replacements at 255 bytes.)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-6}
mkdir -p runs/screen
python3 scripts/screen.py plan > runs/screen/commands.txt
tr '\n' '\0' < runs/screen/commands.txt | xargs -0 -n 1 -P "$JOBS" bash -c \
  'cmd="$1"; out=$(echo "$cmd" | sed -E "s/.*--out ([^ ]+).*/\1/");
   if [ -f "$out/result.json" ]; then echo "skip $out"; exit 0; fi; mkdir -p "$out";
   $cmd > "$out/log.txt" 2>&1 && echo "$(date +%H:%M) finished $out" || echo "$(date +%H:%M) FAILED $out"' _
echo "$(date +%H:%M) screening done"
