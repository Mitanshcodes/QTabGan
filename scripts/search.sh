#!/usr/bin/env bash
# Run the random search 2 jobs at a time. Resumable: runs with a result.json are skipped.
# Usage: bash scripts/search.sh [N configs per arm, default 20]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-20}
JOBS=${JOBS:-2}
mkdir -p runs/search
python3 scripts/search.py plan --n "$N" > runs/search/commands.txt
xargs -P "$JOBS" -I{} bash -c \
  'cmd="{}"; out=$(echo "$cmd" | sed -E "s/.*--out ([^ ]+).*/\1/");
   if [ -f "$out/result.json" ]; then echo "skip $out"; exit 0; fi; mkdir -p "$out";
   $cmd > "$out/log.txt" 2>&1 && echo "$(date +%H:%M) finished $out" || echo "$(date +%H:%M) FAILED $out"' \
  < runs/search/commands.txt
