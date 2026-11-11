#!/usr/bin/env bash
# Retrain the search winners + the literal-paper version on the real split. Resumable.
# Usage: bash scripts/search_final.sh [seeds...]   (default: 0 1)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}
mkdir -p runs/search_final
python3 scripts/search_final.py plan --seeds ${@:-0 1} > runs/search_final/commands.txt
xargs -P "$JOBS" -I{} bash -c \
  'cmd="{}"; out=$(echo "$cmd" | sed -E "s/.*--out ([^ ]+).*/\1/");
   if [ -f "$out/result.json" ]; then echo "skip $out"; exit 0; fi; mkdir -p "$out";
   $cmd > "$out/log.txt" 2>&1 && echo "$(date +%H:%M) finished $out" || echo "$(date +%H:%M) FAILED $out"' \
  < runs/search_final/commands.txt
