#!/usr/bin/env bash
# Steps 3-5 of runs/optuna/PLAN.md, end to end. Resumable: rerun the same command after a crash;
# finished studies/runs are kept (studies stop at 40 finished trials, runs with result.json are skipped).
# Usage: JOBS=6 bash scripts/optuna_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-6}
for objective in score paper; do
  for arm in quantum gaussian; do
    echo "$(date +%H:%M) study ${objective}_${arm}"
    python3 scripts/optuna_search.py run --arm $arm --objective $objective --workers "$JOBS" --trials 40 \
      > "runs/optuna/${objective}_${arm}.log" 2>&1
  done
  python3 scripts/optuna_search.py report --objective $objective --top 10 > "runs/optuna/report_${objective}.md" 2>&1
done
echo "$(date +%H:%M) studies done; step 5"
python3 scripts/confirm.py plan > runs/confirm/commands.txt
tr '\n' '\0' < runs/confirm/commands.txt | xargs -0 -n 1 -P "$JOBS" bash -c \
  'cmd="$1"; out=$(echo "$cmd" | sed -E "s/.*--out ([^ ]+).*/\1/");
   if [ -f "$out/result.json" ]; then echo "skip $out"; exit 0; fi; mkdir -p "$out";
   $cmd > "$out/log.txt" 2>&1 && echo "$(date +%H:%M) finished $out" || echo "$(date +%H:%M) FAILED $out"' _
python3 scripts/summarize.py final runs/confirm runs/reference/result.json > runs/confirm/summary.md 2>&1 || true
echo "$(date +%H:%M) pipeline done"
