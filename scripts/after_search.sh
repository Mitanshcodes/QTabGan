#!/usr/bin/env bash
# Wait for the search (given its PID) to finish, then summarize it and run the final stage.
# Usage: nohup bash scripts/after_search.sh <search_pid> > runs/search_final_progress.log 2>&1 &
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
python3 scripts/search.py summarize > runs/search/summary.md 2>&1
echo "$(date +%H:%M) search done; starting final stage"
bash scripts/search_final.sh 0 1
echo "$(date +%H:%M) final stage done"
