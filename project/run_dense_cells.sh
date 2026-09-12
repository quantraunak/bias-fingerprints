#!/usr/bin/env bash
# Dense cells at k in {1,4,16}. Waits for cell C to finish first.
cd "$(dirname "$0")"
LOG=/tmp/dense_cells.log; : > "$LOG"
while pgrep -f "run_cardinality_docs.py --cell C" >/dev/null; do sleep 30; done
echo "cell C done, starting dense $(date)" >> "$LOG"
for CELL in A B; do
  echo "=== cell $CELL $(date) ===" >> "$LOG"
  for i in $(seq 1 100); do
    python3 -u scripts/run_cardinality_docs.py --cell "$CELL" --levels 1,4,16 >> "$LOG" 2>&1 && break
    sleep 10
  done
done
echo "dense complete $(date)" >> "$LOG"
