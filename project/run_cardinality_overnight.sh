#!/usr/bin/env bash
# 2x2 cells for the cardinality pre-registration. Cell D is already cached.
# Order: C (fast, ~1.5h) first as a smoke test, then B (4.5h) which completes
# the architecture contrast at no-think against cached D, then A (4h).
cd "$(dirname "$0")"
LOG=/tmp/cardinality_cells.log
: > "$LOG"; echo "started $(date)" >> "$LOG"
for CELL in C B A; do
  echo "=== cell $CELL $(date) ===" >> "$LOG"
  for i in $(seq 1 200); do
    python3 scripts/run_cardinality_cells.py --cell "$CELL" >> "$LOG" 2>&1 && break
    echo "--- cell $CELL restart $i $(date) ---" >> "$LOG"
    sleep 5
  done
done
echo "all cells complete $(date)" >> "$LOG"
