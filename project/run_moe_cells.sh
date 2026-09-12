#!/usr/bin/env bash
cd "$(dirname "$0")"
LOG=/tmp/moe_cells.log; : > "$LOG"
for CELL in D C; do
  echo "=== cell $CELL $(date) ===" >> "$LOG"
  for i in $(seq 1 100); do
    python3 -u scripts/run_cardinality_docs.py --cell "$CELL" >> "$LOG" 2>&1 && break
    sleep 5
  done
done
echo "moe cells complete $(date)" >> "$LOG"
