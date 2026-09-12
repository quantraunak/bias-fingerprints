#!/usr/bin/env bash
# Remaining 2x2 cells against the constructed controlled-k documents.
# Order: C (MoE/think, ~0.6h) then B (dense/no-think) then A (dense/think).
# Cell D runs first and separately as the pilot.
cd "$(dirname "$0")"
LOG=/tmp/cardinality_docs.log
: > "$LOG"; echo "started $(date)" >> "$LOG"
for CELL in C B A; do
  echo "=== cell $CELL $(date) ===" >> "$LOG"
  for i in $(seq 1 200); do
    python3 scripts/run_cardinality_docs.py --cell "$CELL" >> "$LOG" 2>&1 && break
    echo "--- cell $CELL restart $i $(date) ---" >> "$LOG"
    sleep 5
  done
done
echo "all cells complete $(date)" >> "$LOG"
