#!/usr/bin/env bash
cd "$(dirname "$0")"
LOG=/tmp/decoding14b.log; : > "$LOG"
for ARM in constrained unconstrained; do
  echo "=== $ARM $(date) ===" >> "$LOG"
  for i in $(seq 1 60); do
    python3 -u scripts/run_decoding_arm.py --arm "$ARM" --model qwen3:14b \
      --levels 1,4,16 --max-per-level 25 >> "$LOG" 2>&1 && break
    sleep 10
  done
done
echo "both arms complete $(date)" >> "$LOG"
