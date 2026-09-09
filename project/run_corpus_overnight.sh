#!/usr/bin/env bash
# Overnight corpus extraction under E10 (Qwen 3 30B-A3B).
#
# The process dies under memory pressure roughly every fifteen minutes, and the
# earlier E5 attempt lost ~2.9 days of wall clock to stalls because it was run
# bare. The cache is the source of truth and the queue is recomputed from disk
# at each start, so re-invoking is always safe and never repeats work.
cd "$(dirname "$0")"
LOG=/tmp/corpus_moe.log
: > "$LOG"
echo "started $(date)" >> "$LOG"
for i in $(seq 1 2000); do
  python3 scripts/extract_corpus.py --model qwen3:30b-a3b --no-think --sp500 \
      --since 2012-06-30 --num-predict 4000 >> "$LOG" 2>&1
  status=$?
  [ $status -eq 0 ] && { echo "completed cleanly $(date)" >> "$LOG"; break; }
  echo "--- restart $i after exit $status $(date) ---" >> "$LOG"
  sleep 5
done
