#!/usr/bin/env bash
# Decide whether a candidate extractor replaces Qwen 3 32B (E5), on the benchmark.
#
# One decision, then commit. Run this only when the corpus run is stopped --
# a 20GB and an 18GB model will not co-reside on a 36GB machine.
#
#   scripts/validate_model.sh qwen3:30b-a3b
#
# Passing means F1 >= 0.82 against E5's 0.857. Anything less and the fallback is
# 32B with --num-predict 4000, which is ~4.8 days and, per docs/SPECIFICATIONS.md,
# not worth running the full corpus for.
set -euo pipefail
CANDIDATE="${1:-qwen3:30b-a3b}"
cd "$(dirname "$0")/.."

if pgrep -f extract_corpus.py > /dev/null; then
  echo "extract_corpus.py is still running. Stop it first -- the two models will not fit."
  exit 1
fi

echo "==> pulling $CANDIDATE"
ollama pull "$CANDIDATE"

echo
echo "==> scoring $CANDIDATE against docs/gold_links.json (10 filings)"
python3 scripts/score_extraction.py --model "$CANDIDATE" | tee /tmp/candidate_score.txt

echo
echo "==> incumbent, from cache (no GPU)"
python3 scripts/score_extraction.py --model qwen3:32b | grep -E "precision|throughput"

echo
echo "Decide on F1 and seconds/filing above."
echo "  F1 >= 0.82  -> switch: re-run the 98 completed filings, then the corpus"
echo "  F1 <  0.82  -> keep 32B, add --num-predict 4000, and do NOT run the full corpus"
