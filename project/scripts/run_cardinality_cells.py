"""Generate and cache the 2x2 cells for the cardinality study.

    python3 scripts/run_cardinality_cells.py --cell B

Cells are (architecture x reasoning mode) from HYPOTHESIS_CARDINALITY.md. Each
response is cached under (model, think) exactly as the scorer expects, so this
script only fills cache; nothing is scored here and no gold is touched.

Resumable: a cached filing is skipped, so a kill costs only the filing in flight.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import PROCESSED  # noqa: E402
from src.graph import extract, filings, local_model, passages  # noqa: E402

CELLS = {
    "A": ("qwen3:32b", None),        # dense, thinking
    "B": ("qwen3:32b", False),       # dense, no thinking
    "C": ("qwen3:30b-a3b", None),    # MoE, thinking
    "D": ("qwen3:30b-a3b", False),   # MoE, no thinking  (already cached)
}
EVAL_SET = Path(__file__).resolve().parents[1] / "reports" / "cardinality_eval_set.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=sorted(CELLS))
    parser.add_argument("--num-predict", type=int, default=8000,
                        help="ceiling per filing. Higher than the corpus run's 4000: the "
                             "thinking cells spend tokens before emitting, and a cell that "
                             "truncates at high k would confound the axis under test")
    args = parser.parse_args()

    model, think = CELLS[args.cell]
    suffix = "" if think is None else ("_nothink" if think is False else "_think")
    directory = PROCESSED / "extract_cache" / (model.replace(":", "_") + suffix)
    directory.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(EVAL_SET, dtype={"accession": str})
    index = filings.load_index().set_index("accession")
    todo = [a for a in selection.accession if not (directory / f"{a}.json").exists()]
    print(f"cell {args.cell}  {model} think={think}  "
          f"{len(selection)} selected, {len(todo)} to run", flush=True)

    began, done = time.time(), 0
    for accession in todo:
        if accession not in index.index:
            print(f"  {accession}: not in filing index, skipped", flush=True)
            continue
        row = index.loc[accession]
        selected = passages.select(Path(row.path).read_text(encoding="utf-8", errors="ignore"))
        if not selected:
            continue
        response = local_model.generate(
            extract.SYSTEM,
            extract.user_prompt(row.ticker, row.ticker,
                                str(pd.Timestamp(row.filed).date()), selected),
            extract.Extraction.model_json_schema(),
            model=model, timeout=3600, think=think, num_predict=args.num_predict,
        )
        if not response.ok:
            print(f"  {row.ticker}: {response.error}", flush=True)
            continue
        if response.truncated:
            (directory / f"{accession}.truncated").write_text(str(response.eval_count))
            print(f"  {row.ticker:6} TRUNCATED at {response.eval_count} tokens", flush=True)
            continue
        (directory / f"{accession}.json").write_text(
            json.dumps({"text": response.text, "seconds": response.seconds})
        )
        done += 1
        left = (len(todo) - done) * (time.time() - began) / done / 3600
        print(f"  [{done:>3}/{len(todo)}] {row.ticker:6} {response.seconds:>5.0f}s "
              f"{response.eval_count:>5}tok  ~{left:.1f}h left", flush=True)

    sys.exit(0 if done >= len(todo) else 1)


if __name__ == "__main__":
    main()
