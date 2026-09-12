"""Run one 2x2 cell against the constructed controlled-k documents.

    python3 scripts/run_cardinality_docs.py --cell D

Caches per (cell, doc_id) so a kill costs only the document in flight.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED
from src.graph import extract, local_model

CELLS = {"A": ("qwen3:32b", None), "B": ("qwen3:32b", False),
         "C": ("qwen3:30b-a3b", None), "D": ("qwen3:30b-a3b", False)}
DOCS = PROCESSED / "cardinality_docs.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, choices=sorted(CELLS))
    p.add_argument("--num-predict", type=int, default=8000)
    p.add_argument("--levels", type=str, default=None,
                   help="comma-separated k levels to run, e.g. 1,4,16. The dense "
                        "cells take 173s per k=16 document, so the full sweep is "
                        "25h a cell; 1,4,16 still supports the rank test and is "
                        "exactly the k<=3 vs k>=16 contrast the threshold is set on")
    args = p.parse_args()

    model, think = CELLS[args.cell]
    out = PROCESSED / "cardinality_runs" / args.cell
    out.mkdir(parents=True, exist_ok=True)
    docs = json.loads(DOCS.read_text())
    if args.levels:
        want = {int(x) for x in args.levels.split(",")}
        docs = [d for d in docs if d["k_intended"] in want]
    todo = [d for d in docs if not (out / f"{d['doc_id']}.json").exists()]
    print(f"cell {args.cell}  {model} think={think}  {len(todo)}/{len(docs)} to run", flush=True)

    began, done = time.time(), 0
    for doc in todo:
        r = local_model.generate(
            extract.SYSTEM,
            extract.user_prompt("ACME", "ACME", "2020-01-01", doc["paragraphs"]),
            extract.Extraction.model_json_schema(),
            model=model, timeout=3600, think=think, num_predict=args.num_predict)
        if not r.ok:
            print(f"  {doc['doc_id']}: {r.error}", flush=True); continue
        (out / f"{doc['doc_id']}.json").write_text(json.dumps(
            {"text": r.text, "seconds": r.seconds, "truncated": bool(r.truncated),
             "eval_count": r.eval_count}))
        done += 1
        left = (len(todo) - done) * (time.time() - began) / done / 3600
        print(f"  [{done:>3}/{len(todo)}] {doc['doc_id']} k={doc['k_actual']:>2} "
              f"{r.seconds:>5.0f}s {r.eval_count:>5}tok{' TRUNC' if r.truncated else ''}  "
              f"~{left:.1f}h left", flush=True)
    sys.exit(0 if done >= len(todo) else 1)


if __name__ == "__main__":
    main()
