"""Screen candidate spans empirically: keep only what the extractor recovers at k=1.

    python3 scripts/screen_item_bank.py

Three syntactic filters were tried for "which spans transplant into a synthetic
document" -- first person, capitalised head, company-candidate shape -- and all
three were wrong, leaving recall at 0.400 where 0.80 is required. This replaces
the guess with a measurement: build a short single-item document from each
candidate span and keep the span only if the extractor returns its counterparty.

Short documents are used deliberately. Dilution costs 0.25 recall at constant k,
so screening in a 14,000-character haystack would reject spans for being buried
rather than for being untransplantable, which is a different defect.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from src.config import PROCESSED
from src.graph import extract, local_model, resolve
from build_cardinality_docs import item_bank, filler_sentences

OUT = PROCESSED / "cardinality_bank_screened.parquet"
SCREEN_BUDGET = 1500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:30b-a3b")
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bank, filler = item_bank(), filler_sentences()
    print(f"screening {len(bank):,} candidate spans at k=1, {SCREEN_BUDGET} char documents",
          flush=True)

    kept = []
    for n, item in enumerate(bank.itertuples(), 1):
        body, total = [], len(item.evidence) + 1
        pool = list(rng.permutation(len(filler)))
        while total < SCREEN_BUDGET and pool:
            s = filler[pool.pop()]; body.append(s); total += len(s) + 1
        body.insert(int(rng.integers(0, len(body) + 1)), item.evidence.strip())
        paragraphs, current = [], ""
        for sentence in body:
            if len(current) + len(sentence) + 1 > 1200 and current:
                paragraphs.append(current.strip()); current = ""
            current = f"{current} {sentence}".strip()
        if current:
            paragraphs.append(current.strip())

        response = local_model.generate(
            extract.SYSTEM,
            extract.user_prompt("ACME", "ACME", "2020-01-01", paragraphs),
            extract.Extraction.model_json_schema(),
            model=args.model, timeout=600, think=False, num_predict=4000)
        if not response.ok:
            continue
        got = {resolve.normalize(l["counterparty"])
               for l in extract.parse_response(response.text) if l.get("counterparty")}
        if resolve.normalize(item.counterparty) in got:
            kept.append({"counterparty": item.counterparty, "relation": item.relation,
                         "evidence": item.evidence})
        if n % 25 == 0:
            print(f"  {n}/{len(bank)}  kept {len(kept)} ({len(kept)/n:.1%})", flush=True)

    frame = pd.DataFrame(kept)
    frame.to_parquet(OUT)
    print(f"\nkept {len(frame)} of {len(bank)} spans ({len(frame)/len(bank):.1%}) -> {OUT}")
    print("These are spans the extractor demonstrably recovers when nothing is hiding them.")


if __name__ == "__main__":
    main()
