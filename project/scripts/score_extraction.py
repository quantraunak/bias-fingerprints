"""Measure extraction quality against the hand-annotated sample.

    python -m scripts.score_extraction

Extracts the annotated filings if they have not come up in the main queue yet,
then scores precision and recall. The pre-registration commits to reporting
extraction error as a measured quantity rather than assuming a local 8B model is
good enough, and this is where that number comes from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.graph import evaluate, extract, filings, local_model, passages, resolve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=local_model.DEFAULT_MODEL)
    args = parser.parse_args()

    config = Config.load(args.config)
    gold = evaluate.load_gold()
    index = filings.load_index().set_index("accession")

    rows = []
    for accession in gold:
        if accession not in index.index:
            print(f"  {accession}: not in filing index, skipped")
            continue
        row = index.loc[accession]
        text = Path(row.path).read_text(encoding="utf-8", errors="ignore")
        selected = passages.select(text)
        response = local_model.generate(
            extract.SYSTEM,
            extract.user_prompt(row.ticker, row.ticker, str(pd.Timestamp(row.filed).date()), selected),
            extract.Extraction.model_json_schema(),
            model=args.model,
        )
        if not response.ok:
            print(f"  {row.ticker}: extraction failed ({response.error})")
            continue
        raw = extract.parse_response(response.text)
        kept, tally = extract.validate(raw, selected)
        for link in kept:
            rows.append({"accession": accession, "ticker": row.ticker,
                         "counterparty": link["counterparty"], "relation": link["relation"]})
        print(f"  {row.ticker:6} raw={len(raw):2} kept={tally['kept']:2} "
              f"gold={len(gold[accession]):2}", flush=True)

    predicted = pd.DataFrame(rows)
    if predicted.empty:
        print("\nno predictions produced")
        return

    reference = resolve.load_reference(config.data.sec_user_agent)
    lookup = resolve.build_lookup(reference)
    frame, agg = evaluate.score(predicted, gold, lookup)

    print(f"\nprecision {agg['precision']}   recall {agg['recall']}   f1 {agg['f1']}")
    print(f"  tp={agg['true_positive']}  fp={agg['false_positive']}  fn={agg['false_negative']}")
    print("\nper filing:")
    print(frame[["ticker", "predicted", "gold", "true_positive", "false_positive", "false_negative"]]
          .to_string(index=False))
    misses = [m for r in frame["missed"] for m in r]
    spurious = [s for r in frame["spurious"] for s in r]
    if misses:
        print(f"\nmissed ({len(misses)}): {', '.join(misses[:12])}")
    if spurious:
        print(f"spurious ({len(spurious)}): {', '.join(spurious[:12])}")


if __name__ == "__main__":
    main()
