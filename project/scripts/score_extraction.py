"""Measure extraction quality against the hand-annotated sample.

    python -m scripts.score_extraction

Extracts the annotated filings if they have not come up in the main queue yet,
then scores precision and recall. The pre-registration commits to reporting
extraction error as a measured quantity rather than assuming a local 8B model is
good enough, and this is where that number comes from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config, PROCESSED  # noqa: E402
from src.graph import evaluate, extract, filings, local_model, passages, resolve  # noqa: E402

# A 32B model needs ~150s per filing and 22GB resident, which on a 36GB machine
# is close enough to the ceiling that the OS kills the run partway through.
# Caching each response keyed by (model, filing) makes a kill cost only the
# filing in flight, so a long scoring run is resumed rather than restarted.
CACHE = PROCESSED / "extract_cache"
TIMINGS: list[float] = []


def cached_generate(model: str, accession: str, system: str, prompt: str, schema: dict):
    path = CACHE / model.replace(":", "_") / f"{accession}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        TIMINGS.append(payload.get("seconds", 0.0))
        return payload["text"], True
    response = local_model.generate(system, prompt, schema, model=model, timeout=1800)
    if not response.ok:
        return None, False
    TIMINGS.append(response.seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": response.text, "seconds": response.seconds}))
    return response.text, False


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
        reply, cache_hit = cached_generate(
            args.model, accession, extract.SYSTEM,
            extract.user_prompt(row.ticker, row.ticker, str(pd.Timestamp(row.filed).date()), selected),
            extract.Extraction.model_json_schema(),
        )
        if reply is None:
            print(f"  {row.ticker}: extraction failed")
            continue
        raw = extract.parse_response(reply)
        kept, tally = extract.validate(raw, selected)
        for link in kept:
            rows.append({"accession": accession, "ticker": row.ticker,
                         "counterparty": link["counterparty"], "relation": link["relation"]})
        print(f"  {row.ticker:6} raw={len(raw):2} kept={tally['kept']:2} "
              f"gold={len(gold[accession]):2}{'  (cached)' if cache_hit else ''}", flush=True)

    predicted = pd.DataFrame(rows)
    if predicted.empty:
        print("\nno predictions produced")
        return

    reference = resolve.load_reference(config.data.sec_user_agent)
    lookup = resolve.build_lookup(reference)
    frame, agg = evaluate.score(predicted, gold, lookup)
    # Throughput matters as much as quality: the corpus is 4,895 filings, so a
    # model that scores higher and runs six times slower can still be the wrong
    # choice. Timings come from the cache, so a resumed run still reports them.
    if TIMINGS:
        mean = sum(TIMINGS) / len(TIMINGS)
        print(f"throughput  {mean:.0f}s per filing"
              f"   corpus of 4,895 filings ~= {mean * 4895 / 3600:.0f}h")

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
