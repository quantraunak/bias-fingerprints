"""Measure how much of an extraction came from the weights rather than the text.

    python scripts/measure_contamination.py --model qwen3:32b --limit 40

For each counterparty the model already extracted and that resolved to a ticker,
this replaces the counterparty's identifying tokens with invented ones, re-runs
extraction on the swapped passages, and records which of three things happened:

    read     the model names the invented company -- it read the document
    silent   the model names neither -- it needed the real name to see a link
    phantom  the model names the REAL company, which is no longer in the text

`phantom` is the measurement. The model was shown a passage that does not
contain the name it emitted, so the name came from training. Its rate is a
direct estimate of how much a historical dataset built this way is contaminated
by knowledge the market did not have on the filing date.

The two breakdowns are the point. If the phantom rate rises with filing recency
or with how often a counterparty appears across the corpus, contamination has a
*shape*: it concentrates on well-covered firms and recent documents, and it
inflates graph density exactly where a cross-sectional study is most sensitive
to it. A flat rate is a much less interesting -- and much more reassuring --
result, and it is reported the same way.

Resumable. Every response is cached the moment it arrives, keyed by model,
filing and counterparty, so a killed run costs only the call in flight.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config, PROCESSED  # noqa: E402
from src.graph import (  # noqa: E402
    contamination, extract, filings, local_model, passages, resolve,
)

CACHE = PROCESSED / "contamination_cache"
CLAIMS = PROCESSED / "link_claims_qwen.parquet"
SEED = 7


def cached_swap_extract(model: str, accession: str, counterparty: str,
                        ticker: str, filed: str, swapped: list[str],
                        think: bool | None) -> list[str] | None:
    """Extraction on swapped passages, cached. Returns counterparty names."""
    suffix = "" if think is None else ("_nothink" if think is False else "_think")
    key = resolve.normalize(counterparty).replace(" ", "_")[:40] or "blank"
    path = CACHE / (model.replace(":", "_") + suffix) / accession / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["counterparties"]

    response = local_model.generate(
        extract.SYSTEM,
        extract.user_prompt(ticker, ticker, filed, swapped),
        extract.Extraction.model_json_schema(),
        model=model, timeout=1800, think=think,
    )
    if not response.ok:
        return None
    # Validation runs against the SWAPPED passages: a quote grounded in text the
    # model was never shown is exactly the failure being measured, so it must
    # not be silently repaired by validating against the original.
    kept, _ = extract.validate(extract.parse_response(response.text), swapped)
    names = [link["counterparty"] for link in kept]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"counterparties": names, "seconds": response.seconds}))
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="qwen3:32b")
    parser.add_argument("--no-think", dest="think", action="store_const",
                        const=False, default=None)
    parser.add_argument("--limit", type=int, default=40,
                        help="counterparties to test; each costs one model call")
    parser.add_argument("--claims", default=str(CLAIMS))
    args = parser.parse_args()

    config = Config.load(args.config)
    lookup = resolve.build_lookup(resolve.load_reference(config.data.sec_user_agent))
    index = filings.load_index().set_index("accession")
    claims = pd.read_parquet(args.claims)

    resolved = resolve.resolve_frame(claims, resolve.load_reference(config.data.sec_user_agent))
    resolved = resolved[resolved.counterparty_ticker.notna()]

    # Corpus mention count is the prominence proxy: how often this counterparty
    # is named across all filings. A firm named in many filings is one the model
    # has seen discussed often, which is the axis contamination should follow.
    prominence = resolved.counterparty_ticker.value_counts().to_dict()

    rng = random.Random(SEED)
    pairs = resolved.drop_duplicates(["accession", "counterparty_ticker"])
    pairs = pairs.sample(frac=1.0, random_state=SEED).head(args.limit)

    rows = []
    for i, row in enumerate(pairs.itertuples(), 1):
        if row.accession not in index.index:
            continue
        meta = index.loc[row.accession]
        text = Path(meta.path).read_text(encoding="utf-8", errors="ignore")
        selected = passages.select(text)

        alias = contamination.mint_alias(row.counterparty, lookup, rng)
        if alias is None:
            continue
        swapped, replaced = contamination.swap(selected, row.counterparty, alias, lookup)
        if replaced == 0:
            # The name never appears verbatim in the selected passages, so there
            # is nothing to ablate and the pair says nothing either way.
            continue

        names = cached_swap_extract(
            args.model, row.accession, row.counterparty, meta.ticker,
            str(pd.Timestamp(meta.filed).date()), swapped, args.think,
        )
        if names is None:
            print(f"  {meta.ticker} / {row.counterparty}: extraction failed", flush=True)
            continue

        verdict = contamination.classify(names, row.counterparty, alias, lookup)
        rows.append({
            "accession": row.accession, "ticker": meta.ticker,
            "counterparty": row.counterparty, "counterparty_ticker": row.counterparty_ticker,
            "alias": alias, "replaced": replaced, "verdict": verdict,
            "year": pd.Timestamp(meta.filed).year,
            "mentions": prominence.get(row.counterparty_ticker, 1),
        })
        print(f"  [{i}/{len(pairs)}] {meta.ticker:6} {row.counterparty[:34]:34} "
              f"-> {verdict}", flush=True)

    if not rows:
        print("no testable pairs")
        return

    frame = pd.DataFrame(rows)
    out = PROCESSED / "contamination.parquet"
    frame.to_parquet(out)

    counts = frame.verdict.value_counts()
    n = len(frame)
    print(f"\n{n} counterparties tested on {frame.accession.nunique()} filings")
    for verdict in ("read", "silent", "phantom"):
        print(f"  {verdict:8} {counts.get(verdict, 0):4}  {counts.get(verdict, 0)/n:6.1%}")

    frame["phantom"] = frame.verdict == "phantom"
    if frame.year.nunique() > 1:
        era = frame.assign(era=pd.cut(frame.year, [0, 2015, 2020, 2100],
                                      labels=["<=2015", "2016-2020", "2021+"]))
        print("\nphantom rate by filing era")
        print(era.groupby("era", observed=True).phantom.agg(["mean", "size"]).to_string())
    if frame.mentions.nunique() > 1:
        band = frame.assign(prom=pd.qcut(frame.mentions, min(3, frame.mentions.nunique()),
                                         duplicates="drop"))
        print("\nphantom rate by counterparty prominence (corpus mentions)")
        print(band.groupby("prom", observed=True).phantom.agg(["mean", "size"]).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
