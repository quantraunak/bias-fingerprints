"""Extract the corpus with a local model, one filing at a time, resumably.

The extraction run is measured in days and the machine kills it roughly every
fifteen minutes under memory pressure, so the design assumption is that the
process dies constantly and that this must cost nothing. Every response is
written to the cache the moment it arrives, and the queue is recomputed from
what is already on disk at each start, so re-invoking is always safe and never
repeats work.

    python scripts/extract_corpus.py --model qwen3:32b --no-think --sp500
    while ! python scripts/extract_corpus.py --model qwen3:32b --no-think --sp500; do :; done

Claims are rebuilt from the cache rather than appended as it goes: the cache is
the source of truth, so a partially written parquet from a killed run cannot
corrupt the output.
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

QUEUE = PROCESSED / "extract_queue.parquet"
CACHE = PROCESSED / "extract_cache"
CLAIMS = PROCESSED / "link_claims_qwen.parquet"


def cache_dir(model: str, think: bool | None) -> Path:
    suffix = "" if think is None else ("_nothink" if think is False else "_think")
    return CACHE / (model.replace(":", "_") + suffix)


def rebuild_claims(directory: Path, index: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the claims table from cached responses.

    Validation runs here rather than at extraction time, so a change to the
    grounding rules is a re-parse of the cache rather than a re-run of the model.
    """
    records = []
    for path in directory.glob("*.json"):
        accession = path.stem
        if accession not in index.index:
            continue
        row = index.loc[accession]
        selected = passages.select(Path(row.path).read_text(encoding="utf-8", errors="ignore"))
        kept, _ = extract.validate(extract.parse_response(json.loads(path.read_text())["text"]), selected)
        for link in kept:
            records.append({
                "ticker": row.ticker, "accession": accession, "filed": row.filed,
                "counterparty": link["counterparty"], "relation": link["relation"],
                "revenue_pct": link.get("revenue_pct"), "confidence": link.get("confidence"),
                "evidence": link["evidence"],
            })
    return extract.to_frame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:32b")
    parser.add_argument("--no-think", dest="think", action="store_const", const=False, default=None)
    parser.add_argument("--sp500", action="store_true", help="restrict to S&P 500 issuers")
    parser.add_argument("--limit", type=int, help="stop after this many filings this invocation")
    parser.add_argument("--rebuild-only", action="store_true")
    args = parser.parse_args()

    index = filings.load_index().set_index("accession")
    directory = cache_dir(args.model, args.think)
    directory.mkdir(parents=True, exist_ok=True)

    if args.rebuild_only:
        claims = rebuild_claims(directory, index)
        claims.to_parquet(CLAIMS)
        print(f"{len(claims):,} claims from {claims.accession.nunique():,} filings -> {CLAIMS}")
        return

    queue = pd.read_parquet(QUEUE)
    queue = queue[queue.passes]
    if args.sp500:
        queue = queue[queue.sp500]
    done = {p.stem for p in directory.glob("*.json")}
    todo = queue[~queue.accession.isin(done)]

    print(f"queue {len(queue):,}   done {len(done):,}   remaining {len(todo):,}", flush=True)
    if todo.empty:
        claims = rebuild_claims(directory, index)
        claims.to_parquet(CLAIMS)
        print(f"complete. {len(claims):,} claims -> {CLAIMS}")
        return

    processed, began = 0, time.time()
    for row in todo.itertuples():
        if args.limit and processed >= args.limit:
            break
        source = index.loc[row.accession]
        selected = passages.select(Path(source.path).read_text(encoding="utf-8", errors="ignore"))
        if not selected:
            continue
        response = local_model.generate(
            extract.SYSTEM,
            extract.user_prompt(source.ticker, source.ticker,
                                str(pd.Timestamp(source.filed).date()), selected),
            extract.Extraction.model_json_schema(),
            model=args.model, timeout=3600, think=args.think,
        )
        if not response.ok:
            print(f"  {source.ticker} {row.accession}: {response.error}", flush=True)
            continue
        (directory / f"{row.accession}.json").write_text(
            json.dumps({"text": response.text, "seconds": response.seconds})
        )
        processed += 1
        rate = (time.time() - began) / processed
        left = (len(todo) - processed) * rate / 3600
        print(f"  [{len(done)+processed:>5}/{len(queue):,}] {source.ticker:6} "
              f"{response.seconds:>4.0f}s   ~{left:.1f}h remaining", flush=True)

    # Exit non-zero while work remains, so a shell `until` loop keeps restarting.
    sys.exit(0 if processed >= len(todo) else 1)


if __name__ == "__main__":
    main()
