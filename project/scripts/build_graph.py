"""Build the economic-link graph from filing text, one stage at a time.

    python -m scripts.build_graph --stage filings --limit 150
    python -m scripts.build_graph --stage extract
    python -m scripts.build_graph --stage resolve

Stages are idempotent and cache to `data/raw/filings` and `data/processed`, so
an interrupted run resumes rather than restarting. `filings` is network-bound
against SEC rate limits; `extract` is compute-bound on the local model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import PROCESSED, Config  # noqa: E402
from src.data import universe  # noqa: E402
from src.graph import extract, filings, local_model, passages  # noqa: E402
from src.graph import resolve, supplier_universe  # noqa: E402

CLAIMS_PATH = PROCESSED / "link_claims.parquet"
EXTRACT_LOG = PROCESSED / "extract_log.parquet"


def pilot_universe(config: Config, limit: int | None) -> list[str]:
    """Names with the most days in the index over the window.

    Membership duration is a proxy for price coverage: a name in the index for
    fifteen years has a full return series to propagate along its edges, while
    one that appeared for six months contributes an edge that can barely be
    tested. For a pilot whose purpose is to measure edge yield, that is the
    right way to spend a limited number of filings.
    """
    spells = universe.load_spells()
    window = (pd.Timestamp(config.data.start), pd.Timestamp(config.data.end))

    days: dict[str, float] = {}
    for row in spells.itertuples():
        start = max(pd.Timestamp(row.start), window[0])
        end = min(pd.Timestamp(row.end) if pd.notna(row.end) else window[1], window[1])
        if end > start:
            days[row.ticker] = days.get(row.ticker, 0.0) + (end - start).days

    ranked = sorted(days, key=lambda t: -days[t])
    return ranked[:limit] if limit else ranked


def resolve_universe(config: Config, name: str, limit: int | None) -> list[str]:
    if name == "supplier":
        frame = supplier_universe.build_candidates(config.data.sec_user_agent)
        tickers = frame["ticker"].tolist()
        return tickers[:limit] if limit else tickers
    return pilot_universe(config, limit)


def stage_filings(config: Config, limit: int | None, start: str, end: str,
                  universe_name: str = "index") -> None:
    tickers = resolve_universe(config, universe_name, limit)
    print(f"universe: {len(tickers)} names, 10-Ks filed {start} to {end}")

    began = time.time()
    index = filings.build_index(tickers, config.data.sec_user_agent, start, end)
    if index.empty:
        raise RuntimeError("No filings retrieved; check the SEC user agent and network.")

    print(f"\n{len(index):,} filings from {index['ticker'].nunique()} issuers in {time.time()-began:.0f}s")
    print(f"  per issuer: median {index.groupby('ticker').size().median():.0f}")
    print(f"  filed span: {index['filed'].min().date()} to {index['filed'].max().date()}")
    missing = index.attrs.get("no_cik", [])
    if missing:
        print(f"  no CIK mapping ({len(missing)}): {', '.join(missing[:12])}")


def _prioritise(todo: pd.DataFrame, lookup) -> pd.DataFrame:
    scores = []
    for row in todo.itertuples():
        try:
            selected = passages.select(Path(row.path).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            scores.append(-1)
            continue
        if not selected:
            scores.append(-1)
            continue
        _send, hits = passages.prescreen(selected, lookup, resolve.resolve_one, row.ticker)
        scores.append(hits)
    ordered = todo.assign(_priority=scores).sort_values("_priority", ascending=False)
    positive = int((ordered["_priority"] > 0).sum())
    print(f"  prioritised: {positive:,} filings with a resolvable candidate first", flush=True)
    return ordered.drop(columns="_priority")


def _extract_one(row, model: str, lookup, source_lookup) -> tuple[list[dict], dict]:
    """Extract one filing. Pure enough to run on a worker thread."""
    text = Path(row.path).read_text(encoding="utf-8", errors="ignore")
    selected = passages.select(text)
    if not selected:
        return [], {"accession": row.accession, "ticker": row.ticker, "status": "no_passages",
                    "n_raw": 0, "n_kept": 0, "seconds": 0.0}

    send, _candidates = passages.prescreen(selected, lookup, source_lookup, row.ticker)
    if not send:
        return [], {"accession": row.accession, "ticker": row.ticker, "status": "prescreened",
                    "n_raw": 0, "n_kept": 0, "seconds": 0.0}

    filed = pd.Timestamp(row.filed).date().isoformat()
    response = local_model.generate(
        extract.SYSTEM,
        extract.user_prompt(row.ticker, row.ticker, filed, selected),
        extract.Extraction.model_json_schema(),
        model=model,
    )
    if not response.ok:
        return [], {"accession": row.accession, "ticker": row.ticker, "status": "error",
                    "n_raw": 0, "n_kept": 0, "seconds": 0.0, "error": response.error}

    raw = extract.parse_response(response.text)
    kept, tally = extract.validate(raw, selected)
    records = [{
        "ticker": row.ticker, "accession": row.accession, "filed": row.filed,
        "counterparty": link["counterparty"], "relation": link["relation"],
        "revenue_pct": link.get("revenue_pct"), "confidence": link["confidence"],
        "evidence": link["evidence"],
    } for link in kept]
    log = {"accession": row.accession, "ticker": row.ticker, "status": "ok",
           "n_raw": len(raw), "n_kept": tally["kept"], "seconds": response.seconds,
           **{f"rej_{k}": v for k, v in tally.items() if k != "kept"}}
    return records, log


def stage_extract(config: Config, limit: int | None, model: str, workers: int = 4) -> None:
    if not local_model.available():
        raise RuntimeError("Ollama is not responding on :11434. Start it with `ollama serve`.")

    index = filings.load_index()
    done: set[str] = set()
    if CLAIMS_PATH.exists():
        done = set(pd.read_parquet(CLAIMS_PATH)["accession"].unique())
    if EXTRACT_LOG.exists():
        done |= set(pd.read_parquet(EXTRACT_LOG)["accession"].unique())

    todo = index[~index["accession"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"{len(todo):,} filings to extract ({len(done):,} done), model={model}, workers={workers}",
          flush=True)

    reference = resolve.load_reference(config.data.sec_user_agent)
    lookup = resolve.build_lookup(reference)

    # Order by expected yield. Extraction runs at ~14s a filing whatever else is
    # tuned -- parallel workers gave nothing, because one 8B stream already
    # saturates the GPU -- so the run is long regardless. Sorting by how many
    # resolvable counterparties the prescreen can see means the edges arrive in
    # the first hours rather than uniformly across thirteen, and a run stopped
    # early is still a usable graph instead of an arbitrary slice.
    todo = _prioritise(todo, lookup)

    records, log, began, n = [], [], time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_extract_one, row, model, lookup, resolve.resolve_one): row.accession
            for row in todo.itertuples()
        }
        for future in as_completed(futures):
            n += 1
            try:
                got_records, got_log = future.result()
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                log.append({"accession": futures[future], "ticker": None, "status": "exception",
                            "n_raw": 0, "n_kept": 0, "seconds": 0.0, "error": repr(exc)})
                continue
            records.extend(got_records)
            log.append(got_log)

            if n % 50 == 0 or n == len(todo):
                rate = (time.time() - began) / n
                print(f"  {n}/{len(todo)}  {len(records)} new claims  {rate:.2f}s/filing  "
                      f"eta {(len(todo)-n)*rate/60:.0f}m", flush=True)
                _checkpoint(records, log)
                records, log = [], []

    _checkpoint(records, log)
    _report()


def _checkpoint(records: list[dict], log: list[dict]) -> None:
    """Append to disk so a long run survives interruption."""
    if records:
        frame = extract.to_frame(records)
        if CLAIMS_PATH.exists():
            frame = pd.concat([pd.read_parquet(CLAIMS_PATH), frame], ignore_index=True)
        frame.to_parquet(CLAIMS_PATH)
    if log:
        frame = pd.DataFrame(log)
        if EXTRACT_LOG.exists():
            frame = pd.concat([pd.read_parquet(EXTRACT_LOG), frame], ignore_index=True)
        frame.to_parquet(EXTRACT_LOG)


def _report() -> None:
    if not CLAIMS_PATH.exists():
        print("no claims extracted")
        return
    claims = pd.read_parquet(CLAIMS_PATH)
    log = pd.read_parquet(EXTRACT_LOG) if EXTRACT_LOG.exists() else pd.DataFrame()

    print(f"\n{len(claims):,} claims from {claims['accession'].nunique():,} filings")
    print(f"  distinct counterparties: {claims['counterparty'].nunique():,}")
    print("\n  by relation:")
    for relation, count in claims["relation"].value_counts().items():
        print(f"    {relation:12} {count:6,}")
    print("\n  by confidence:")
    for level, count in claims["confidence"].value_counts().items():
        print(f"    {level:12} {count:6,}")

    if not log.empty and "n_raw" in log:
        ok = log[log["status"] == "ok"]
        raw_total, kept_total = ok["n_raw"].sum(), ok["n_kept"].sum()
        if raw_total:
            print(f"\n  validator: {kept_total:,}/{raw_total:,} claims kept "
                  f"({100*kept_total/raw_total:.1f}%)")
            for column in [c for c in ok.columns if c.startswith("rej_")]:
                if ok[column].sum():
                    print(f"    rejected {column[4:]:14} {int(ok[column].sum()):,}")
        print(f"  mean {ok['seconds'].mean():.1f}s/filing over {len(ok):,} filings")

    print("\n  most-named counterparties:")
    for name, count in claims["counterparty"].value_counts().head(15).items():
        print(f"    {count:4}  {name[:60]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", required=True, choices=["filings", "extract", "report"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--model", default=local_model.DEFAULT_MODEL)
    parser.add_argument("--universe", default="index", choices=["index", "supplier"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.stage == "filings":
        stage_filings(config, args.limit, args.start, args.end, args.universe)
    elif args.stage == "extract":
        stage_extract(config, args.limit, args.model, args.workers)
    else:
        _report()


if __name__ == "__main__":
    main()
