"""Extract economic links with Claude via the Batch API.

The local 8B model scores precision 0.22 against the annotated sample: it finds
nearly every real link and emits three wrong ones for each. The errors are not
hallucinations -- the evidence is verbatim -- they are misclassifications, which
is the failure mode the evidence check cannot see. Director biographies became
supply relationships, an IP sale became a supplier, and a parts maker's OEM
customers came back labelled as its suppliers.

Batch is the right surface here: thousands of independent, latency-insensitive
calls at half price. Requests are keyed by accession number, so results are
matched by `custom_id` and never by position, and a run that dies after
submission is recovered from the batch id rather than re-paid for.

    python scripts/extract_batch.py --dry-run          # cost and coverage first
    python scripts/extract_batch.py --submit --limit 500
    python scripts/extract_batch.py --collect <batch_id>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import PROCESSED, Config  # noqa: E402
from src.graph import extract, filings, passages, resolve  # noqa: E402

CLAIMS_PATH = PROCESSED / "link_claims_claude.parquet"
LOG_PATH = PROCESSED / "extract_log_claude.parquet"
BATCH_STATE = PROCESSED / "_batches.json"

# Opus 5, halved for batch. Kept here so a cost estimate cannot silently drift
# from the model actually being called.
INPUT_PER_MTOK = 5.00 / 2
OUTPUT_PER_MTOK = 25.00 / 2
CHARS_PER_TOKEN = 3.6  # measured on filing prose; count_tokens is exact but needs a key
OUTPUT_TOKENS_ESTIMATE = 700

MAX_PER_BATCH = 20_000


def candidates(config: Config, limit: int | None, skip_done: bool = True) -> pd.DataFrame:
    """Filings worth paying for: those whose passages could yield a usable edge."""
    index = filings.load_index()
    if skip_done and CLAIMS_PATH.exists():
        done = set(pd.read_parquet(CLAIMS_PATH)["accession"])
        index = index[~index["accession"].isin(done)]

    reference = resolve.load_reference(config.data.sec_user_agent)
    lookup = resolve.build_lookup(reference)

    rows = []
    for n, row in enumerate(index.itertuples(), 1):
        try:
            text = Path(row.path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        selected = passages.select(text)
        if not selected:
            continue
        worth_it, hits = passages.prescreen(selected, lookup, resolve.resolve_one, row.ticker)
        if not worth_it:
            continue
        rows.append({
            "accession": row.accession, "ticker": row.ticker,
            "filed": row.filed, "passages": selected,
            "chars": sum(len(p) for p in selected), "prescreen_hits": hits,
        })
        if n % 250 == 0:
            print(f"  screened {n}/{len(index)}, kept {len(rows)}", flush=True)
        if limit and len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def estimate(work: pd.DataFrame) -> dict:
    input_tokens = work["chars"].sum() / CHARS_PER_TOKEN
    output_tokens = len(work) * OUTPUT_TOKENS_ESTIMATE
    return {
        "requests": len(work),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "usd": round(input_tokens / 1e6 * INPUT_PER_MTOK + output_tokens / 1e6 * OUTPUT_PER_MTOK, 2),
    }


def submit(work: pd.DataFrame) -> list[str]:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    ids = []
    for start in range(0, len(work), MAX_PER_BATCH):
        chunk = work.iloc[start : start + MAX_PER_BATCH]
        requests = [
            Request(
                custom_id=row.accession,
                params=MessageCreateParamsNonStreaming(
                    **extract.build_params(
                        row.ticker, row.ticker, str(pd.Timestamp(row.filed).date()), row.passages
                    )
                ),
            )
            for row in chunk.itertuples()
        ]
        batch = client.messages.batches.create(requests=requests)
        ids.append(batch.id)
        print(f"  submitted {batch.id}: {len(requests)} requests ({batch.processing_status})")

    state = json.loads(BATCH_STATE.read_text()) if BATCH_STATE.exists() else []
    state.extend({"id": b, "submitted": pd.Timestamp.utcnow().isoformat()} for b in ids)
    BATCH_STATE.write_text(json.dumps(state, indent=1))
    return ids


def collect(batch_id: str, work: pd.DataFrame, poll_seconds: int = 60) -> None:
    """Wait for one batch, validate every result, and write claims plus a log."""
    import anthropic

    client = anthropic.Anthropic()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"  {batch.processing_status}: {counts.processing} processing, "
              f"{counts.succeeded} done, {counts.errored} errored", flush=True)
        time.sleep(poll_seconds)

    by_accession = work.set_index("accession")
    records, log = [], []
    for result in client.messages.batches.results(batch_id):
        accession = result.custom_id
        if accession not in by_accession.index:
            continue
        row = by_accession.loc[accession]
        kind = result.result.type
        if kind != "succeeded":
            error = getattr(getattr(result.result, "error", None), "type", kind)
            log.append({"accession": accession, "ticker": row.ticker, "status": kind,
                        "error": error, "n_raw": 0, "n_kept": 0})
            continue

        message = result.result.message
        text = next((b.text for b in message.content if b.type == "text"), "")
        raw = extract.parse_response(text)
        kept, tally = extract.validate(raw, row.passages)
        for link in kept:
            records.append({
                "ticker": row.ticker, "accession": accession, "filed": row.filed,
                "counterparty": link["counterparty"], "relation": link["relation"],
                "revenue_pct": link.get("revenue_pct"), "confidence": link.get("confidence"),
                "evidence": link["evidence"],
            })
        log.append({
            "accession": accession, "ticker": row.ticker, "status": "ok", "error": None,
            "n_raw": len(raw), "n_kept": tally["kept"],
            "rej_no_evidence": tally["no_evidence"], "rej_anonymous": tally["anonymous"],
            "rej_empty_name": tally["empty_name"],
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        })

    _append(CLAIMS_PATH, extract.to_frame(records))
    _append(LOG_PATH, pd.DataFrame(log))

    frame = pd.DataFrame(log)
    ok = frame[frame.status == "ok"]
    print(f"\n{len(ok)} filings extracted, {len(frame) - len(ok)} failed")
    if not ok.empty:
        print(f"  claims kept        {int(ok.n_kept.sum()):,} of {int(ok.n_raw.sum()):,} raw")
        rejected = int(ok.n_raw.sum() - ok.n_kept.sum())
        print(f"  rejected by validation {rejected:,} "
              f"({rejected / max(int(ok.n_raw.sum()), 1):.1%}) -- the measured hallucination"
              f" and anonymous-reference rate")
        if "input_tokens" in ok:
            spend = (ok.input_tokens.sum() / 1e6 * INPUT_PER_MTOK
                     + ok.output_tokens.sum() / 1e6 * OUTPUT_PER_MTOK)
            print(f"  actual spend       ${spend:,.2f}")


def _append(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        key = "accession" if "n_raw" in frame.columns else None
        frame = frame.drop_duplicates(subset=key) if key else frame.drop_duplicates()
    frame.to_parquet(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", metavar="BATCH_ID")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = Config.load("configs/default.yaml")
    print("screening filings for extractable passages ...", flush=True)
    work = candidates(config, args.limit, skip_done=not args.collect)
    if work.empty:
        raise SystemExit("Nothing to extract.")

    numbers = estimate(work)
    print(f"\n{numbers['requests']:,} filings from {work.ticker.nunique()} issuers")
    print(f"  input  ~{numbers['input_tokens']:,} tokens")
    print(f"  output ~{numbers['output_tokens']:,} tokens (estimated)")
    print(f"  cost   ~${numbers['usd']:,.2f} at batch rates ({extract.MODEL})")

    if args.collect:
        collect(args.collect, work)
    elif args.submit:
        submit(work)
    elif not args.dry_run:
        print("\nNothing done. Pass --dry-run, --submit, or --collect <batch_id>.")
