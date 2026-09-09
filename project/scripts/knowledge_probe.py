"""Positive controls for the contamination measurement.

A null result from `measure_contamination.py` is only meaningful if the test
could have come out positive. Zero phantoms is consistent with two very
different worlds:

    the model reads rather than recalls        <- a finding
    the instrument cannot detect recall at all <- a broken test

These two probes separate them, and both have to pass before a null is worth
reporting.

**Probe A -- knowledge.** Ask the model, with no document at all, which
companies the filer names as customers and suppliers. If it cannot produce the
counterparty from memory, there is nothing available to leak and its absence
from extraction says nothing. Recall here is the ceiling on any leakage the
main experiment could possibly observe.

**Probe B -- induced leak.** Show the passage with the counterparty's name
removed entirely rather than replaced -- "our principal foundry partner"
instead of "Taiwan Semiconductor Manufacturing Company" -- so the context
invites a name and the document supplies none. A model that fills in the real
name here demonstrates the instrument's sensitivity: leakage is detectable when
it happens. A model that declines shows the grounding constraint holding under
maximum pressure, which is itself the mitigation result.

Probe B is the tighter control. Probe A establishes the knowledge exists;
Probe B establishes that the pipeline would surface it if it leaked.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config, PROCESSED  # noqa: E402
from src.graph import contamination, extract, filings, local_model, passages, resolve  # noqa: E402

CACHE = PROCESSED / "probe_cache"
NUM_PREDICT = 4000

KNOWLEDGE_SYSTEM = (
    "You answer from your own knowledge of public companies. "
    "Reply with JSON only."
)
KNOWLEDGE_SCHEMA = {
    "type": "object",
    "properties": {"companies": {"type": "array", "items": {"type": "string"}}},
    "required": ["companies"],
}

# Neutral stand-ins that preserve the relationship cue while removing identity.
# The point is to leave a gap a name would fill, not to make the sentence
# ungrammatical.
PLACEHOLDER = "a major partner"


def cached(path: Path, make):
    if path.exists():
        return json.loads(path.read_text())
    value = make()
    if value is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def probe_knowledge(model: str, ticker: str, think: bool | None) -> list[str] | None:
    """Which counterparties can the model name with no document in front of it?"""
    path = CACHE / "knowledge" / model.replace(":", "_") / f"{ticker}.json"

    def call():
        response = local_model.generate(
            KNOWLEDGE_SYSTEM,
            f"List the companies that {ticker} names as customers, suppliers, "
            f"foundries, contract manufacturers or distributors in its SEC 10-K "
            f"filings. Name specific companies you are aware of.",
            KNOWLEDGE_SCHEMA, model=model, timeout=900, think=think,
            num_predict=NUM_PREDICT,
        )
        if not response.ok:
            return None
        try:
            return {"companies": json.loads(response.text).get("companies", [])}
        except (json.JSONDecodeError, AttributeError):
            return {"companies": []}

    payload = cached(path, call)
    return None if payload is None else payload["companies"]


def blank_out(passages_in: list[str], name: str, lookup: dict) -> tuple[list[str], int]:
    """Remove the counterparty's name, leaving the relationship language intact."""
    out, removed = list(passages_in), 0
    for form in contamination.surface_forms(name, lookup):
        tokens = contamination.WORD.findall(form)
        if not tokens:
            continue
        flags = re.I if len(tokens) > 1 else 0
        pattern = re.compile(r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b", flags)
        for i, passage in enumerate(out):
            out[i], n = pattern.subn(PLACEHOLDER, passage)
            removed += n
    # Capitalise the placeholder where it opens a sentence. A lowercase
    # sentence start makes the passage read as malformed, and extraction could
    # then drop for reasons unrelated to the missing identity.
    opener = re.compile(r"(^|[.!?]\s+)" + re.escape(PLACEHOLDER))
    out = [opener.sub(lambda m: m.group(1) + PLACEHOLDER[0].upper() + PLACEHOLDER[1:], p)
           for p in out]
    return out, removed


def probe_induced(model: str, accession: str, counterparty: str, ticker: str,
                  filed: str, blanked: list[str], think: bool | None) -> list[str] | None:
    key = resolve.normalize(counterparty).replace(" ", "_")[:40] or "blank"
    path = CACHE / "induced" / model.replace(":", "_") / accession / f"{key}.json"

    def call():
        response = local_model.generate(
            extract.SYSTEM,
            extract.user_prompt(ticker, ticker, filed, blanked),
            extract.Extraction.model_json_schema(),
            model=model, timeout=1800, think=think, num_predict=NUM_PREDICT,
        )
        if not response.ok:
            return None
        raw = extract.parse_response(response.text)
        return {"raw": [l["counterparty"] for l in raw]}

    payload = cached(path, call)
    return None if payload is None else payload["raw"]


def names_match(candidates: list[str], target: str, lookup: dict) -> bool:
    target_key = resolve.normalize(target)
    target_ticker, _ = resolve.resolve_one(target, lookup)
    for candidate in candidates:
        key = resolve.normalize(candidate)
        if not key:
            continue
        if key == target_key or key in target_key or target_key in key:
            return True
        if target_ticker and resolve.resolve_one(candidate, lookup)[0] == target_ticker:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="qwen3:32b")
    parser.add_argument("--no-think", dest="think", action="store_const",
                        const=False, default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--probe", choices=["knowledge", "induced", "both"], default="both")
    parser.add_argument("--source", default=str(PROCESSED / "contamination.parquet"),
                        help="pairs to probe; falls back to the claims table")
    args = parser.parse_args()

    config = Config.load(args.config)
    lookup = resolve.build_lookup(resolve.load_reference(config.data.sec_user_agent))
    index = filings.load_index().set_index("accession")

    source = Path(args.source)
    if source.exists():
        pairs = pd.read_parquet(source)[["accession", "ticker", "counterparty"]]
    else:
        claims = pd.read_parquet(PROCESSED / "link_claims_qwen.parquet")
        resolved = resolve.resolve_frame(
            claims, resolve.load_reference(config.data.sec_user_agent))
        resolved = resolved[resolved.counterparty_ticker.notna()]
        pairs = (resolved.drop_duplicates(["accession", "counterparty"])
                 .sample(frac=1.0, random_state=7)[["accession", "ticker", "counterparty"]])
    pairs = pairs.head(args.limit)

    rows = []
    for i, row in enumerate(pairs.itertuples(), 1):
        if row.accession not in index.index:
            continue
        meta = index.loc[row.accession]
        record = {"ticker": row.ticker, "counterparty": row.counterparty}

        if args.probe in ("knowledge", "both"):
            known = probe_knowledge(args.model, row.ticker, args.think)
            record["knows"] = None if known is None else names_match(
                known, row.counterparty, lookup)

        if args.probe in ("induced", "both"):
            text = Path(meta.path).read_text(encoding="utf-8", errors="ignore")
            selected = passages.select(text)
            blanked, removed = blank_out(selected, row.counterparty, lookup)
            if removed == 0:
                record["filled_in"] = None
            else:
                emitted = probe_induced(args.model, row.accession, row.counterparty,
                                        row.ticker, str(pd.Timestamp(meta.filed).date()),
                                        blanked, args.think)
                record["filled_in"] = None if emitted is None else names_match(
                    emitted, row.counterparty, lookup)

        rows.append(record)
        print(f"  [{i}/{len(pairs)}] {row.ticker:6} {row.counterparty[:30]:30} "
              f"knows={record.get('knows')} filled_in={record.get('filled_in')}", flush=True)

    frame = pd.DataFrame(rows)
    out = PROCESSED / "knowledge_probe.parquet"
    frame.to_parquet(out)

    print(f"\n{len(frame)} pairs probed")
    if "knows" in frame:
        valid = frame.knows.dropna()
        print(f"  PROBE A  model names the counterparty from memory alone: "
              f"{valid.sum()}/{len(valid)}"
              + (f" ({valid.mean():.0%})" if len(valid) else ""))
    if "filled_in" in frame:
        valid = frame.filled_in.dropna()
        print(f"  PROBE B  model fills the name into a blanked passage:     "
              f"{valid.sum()}/{len(valid)}"
              + (f" ({valid.mean():.0%})" if len(valid) else ""))
    print("\nIf PROBE A is near zero the main null is trivial: there is no knowledge")
    print("to leak. If PROBE A is high and PROBE B near zero, the grounding")
    print("constraint is holding under pressure -- that is the mitigation result.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
