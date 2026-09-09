"""Recall against the number of relationships a real filing discloses.

    python scripts/ablation_curve.py --baseline          # go/no-go check first
    python scripts/ablation_curve.py --model qwen3:30b-a3b

Documents are real filings with some disclosures removed and replaced by
neutral prose from the same filing, so cardinality varies while the text stays
a coherent 10-K. This replaces the synthetic-document approach, which produced
documents the model did not read like filings: recall at one planted item was
1/6 against 0.66-1.00 on real filings, so the construction dominated the effect
under study.

`--baseline` is the control, and it runs first by design. It measures recall at
full cardinality, where the document is unmodified. If that does not land in the
range the model achieves on real filings, the instrument is not measuring the
model and no curve from it means anything.

The reference item set comes from the dense model's validated extractions, so
measuring the dense model with it would be circular. Measure a different model,
or use `--gold` to restrict to the annotated filings.
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
    ablate, extract, filings, local_model, passages, resolve,
)

CACHE = PROCESSED / "ablation_cache"
OUT = PROCESSED / "ablation.parquet"
REFERENCE = PROCESSED / "extract_cache" / "qwen3_32b"
NUM_PREDICT = 4000
SEED = 7
MIN_ITEMS = 12


def load_filing(accession: str, index: pd.DataFrame):
    """Selected passages and the validated items disclosed in them."""
    row = index.loc[accession]
    text = Path(row.path).read_text(encoding="utf-8", errors="ignore")
    selected = passages.select(text)
    payload = json.loads((REFERENCE / f"{accession}.json").read_text())
    kept, _ = extract.validate(extract.parse_response(payload["text"]), selected)
    items = [{"counterparty": l["counterparty"], "evidence": l["evidence"]}
             for l in kept
             if ablate.locate(selected, l["evidence"]) is not None]
    return row, selected, items


def run(model: str, document: list[str], ticker: str, filed: str,
        think: bool | None, path: Path) -> list[str] | None:
    if path.exists():
        return json.loads(path.read_text())["counterparties"]
    response = local_model.generate(
        extract.SYSTEM, extract.user_prompt(ticker, ticker, filed, document),
        extract.Extraction.model_json_schema(),
        model=model, timeout=1800, think=think, num_predict=NUM_PREDICT,
    )
    if not response.ok:
        return None
    names = [l["counterparty"] for l in extract.parse_response(response.text)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"counterparties": names, "seconds": response.seconds}))
    return names


def recovered(extracted: list[str], retained: list[str], lookup) -> int:
    def key(name):
        ticker, _ = resolve.resolve_one(name, lookup)
        return ticker or resolve.normalize(name)
    return len({key(n) for n in extracted if str(n).strip()} & {key(n) for n in retained})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="qwen3:30b-a3b")
    parser.add_argument("--no-think", dest="think", action="store_const",
                        const=False, default=None)
    parser.add_argument("--baseline", action="store_true",
                        help="control only: full cardinality, no ablation")
    parser.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--filings", type=int, default=6)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--min-items", type=int, default=MIN_ITEMS)
    parser.add_argument("--max-items", type=int, default=10**6,
                        help="upper bound on reference-set size, for the "
                             "low-cardinality arm of the baseline control")
    args = parser.parse_args()

    config = Config.load(args.config)
    lookup = resolve.build_lookup(resolve.load_reference(config.data.sec_user_agent))

    def has_company(sentence: str) -> bool:
        return any(resolve.resolve_one(n, lookup)[0]
                   for _, n in passages.company_candidates(sentence))

    index = filings.load_index().set_index("accession")
    candidates = []
    for path in REFERENCE.glob("*.json"):
        if path.stem not in index.index:
            continue
        try:
            row, selected, items = load_filing(path.stem, index)
        except (json.JSONDecodeError, KeyError):
            continue
        if args.min_items <= len(items) <= args.max_items:
            candidates.append((path.stem, row, selected, items))
    candidates.sort(key=lambda c: -len(c[3]))
    if args.max_items < 10**6:
        candidates.sort(key=lambda c: len(c[3]))
    candidates = candidates[: args.filings]
    print(f"{len(candidates)} filings with {args.min_items}-{args.max_items} locatable items")

    tag = args.model.replace(":", "_") + ("_nothink" if args.think is False else "")
    fractions = [1.0] if args.baseline else [float(f) for f in args.fractions.split(",")]
    rows = []

    for accession, row, selected, items in candidates:
        carriers = {s for it in items
                    for s in [(ablate.locate(selected, it["evidence"]) or (0, ""))[1]]}
        replacements = ablate.neutral_sentences(selected, carriers, has_company)
        for fraction in fractions:
            keep = max(1, round(fraction * len(items)))
            for replicate in range(1 if fraction == 1.0 else args.replicates):
                rng = random.Random(SEED + hash((accession, keep, replicate)) % 10000)
                document, retained = ablate.ablate(selected, items, keep, replacements, rng)
                path = CACHE / tag / accession / f"k{keep:02d}_r{replicate}.json"
                names = run(args.model, document, row.ticker,
                            str(pd.Timestamp(row.filed).date()), args.think, path)
                if names is None:
                    print(f"  {row.ticker} keep={keep}: failed", flush=True)
                    continue
                got = recovered(names, retained, lookup)
                rows.append({
                    "model": args.model, "accession": accession, "ticker": row.ticker,
                    "n_items": len(items), "keep": keep, "replicate": replicate,
                    "recovered": got, "returned": len(names),
                    "recall": got / keep, "chars": sum(len(p) for p in document),
                })
                print(f"  {row.ticker:6} keep={keep:2d}/{len(items):2d} r={replicate}  "
                      f"recovered {got:2d}  recall {got/keep:.2f}", flush=True)

    if not rows:
        print("no results")
        return
    frame = pd.DataFrame(rows)

    if args.baseline:
        print("\n=== BASELINE (unmodified documents) ===")
        print(frame[["ticker", "n_items", "recovered", "recall", "chars"]].to_string(index=False))
        print(f"\nmean recall {frame.recall.mean():.3f}")
        print("Real-filing recall for this model on benchmark v2: 0.66-1.00.")
        print("Below that range means the ablation pipeline, not the model, is")
        print("driving the result, and no curve from it is interpretable.")
        return

    if OUT.exists():
        keys = ["model", "accession", "keep", "replicate"]
        frame = (pd.concat([pd.read_parquet(OUT), frame])
                 .drop_duplicates(keys, keep="last").reset_index(drop=True))
    frame.to_parquet(OUT)
    print("\n" + frame.groupby("keep").agg(
        recall=("recall", "mean"), recovered=("recovered", "mean"),
        n=("recall", "size")).round(3).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
