"""Measure extraction recall against the number of items planted in a document.

    python scripts/cardinality_curve.py --model qwen3:30b-a3b --replicates 6

Documents are constructed rather than annotated, so the label is exact and the
cardinality is a controlled variable rather than an observed one. Total length
is held constant across cardinalities, which is the confound that would
otherwise make a length effect look like a cardinality effect.

Three mechanisms predict different curves for recovered-versus-planted, and the
point of measuring many cardinalities rather than two buckets is that they are
distinguishable:

    independent   R(n) = p * n              recall per item is constant
    saturating    R(n) = k * (1 - e^-n/k)   a capacity ceiling at k items
    dilution      R(n) = n * p0 * e^-(l*n)  per-item recall falls as n grows

`fit_curves.py` does the model selection. This script only produces the data.

Resumable: every response is cached by (model, cardinality, replicate, spread),
so a killed run costs one call.
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
    extract, filings, local_model, passages, resolve, synthetic,
)

CACHE = PROCESSED / "cardinality_cache"
OUT = PROCESSED / "cardinality.parquet"
NUM_PREDICT = 4000  # E7: must leave room under the 8192-token context
SEED = 7


def build_pools(config: Config) -> tuple[pd.DataFrame, list[str]]:
    """Relationship sentences with known labels, and relationship-free filler."""
    claims = pd.read_parquet(PROCESSED / "link_claims_qwen.parquet")
    evidence = synthetic.evidence_pool(claims)

    index = filings.load_index().set_index("accession")
    cache = PROCESSED / "extract_cache" / "qwen3_32b"
    texts = []
    for path in cache.glob("*.json"):
        payload = json.loads(path.read_text())
        # Filings the extractor read and found nothing in: the model itself is
        # the filter, which is a stronger guarantee than a keyword screen.
        if payload.get("text", "").replace(" ", "") not in ('{"links":[]}',):
            continue
        accession = path.stem
        if accession not in index.index:
            continue
        raw = Path(index.loc[accession].path).read_text(encoding="utf-8", errors="ignore")
        texts.extend(passages.select(raw))
    return evidence, synthetic.filler_pool(texts)


def run_one(model: str, document: str, think: bool | None,
            cache_path: Path) -> list[str] | None:
    if cache_path.exists():
        return json.loads(cache_path.read_text())["counterparties"]

    response = local_model.generate(
        extract.SYSTEM,
        extract.user_prompt("TESTCO", "TESTCO", "2020-01-01", [document]),
        extract.Extraction.model_json_schema(),
        model=model, timeout=1800, think=think, num_predict=NUM_PREDICT,
    )
    if not response.ok:
        return None
    # Grounding is NOT applied. Every planted sentence is verbatim in the
    # document, so a correct extraction is grounded by construction; applying
    # the substring check here would only add a second failure mode on top of
    # the one being measured.
    links = extract.parse_response(response.text)
    names = [l["counterparty"] for l in links]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"counterparties": names,
                                      "seconds": response.seconds}))
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="qwen3:30b-a3b")
    parser.add_argument("--no-think", dest="think", action="store_const",
                        const=False, default=None)
    parser.add_argument("--cardinalities", default="0,1,2,3,5,8,12,16,20,25,30")
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--target-chars", type=int, default=4000)
    parser.add_argument("--spread", default="uniform", choices=["uniform", "front", "back"])
    args = parser.parse_args()

    config = Config.load(args.config)
    lookup = resolve.build_lookup(resolve.load_reference(config.data.sec_user_agent))
    evidence, filler = build_pools(config)
    print(f"pools: {len(evidence)} relationship sentences, {len(filler)} filler sentences")

    wanted = [int(c) for c in args.cardinalities.split(",")]
    usable = [n for n in wanted if n <= len(evidence)]
    if usable != wanted:
        print(f"  capped at pool size: {usable}")

    tag = args.model.replace(":", "_") + ("_nothink" if args.think is False else "")
    rows = []
    for n in usable:
        for replicate in range(args.replicates):
            # Seeded per cell so a rerun rebuilds the identical document, and so
            # every model sees the same documents at the same cardinality.
            rng = random.Random(SEED * 1000 + n * 10 + replicate)
            document, planted = synthetic.build(
                n, evidence, filler, rng,
                target_chars=args.target_chars, spread=args.spread,
            )
            path = CACHE / tag / args.spread / f"n{n:02d}_r{replicate}.json"
            names = run_one(args.model, document, args.think, path)
            if names is None:
                print(f"  n={n} r={replicate}: failed", flush=True)
                continue
            got = synthetic.recovered(names, planted, lookup, resolve.resolve_one)
            rows.append({
                "model": args.model, "spread": args.spread, "n_planted": n,
                "replicate": replicate, "n_recovered": got,
                "n_returned": len(names), "chars": len(document),
            })
            print(f"  n={n:2d} r={replicate}  recovered {got:2d}/{n:2d}  "
                  f"returned {len(names):2d}", flush=True)

    if not rows:
        print("no results")
        return

    frame = pd.DataFrame(rows)
    if OUT.exists():
        previous = pd.read_parquet(OUT)
        keys = ["model", "spread", "n_planted", "replicate"]
        frame = (pd.concat([previous, frame])
                 .drop_duplicates(keys, keep="last").reset_index(drop=True))
    frame.to_parquet(OUT)

    summary = (frame[frame.model == args.model]
               .groupby("n_planted")
               .agg(recovered=("n_recovered", "mean"),
                    returned=("n_returned", "mean"),
                    reps=("n_recovered", "size")))
    summary["recall"] = (summary.recovered / summary.index.to_series()).round(3)
    print("\n" + summary.to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
