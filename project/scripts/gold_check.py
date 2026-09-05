"""Validate hand annotations against the passages they claim to come from.

The annotations are held to the same grounding standard the models are: every
evidence span must survive `extract._grounded` against the filing's selected
passages. An annotator misremembering a quote is the same defect as a model
inventing one, and there is no reason to check only one of them.

Also reports which annotated counterparties resolve, so the share of the gold
set that can become a graph edge is known before anything is scored against it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config, PROCESSED  # noqa: E402
from src.graph import evaluate, extract, resolve  # noqa: E402

DIR = PROCESSED / "gold_candidates"
WORK = PROCESSED / "gold_work"


def main() -> None:
    config = Config.load("configs/default.yaml")
    lookup = resolve.build_lookup(resolve.load_reference(config.data.sec_user_agent))

    # Read the canonical benchmark, not the per-batch drafts. The scorer loads
    # this file, and a checker reading a different copy validates a set nobody
    # is scored against -- which is how five NVIDIA suppliers were added to the
    # benchmark and reported as absent from it in the same run.
    gold = evaluate.load_gold()

    manifest = {e["accession"]: e for e in json.loads((DIR / "_manifest.json").read_text())}
    bad, rows = [], []
    for accession, links in gold.items():
        body = (DIR / f"{accession}.txt").read_text(encoding="utf-8")
        haystack = f" {extract._normalise(body)} "
        tokens = set(haystack.split())
        for link in links:
            evidence = extract._normalise(link["evidence"])
            ok = extract._grounded(evidence, haystack, tokens)
            if not ok:
                bad.append((accession, link["counterparty"], link["evidence"][:70]))
            ticker, tier = resolve.resolve_one(link["counterparty"], lookup)
            rows.append({
                "accession": accession, "ticker": manifest[accession]["ticker"],
                "counterparty": link["counterparty"], "relation": link["relation"],
                "grounded": ok, "resolves_to": ticker, "tier": tier,
                "flagged": bool(link.get("flag")),
            })

    frame = pd.DataFrame(rows)
    annotated = len(gold)
    print(f"annotated {annotated} of {len(manifest)} filings   {len(frame)} links")
    if frame.empty:
        return
    print(f"  empty filings      {sum(1 for v in gold.values() if not v)}")
    print(f"  grounded           {int(frame.grounded.sum())}/{len(frame)}")
    print(f"  resolves to ticker {int(frame.resolves_to.notna().sum())}/{len(frame)} "
          f"({frame.resolves_to.notna().mean():.0%}) -- the share that can become an edge")
    print(f"  flagged for review {int(frame.flagged.sum())}")
    print("\n  by relation"); print(frame.relation.value_counts().to_string())
    if bad:
        print(f"\nUNGROUNDED EVIDENCE ({len(bad)}) -- fix these:")
        for accession, name, span in bad:
            print(f"  {accession} {name}: {span!r}")
    frame.to_csv(WORK / "_summary.csv", index=False)


if __name__ == "__main__":
    main()
