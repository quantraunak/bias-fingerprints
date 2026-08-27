"""Measure what the local model costs in extraction quality.

The pre-registration commits to reporting extraction error as a measured
quantity. Running an 8B model over the corpus is a cost decision, and a cost
decision without a measured consequence is just an assumption. This scores model
output against a hand-checked sample.

Matching is on (counterparty, relation) after normalisation, because the same
company is written a dozen ways across filings and an exact string comparison
would score correct extractions as misses. Direction is *not* forgiven: calling
a supplier a customer inverts the return-propagation hypothesis for that edge,
so it counts as an error rather than a partial credit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.graph import resolve

# parents: [0] graph, [1] src, [2] project, [3] repository root -- docs/ is
# at the repository root, not under project/.
GOLD_PATH = Path(__file__).resolve().parents[3] / "docs" / "gold_links.json"


def _key(counterparty: str, relation: str, lookup: dict | None) -> tuple[str, str]:
    """Compare on resolved ticker where possible, normalised name otherwise."""
    if lookup is not None:
        ticker, _ = resolve.resolve_one(counterparty, lookup)
        if ticker:
            return (ticker, relation)
    return (resolve.normalize(counterparty), relation)


def score_filing(
    predicted: list[dict], gold: list[dict], lookup: dict | None = None
) -> dict:
    p = {_key(l["counterparty"], l["relation"], lookup) for l in predicted}
    g = {_key(l["counterparty"], l["relation"], lookup) for l in gold}
    return {
        "true_positive": len(p & g),
        "false_positive": len(p - g),
        "false_negative": len(g - p),
        "predicted": len(p),
        "gold": len(g),
        "missed": sorted(f"{n}:{r}" for n, r in (g - p)),
        "spurious": sorted(f"{n}:{r}" for n, r in (p - g)),
    }


def score(
    claims: pd.DataFrame, gold: dict[str, list[dict]], lookup: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Per-filing and aggregate precision/recall over the annotated sample."""
    rows = []
    for accession, gold_links in gold.items():
        predicted = claims[claims["accession"] == accession]
        rows.append(
            {
                "accession": accession,
                "ticker": gold_links[0].get("_ticker") if gold_links else None,
                **score_filing(predicted.to_dict("records"), gold_links, lookup),
            }
        )

    frame = pd.DataFrame(rows)
    tp, fp, fn = frame["true_positive"].sum(), frame["false_positive"].sum(), frame["false_negative"].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return frame, {
        "filings": len(frame),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(2 * precision * recall / (precision + recall)), 3) if precision + recall else 0.0,
    }


def load_gold(path: Path = GOLD_PATH) -> dict[str, list[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"No annotated sample at {path}")
    return json.loads(path.read_text())


def save_gold(gold: dict[str, list[dict]], path: Path = GOLD_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gold, indent=2, sort_keys=True))
