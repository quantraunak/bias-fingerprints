"""Measure agreement between two independent annotations of the same filings.

    python scripts/annotator_agreement.py docs/gold_links.json docs/gold_links_b.json

A benchmark annotated once by one person is reproducible but not validated: the
rules in `SPECIFICATIONS.md` may simply encode that annotator's reading. Double
annotating a subset and reporting agreement is what separates "these are the
labels" from "these labels are recoverable by someone else following the rules".

Three numbers, because the task has three distinct decisions and they fail
differently:

* **Disclosure decision** -- does this filing disclose any relationship at all?
  A fixed binary judgement over a fixed set of filings, so Cohen's kappa is
  well defined here and is the headline number. It is also the decision the
  benchmark's five deliberately-empty filings turn on: if two annotators cannot
  agree which filings are empty, the auditable negatives are not auditable.
* **Link-set agreement** -- pairwise F1 over (counterparty, relation) pairs.
  Kappa is not defined for this: the set of links an annotator *could* have
  written is unbounded, so chance agreement has no denominator. F1 is symmetric
  between two annotators and is the same quantity the model is scored on, which
  makes annotator disagreement directly comparable to model error.
* **Relation agreement** -- among counterparties both annotators found, how
  often do they assign the same direction. Isolates disagreement about what a
  relationship *is* from disagreement about whether it is there.

Every disagreement is printed, because the point of measuring it is to
adjudicate it and tighten the rule that failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.graph import evaluate, resolve  # noqa: E402


def kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for a binary decision over the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1.0:
        return float("nan")  # no variation to agree about
    return (observed - expected) / (1 - expected)


def f1(a: set, b: set) -> float:
    """Symmetric pairwise F1. Either annotator as reference gives the same value."""
    if not a and not b:
        return 1.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(b), overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-resolve", action="store_true",
                        help="compare normalised names rather than resolved tickers")
    args = parser.parse_args()

    first = json.loads(Path(args.first).read_text())
    second = json.loads(Path(args.second).read_text())
    shared = sorted(set(first) & set(second))
    if not shared:
        print("no filings annotated by both")
        return

    lookup = None
    if not args.no_resolve:
        lookup = resolve.build_lookup(
            resolve.load_reference(Config.load(args.config).data.sec_user_agent)
        )

    def keys(links: list[dict]) -> set:
        return {evaluate._key(l["counterparty"], l["relation"], lookup) for l in links}

    def names(links: list[dict]) -> dict:
        return {evaluate._key(l["counterparty"], "", lookup)[0]: l["relation"] for l in links}

    rows, disclosure_a, disclosure_b = [], [], []
    relation_same = relation_total = 0
    for accession in shared:
        ka, kb = keys(first[accession]), keys(second[accession])
        disclosure_a.append(bool(first[accession]))
        disclosure_b.append(bool(second[accession]))
        na, nb = names(first[accession]), names(second[accession])
        for name in set(na) & set(nb):
            relation_total += 1
            relation_same += na[name] == nb[name]
        rows.append((accession, ka, kb))

    overall_a = {(i, k) for i, ka, _ in rows for k in ka}
    overall_b = {(i, k) for i, _, kb in rows for k in kb}

    print(f"filings annotated by both: {len(shared)}")
    print(f"links: {len(overall_a)} (first) vs {len(overall_b)} (second)\n")
    print(f"disclosure decision   Cohen's kappa {kappa(disclosure_a, disclosure_b):.3f}"
          f"   ({sum(x == y for x, y in zip(disclosure_a, disclosure_b))}/{len(shared)} filings agree)")
    print(f"link set              pairwise F1   {f1(overall_a, overall_b):.3f}")
    if relation_total:
        print(f"relation direction    {relation_same}/{relation_total} agree "
              f"({relation_same / relation_total:.1%}) on shared counterparties")

    exact = sum(1 for _, ka, kb in rows if ka == kb)
    print(f"\nexact agreement on {exact}/{len(shared)} filings\n")
    for accession, ka, kb in rows:
        if ka == kb:
            continue
        print(f"  {accession}")
        for key in sorted(ka - kb):
            print(f"    first only   {key[0]}:{key[1]}")
        for key in sorted(kb - ka):
            print(f"    second only  {key[0]}:{key[1]}")


if __name__ == "__main__":
    main()
