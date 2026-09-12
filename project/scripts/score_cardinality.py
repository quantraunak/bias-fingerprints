"""Score the controlled-k runs. Recall and precision per document, per cell.

    python3 scripts/score_cardinality.py

Gold is the injected set, so k is exact. Matching is on the normalised
counterparty string, using the same normaliser the resolver uses, so a model
that writes "Taiwan Semiconductor Manufacturing Co." is not scored as a miss
against "Taiwan Semiconductor Manufacturing".
"""
from __future__ import annotations

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from src.config import PROCESSED
from src.graph import extract, resolve

DOCS = PROCESSED / "cardinality_docs.json"
RUNS = PROCESSED / "cardinality_runs"
CELLS = {"A": "dense/think", "B": "dense/no-think", "C": "MoE/think", "D": "MoE/no-think"}



def _subsumes(want: str, got: set[str]) -> bool:
    """Credit a gold item when an emission names the same entity more specifically.

    Gold "Komatsu" against emitted "Komatsu Cummins Chile, Ltda." is a hit, not a
    miss: both are drawn from the same injected sentence. Requires one token set
    to contain the other and the shared tokens to be more than a single generic
    word, which is the same test the resolver uses to refuse "City Council" -> CHCO.
    """
    wt = set(want.split())
    if not wt:
        return False
    for g in got:
        gt = set(g.split())
        if not gt or not (wt <= gt or gt <= wt):
            continue
        shared = wt & gt
        if shared and not (len(shared) == 1 and next(iter(shared)) in resolve.AMBIGUOUS_SINGLE):
            return True
    return False


def main() -> None:
    docs = {d["doc_id"]: d for d in json.loads(DOCS.read_text())}
    rows = []
    for cell in sorted(CELLS):
        directory = RUNS / cell
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            doc = docs[path.stem]
            payload = json.loads(path.read_text())
            emitted = extract.parse_response(payload["text"])
            got = {resolve.normalize(l["counterparty"]) for l in emitted if l.get("counterparty")}
            want = {resolve.normalize(g["counterparty"]) for g in doc["gold"]}
            tp = len(got & want) + sum(1 for w in (want - got) if _subsumes(w, got - want))
            rows.append({"cell": cell, "doc_id": path.stem, "k": doc["k_actual"],
                         "emitted": len(got), "tp": tp,
                         "recall": tp / max(len(want), 1),
                         "precision": tp / max(len(got), 1) if got else np.nan,
                         "truncated": payload.get("truncated", False),
                         "seconds": payload.get("seconds", 0.0)})
    if not rows:
        sys.exit("no runs found")
    f = pd.DataFrame(rows)
    f.to_csv(Path(__file__).resolve().parents[1] / "reports" / "cardinality_scores.csv", index=False)

    for cell, group in f.groupby("cell"):
        print(f"\n=== cell {cell}  {CELLS[cell]}   n={len(group)} docs, "
              f"{group.truncated.sum()} truncated ===")
        print(f"{'k':>4} {'n':>4} {'recall':>8} {'precision':>10} {'emitted':>8}")
        for k, g in group.groupby("k"):
            print(f"{k:>4} {len(g):>4} {g.recall.mean():>8.3f} "
                  f"{g.precision.mean():>10.3f} {g.emitted.mean():>8.1f}")
        if group.k.nunique() > 2:
            from scipy import stats
            rho = stats.spearmanr(group.k, group.recall)
            x = np.log2(group.k.values); y = group.recall.values
            slope = np.polyfit(x, y, 1)[0]
            print(f"  Spearman(k, recall) = {rho.statistic:+.3f}  p = {rho.pvalue:.2e}")
            print(f"  OLS slope on log2(k) = {slope:+.4f} recall per doubling")
            lo = group[group.k <= 3].recall.mean(); hi = group[group.k >= 16].recall.mean()
            print(f"  recall k<=3 {lo:.3f} vs k>=16 {hi:.3f}  ->  drop {lo-hi:+.3f} "
                  f"({'MEETS' if lo-hi >= 0.15 else 'below'} the 0.15 pre-registered threshold)")


if __name__ == "__main__":
    main()
