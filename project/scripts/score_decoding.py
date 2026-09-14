"""Constrained vs unconstrained decoding: recall against k, matched documents.

    python3 scripts/score_decoding.py

Constrained arm reuses the completed cell D runs; unconstrained comes from
run_decoding_arm.py. Both are re-parsed from raw cached text at scoring time,
so a parser fix applies retroactively without re-running the model.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from src.config import PROCESSED
from src.graph import extract, resolve
from run_decoding_arm import parse_loose
from score_cardinality import _subsumes

DOCS = {d["doc_id"]: d for d in json.loads((PROCESSED / "cardinality_docs.json").read_text())}
# qwen3:14b at num_ctx 32768. The 30B arm was context-bound -- it wrote 7,000 to
# 12,000 tokens unconstrained against an 8,192 window -- so both arms moved to a
# model whose generation fits. See HYPOTHESIS_CARDINALITY.md, 2026-09-14.
ARMS = {
    "constrained":   (PROCESSED / "decoding_runs" / "qwen3_14b_constrained", False),
    "unconstrained": (PROCESSED / "decoding_runs" / "qwen3_14b_unconstrained", True),
}


def score(directory: Path, loose: bool) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.stem not in DOCS:
            continue
        doc, payload = DOCS[path.stem], json.loads(path.read_text())
        links = parse_loose(payload["text"]) if loose else extract.parse_response(payload["text"])
        got = {resolve.normalize(l["counterparty"]) for l in links if l.get("counterparty")}
        want = {resolve.normalize(g["counterparty"]) for g in doc["gold"]}
        tp = len(got & want) + sum(1 for w in (want - got) if _subsumes(w, got - want))
        rows.append({"doc_id": path.stem, "k": doc["k_actual"], "emitted": len(got), "tp": tp,
                     "recall": tp / max(len(want), 1),
                     "precision": tp / max(len(got), 1) if got else np.nan,
                     "tokens": payload.get("eval_count", 0)})
    return pd.DataFrame(rows)


def main() -> None:
    frames = {}
    for arm, (directory, loose) in ARMS.items():
        if directory.exists():
            frames[arm] = score(directory, loose)

    common = None
    for frame in frames.values():
        ids = set(frame.doc_id)
        common = ids if common is None else common & ids
    print(f"documents scored in every arm: {len(common or [])}\n")

    for arm, frame in frames.items():
        frame = frame[frame.doc_id.isin(common)]
        if frame.empty:
            continue
        print(f"=== {arm}  n={len(frame)} ===")
        print(f"{'k':>4} {'n':>4} {'recall':>8} {'precision':>10} {'emitted':>8} {'tokens':>8}")
        for k, g in frame.groupby("k"):
            print(f"{k:>4} {len(g):>4} {g.recall.mean():>8.3f} {g.precision.mean():>10.3f} "
                  f"{g.emitted.mean():>8.1f} {g.tokens.mean():>8.0f}")
        if frame.k.nunique() > 2:
            from scipy import stats
            rho = stats.spearmanr(frame.k, frame.recall)
            slope = np.polyfit(np.log2(frame.k), frame.recall, 1)[0]
            lo, hi = frame[frame.k <= 3].recall.mean(), frame[frame.k >= 16].recall.mean()
            print(f"  Spearman {rho.statistic:+.3f} p={rho.pvalue:.2e}   "
                  f"slope {slope:+.4f}/doubling   k<=3 {lo:.3f} vs k>=16 {hi:.3f} "
                  f"drop {lo-hi:+.3f}")
        print()

    if len(frames) == 2:
        a = frames["constrained"].set_index("doc_id").loc[sorted(common)]
        b = frames["unconstrained"].set_index("doc_id").loc[sorted(common)]
        print("=== paired difference, unconstrained - constrained ===")
        print(f"{'k':>4} {'n':>4} {'d_recall':>10} {'token ratio':>12}")
        for k in sorted(a.k.unique()):
            m = a.k == k
            d = (b.recall[m] - a.recall[m])
            ratio = b.tokens[m].mean() / max(a.tokens[m].mean(), 1)
            print(f"{k:>4} {int(m.sum()):>4} {d.mean():>+10.3f} {ratio:>11.1f}x")


if __name__ == "__main__":
    main()
