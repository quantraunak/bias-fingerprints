"""Construct documents with a known item count k, for the cardinality study.

    python3 scripts/build_cardinality_docs.py --per-level 15

Each document is real filing prose: k verbatim-validated link-bearing sentences
injected at uniformly random positions into filler drawn from filings where no
configuration found a link, with every capitalised-run sentence stripped from the
filler so it cannot carry a legitimate counterparty. Padded to the same 14,000
character budget `passages.select` enforces, so input length is constant by
construction rather than merely uncorrelated with k.

Gold is the injected set, which makes k exact and annotation unnecessary.
See docs/HYPOTHESIS_CARDINALITY.md, "Superseding design".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROCESSED  # noqa: E402
from src.graph import filings, passages  # noqa: E402

LEVELS = [1, 2, 4, 8, 16]
BUDGET = 9000
MAX_SPAN = 200
SENTENCE = re.compile(r"(?<=[.!?])\s+")
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "cardinality_docs.json"
SCRATCH = "/private/tmp/claude-501/-Users-raunaksood/92717330-1c6f-4524-b9d9-d5009eb79b08/scratchpad"


SCREENED = PROCESSED / "cardinality_bank_screened.parquet"


def item_bank() -> pd.DataFrame:
    """Evidence spans carrying exactly one counterparty, from both model families.

    Once `screen_item_bank.py` has run, the screened bank supersedes this: spans
    the extractor demonstrably recovers at k=1 in a short document. Syntactic
    guesses at which spans transplant were wrong three times running.
    """
    if SCREENED.exists():
        frame = pd.read_parquet(SCREENED)
        # Long spans force long documents, which floor the baseline: k=16 at
        # 300 chars is 4,800 characters of needles before any filler.
        return frame[frame.evidence.str.len() <= MAX_SPAN].reset_index(drop=True)
    frames = [pd.read_parquet(PROCESSED / "link_claims_qwen.parquet")]
    dense = Path(SCRATCH) / "claims_qwen3_32b.parquet"
    if dense.exists():
        frames.append(pd.read_parquet(dense))
    claims = pd.concat(frames, ignore_index=True)
    claims = claims[claims.evidence.notna()]
    claims["evidence"] = claims.evidence.astype(str).str.strip()
    claims = claims[claims.evidence.str.len().between(80, 300)]
    # A span naming several counterparties would make items and sentences covary.
    per_span = claims.groupby("evidence").counterparty.nunique()
    claims = claims[claims.evidence.isin(per_span[per_span == 1].index)]
    # A span must be transplantable. The prompt asks for the counterparties of
    # *this* filer, so a span naming its original filer in the third person
    # ("BDS faces competition from Lockheed Martin") answers a question the
    # constructed document does not ask, and the model correctly declines it.
    # First-person spans are filer-agnostic: "Wal-Mart accounted for 22 percent
    # of our net sales" is true of whoever files it. Measured in the k=1 pilot:
    # mixed spans gave recall 0.333 at k=1, which is a broken instrument.
    claims = claims[claims.evidence.str.contains(r"\b(?:we|our|us)\b", case=False, regex=True)]
    # The counterparty must be a name. Validation only ever checked that the
    # evidence was verbatim, so strings like "One of our end customers" reached
    # the claim table and would otherwise become gold items no model can hit.
    name = claims.counterparty.astype(str).str.strip()
    claims = claims[
        name.str.match(r"^[A-Z]")                       # starts capitalised
        & ~name.str.match(r"^(One|Our|The|A|An|Some|Certain|Two|Three)\b")
        & (name.str.len() >= 3)
        & name.apply(lambda n: bool(passages.company_candidates(n)))
    ]
    bank = claims.drop_duplicates("evidence").drop_duplicates("counterparty")
    return bank[["counterparty", "relation", "evidence"]].reset_index(drop=True)


def filler_sentences(limit: int = 200) -> list[str]:
    """Relationship-dense sentences that name no company.

    Sourced from the selected passages of *productive* filings, not from filings
    with no claims. Empty filings are the least relationship-dense text in the
    corpus, and using them made the haystack far more inert than a real prompt --
    `passages.select` keeps only paragraphs scoring above 0.5, so a genuine
    14,000-character prompt is dense with relationship language throughout.
    Measured cost of getting this wrong: 0.25 recall at constant k.
    """
    claims = pd.read_parquet(PROCESSED / "link_claims_qwen.parquet")
    productive = set(claims.accession)
    index = filings.load_index()
    rows = [r for r in index.itertuples() if r.accession in productive][:limit]

    out: list[str] = []
    for row in rows:
        try:
            selected = passages.select(Path(row.path).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        for paragraph in selected:
            for sentence in SENTENCE.split(paragraph):
                sentence = sentence.strip()
                if not (60 <= len(sentence) <= 300):
                    continue
                if passages.company_candidates(sentence):
                    continue
                # company_candidates keys on capitalised runs, so an all-caps
                # ticker or acronym slips through. ADP reached the filler this
                # way and was emitted as a counterparty across many documents.
                if re.search(r"\b[A-Z]{2,}\b", sentence):
                    continue
                out.append(sentence)
        if len(out) > 8000:
            break
    seen, unique = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); unique.append(s)
    return unique


def compose(items: pd.DataFrame, filler: list[str], rng) -> tuple[list[str], list[dict]]:
    """One document: k injected sentences at random positions, padded to BUDGET."""
    injected = [s.strip() for s in items.evidence]
    body: list[str] = []
    total = sum(len(s) + 1 for s in injected)
    pool = list(rng.permutation(len(filler)))
    while total < BUDGET and pool:
        s = filler[pool.pop()]
        body.append(s); total += len(s) + 1

    positions = sorted(rng.choice(len(body) + 1, size=len(injected), replace=False)
                       if len(body) + 1 >= len(injected) else range(len(injected)))
    for offset, (at, sentence) in enumerate(zip(positions, injected)):
        body.insert(at + offset, sentence)

    # Paragraphs are built from whole sentences. Slicing at fixed character
    # offsets cut injected sentences across a paragraph boundary, which leaves
    # the needle fragmented and unfindable and breaks the verbatim check too.
    paragraphs, current, kept = [], "", []
    for sentence in body:
        if len(current) + len(sentence) + 1 > 1200 and current:
            paragraphs.append(current.strip()); current = ""
        current = f"{current} {sentence}".strip()
    if current:
        paragraphs.append(current.strip())
    text = " ".join(paragraphs)
    kept = [i for i, s in enumerate(injected) if s in text]
    gold = [{"counterparty": r.counterparty, "relation": r.relation}
            for i, r in enumerate(items.itertuples()) if i in set(kept)]
    return paragraphs, gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-level", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    bank = item_bank()
    filler = filler_sentences()
    print(f"item bank {len(bank):,} single-counterparty spans")
    print(f"filler    {len(filler):,} candidate-free sentences")
    if len(filler) < 200:
        sys.exit("too little filler; widen the empty-filing limit")

    docs = []
    for k in LEVELS:
        for replicate in range(args.per_level):
            items = bank.sample(n=k, random_state=int(rng.integers(1 << 31)))
            paragraphs, gold = compose(items, filler, rng)
            docs.append({"doc_id": f"k{k:02d}_r{replicate:02d}", "k_intended": k,
                         "k_actual": len(gold), "chars": sum(len(p) for p in paragraphs),
                         "paragraphs": paragraphs, "gold": gold})
        got = [d for d in docs if d["k_intended"] == k]
        print(f"  k={k:>2}  {len(got)} docs  "
              f"k_actual mean {np.mean([d['k_actual'] for d in got]):.2f}  "
              f"chars mean {np.mean([d['chars'] for d in got]):.0f}")

    OUT.write_text(json.dumps(docs))
    print(f"\n{len(docs)} documents -> {OUT}")


if __name__ == "__main__":
    main()
