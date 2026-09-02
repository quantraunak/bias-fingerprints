"""Condense each sampled filing to the spans where a named link could be stated.

Reading 618k characters of selected passages to annotate a few hundred links is
mostly reading boilerplate. This narrows to the sentences that can carry a named
counterparty, without narrowing what counts as one.

The completeness argument: a named counterparty must appear inside a sentence
containing a capitalised run, because that is what a company name is. So every
sentence carrying a candidate is kept, together with one sentence either side,
since filings routinely state the name and the relationship in adjacent
sentences ("We depend on a single foundry. Taiwan Semiconductor manufactures
substantially all of our wafers."). Sentences with neither a candidate nor
relationship language cannot state a named link and are dropped.

`company_candidates` is deliberately over-inclusive, so this is a superset of
what any extractor working from these passages could legitimately find.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED  # noqa: E402
from src.graph import passages  # noqa: E402

DIR = PROCESSED / "gold_candidates"
CONTEXT_SENTENCES = 1


def condense(body: str) -> str:
    sentences = [s.strip() for s in passages.SENTENCE_END.split(body.replace("\n", " ")) if s.strip()]
    keep: set[int] = set()
    for i, sentence in enumerate(sentences):
        if not passages.company_candidates(sentence):
            continue
        if not (passages.WEAK.search(sentence) or passages.STRONG.search(sentence)):
            # A name with no relationship language in its own sentence still
            # counts if a neighbour supplies it.
            window = " ".join(sentences[max(0, i - 1) : i + 2])
            if not (passages.WEAK.search(window) or passages.STRONG.search(window)):
                continue
        keep.update(range(max(0, i - CONTEXT_SENTENCES), min(len(sentences), i + CONTEXT_SENTENCES + 1)))

    out, previous = [], None
    for i in sorted(keep):
        if previous is not None and i > previous + 1:
            out.append("   …")
        out.append(sentences[i])
        previous = i
    return "\n".join(out)


if __name__ == "__main__":
    manifest = json.loads((DIR / "_manifest.json").read_text())
    total_in = total_out = 0
    for entry in manifest:
        body = (DIR / f"{entry['accession']}.txt").read_text(encoding="utf-8")
        view = condense(body)
        (DIR / f"{entry['accession']}.view.txt").write_text(view, encoding="utf-8")
        entry["view_chars"] = len(view)
        total_in += len(body)
        total_out += len(view)
    (DIR / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} filings   {total_in:,} chars -> {total_out:,} "
          f"({total_out / total_in:.0%})")
    empty = [e["accession"] for e in manifest if e["view_chars"] < 200]
    print(f"near-empty views: {len(empty)}  {empty}")
