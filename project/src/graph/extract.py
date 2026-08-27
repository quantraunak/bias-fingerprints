"""Extract named economic links from filing passages with Claude.

The output of this stage is a set of *claims*: "this filer, in this filing,
described this named counterparty as a customer." Whether the counterparty is in
the investable universe, and whether the claim becomes a graph edge, is decided
later in `resolve.py` and `build.py`. Keeping extraction free of that judgement
means the expensive stage runs once and the cheap stages can be re-run.

Three constraints on the model matter for research integrity:

* **Named counterparties only.** ASC 280 forces disclosure of customers above
  10% of revenue but does not force naming them, so a large share of filings say
  "Customer A accounted for 14%". Those are real economic facts and useless
  here -- an unnamed node cannot be joined to a return series. They are dropped
  at extraction rather than silently resolved to something plausible.
* **Verbatim evidence.** Every claim carries the sentence it came from, checked
  as a literal substring of the input. A claim whose evidence is not in the
  source is a fabrication and is discarded without being counted against
  coverage, which turns hallucination into a measurable rate instead of a silent
  contaminant.
* **Direction from the filer's perspective.** `customer` means the counterparty
  buys from the filer; `supplier` means it sells to the filer. Getting this
  backwards inverts the signal, so it is stated explicitly and spot-checked.
"""

from __future__ import annotations

import json
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

MODEL = "claude-opus-5"
MAX_TOKENS = 4096

Relation = Literal["customer", "supplier", "partner", "competitor"]


class Link(BaseModel):
    counterparty: str = Field(description="Company or organisation name exactly as written in the passage.")
    relation: Relation = Field(
        description=(
            "From the filing company's perspective. 'customer': the counterparty buys from the filer. "
            "'supplier': the counterparty sells to or manufactures for the filer. "
            "'partner': joint venture, licensing, or distribution alliance with no clear buy/sell direction. "
            "'competitor': named as a competitor."
        )
    )
    revenue_pct: float | None = Field(
        default=None,
        description="Percent of the filer's revenue attributed to this counterparty, if the passage states a number. Otherwise null.",
    )
    evidence: str = Field(description="The verbatim sentence from the passage that supports this link. Must appear in the passage exactly.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="high: the passage names the counterparty and the relationship explicitly. medium: the relationship is clear but described loosely. low: inferred."
    )


class Extraction(BaseModel):
    links: list[Link]


SYSTEM = """You extract named economic relationships between companies from SEC 10-K filings.

You will be given passages from one company's 10-K. Identify every relationship the filing describes between that company and a SPECIFICALLY NAMED other organisation.

Rules:

1. The counterparty must be a REAL, NAMED organisation — "Walmart", "Taiwan Semiconductor Manufacturing Company", "the U.S. Department of Defense".
   REJECT anonymous or generic references. Do not emit a link for: "Customer A", "our largest customer", "certain customers", "a major distributor", "retailers", "OEMs", "third-party manufacturers", "government agencies", "two customers accounted for 30%".
   If a passage discloses a concentration but does not name the counterparty, emit nothing for it.

2. `evidence` must be copied VERBATIM from the passage — an exact substring, not a paraphrase. If you cannot quote it exactly, do not emit the link.

3. `relation` is from the FILING COMPANY's perspective:
   - customer  — the named counterparty BUYS FROM the filer
   - supplier  — the named counterparty SELLS TO or MANUFACTURES FOR the filer
   - partner   — alliance, joint venture, licensing, or distribution with no clear buy/sell direction
   - competitor — named as a competitor

4. Only relationships involving the FILING COMPANY. Ignore relationships described between two other parties.

5. Do not include the filer's own subsidiaries, brands, or product names as counterparties.

6. `revenue_pct` only when the passage states a specific percentage tied to that named counterparty. Otherwise null.

Return every qualifying link. If there are none, return an empty list."""


def user_prompt(ticker: str, company: str, filed: str, passages: list[str]) -> str:
    body = "\n\n---\n\n".join(passages)
    return (
        f"Filing company: {company} (ticker {ticker})\n"
        f"Form: 10-K, filed {filed}\n\n"
        f"Passages:\n\n{body}"
    )


def build_params(ticker: str, company: str, filed: str, passages: list[str]) -> dict:
    """Message-create params, usable directly or inside a batch request."""
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"format": {"type": "json_schema", "schema": Extraction.model_json_schema()}},
        "messages": [{"role": "user", "content": user_prompt(ticker, company, filed, passages)}],
    }


# ------------------------------------------------------------------ validate

WHITESPACE = re.compile(r"\s+")

# Anonymous references the model is told to reject. Checked again here because a
# prompt rule is a request, not a guarantee, and this is cheap.
ANONYMOUS = re.compile(
    r"^(customer|supplier|client|vendor|distributor|reseller)\s*[a-z0-9]?$"
    r"|^(a|an|our|the|certain|two|three|several|various|other|major|largest|significant|key|primary)\b"
    r"|^(customers|suppliers|retailers|oems|resellers|distributors|wholesalers|government agencies|third parties)$",
    re.I,
)


PUNCT = re.compile(r"[^\w\s]")

# A faithful quote may elide, but it may not invent. These two thresholds
# separate the cases: every token has to come from the source, and some span of
# it has to be contiguous.
EVIDENCE_TOKEN_COVERAGE = 0.90
EVIDENCE_MIN_RUN = 8


def _normalise(text: str) -> str:
    text = PUNCT.sub(" ", text.lower())
    return WHITESPACE.sub(" ", text).strip()


def _longest_run(tokens: list[str], haystack: str) -> int:
    """Longest contiguous span of `tokens` appearing verbatim in the source."""
    best = 0
    for start in range(len(tokens)):
        if len(tokens) - start <= best:
            break
        for end in range(len(tokens), start + best, -1):
            if f" {' '.join(tokens[start:end])} " in haystack:
                best = end - start
                break
    return best


def _grounded(evidence: str, haystack: str, haystack_tokens: set[str]) -> bool:
    """Is this quote actually in the source, allowing for faithful elision?

    An exact substring test looked like the strict, safe choice and was wrong in
    a way that cost real edges. Filings write counterparties with parenthetical
    short forms -- `Boeing Corporation ("Boeing"), Northrop Grumman Corporation
    ("Northrop")` -- and the model quotes them without the parentheticals. That
    is a faithful quote of a real sentence but not a substring of it, so
    Frequency Electronics lost all three of its correctly extracted customers
    and the loss was counted as a hallucination.

    N-gram coverage was the second attempt and also failed here: three
    elisions in a twenty-six token quote break enough overlapping windows to
    push coverage to 0.50, indistinguishable from invention by that measure.

    Two conditions separate the cases properly. Every token must appear
    somewhere in the source, so nothing can be introduced; and some span must
    match contiguously, so the tokens cannot merely be scattered words
    reassembled into a sentence the filing never contained. Elision satisfies
    both -- the FEIM quote keeps an eleven-token run -- while a fabricated
    sentence fails the second even when it borrows the vocabulary.
    """
    tokens = evidence.split()
    if len(tokens) < EVIDENCE_MIN_RUN:
        return False
    coverage = sum(1 for t in tokens if t in haystack_tokens) / len(tokens)
    if coverage < EVIDENCE_TOKEN_COVERAGE:
        return False
    required = min(EVIDENCE_MIN_RUN, len(tokens))
    return _longest_run(tokens, haystack) >= required


def validate(links: list[dict], passages: list[str]) -> tuple[list[dict], dict]:
    """Keep links whose evidence really is in the source and whose name is real.

    Returns the surviving links and a per-filing tally, so hallucination and
    anonymous-reference rates are measured rather than assumed.
    """
    haystack = f" {_normalise(chr(10).join(passages))} "
    haystack_tokens = set(haystack.split())
    kept, rejected = [], {"no_evidence": 0, "anonymous": 0, "empty_name": 0}

    for link in links:
        name = (link.get("counterparty") or "").strip()
        if not name or len(name) < 2:
            rejected["empty_name"] += 1
            continue
        if ANONYMOUS.match(name):
            rejected["anonymous"] += 1
            continue
        evidence = _normalise(link.get("evidence") or "")
        if len(evidence) < 20 or not _grounded(evidence, haystack, haystack_tokens):
            rejected["no_evidence"] += 1
            continue
        kept.append(link)

    return kept, {"kept": len(kept), **rejected}


def parse_response(text: str) -> list[dict]:
    """Structured outputs return JSON text; be tolerant of an empty response."""
    if not text or not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    links = payload.get("links", []) if isinstance(payload, dict) else []
    return [link for link in links if isinstance(link, dict)]


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Claims table: one row per extracted link, keyed on the filing."""
    if not records:
        return pd.DataFrame(
            columns=["ticker", "accession", "filed", "counterparty", "relation", "revenue_pct", "confidence", "evidence"]
        )
    frame = pd.DataFrame(records)
    frame["filed"] = pd.to_datetime(frame["filed"])
    return frame.sort_values(["filed", "ticker"]).reset_index(drop=True)
