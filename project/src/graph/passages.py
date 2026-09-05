"""Cut a 10-K down to the passages that can name an economic link.

A 10-K runs 250k characters, of which the relationship language occupies a few
thousand. Sending the whole document to a model would cost roughly twenty times
as much and extract worse: the relevant sentences get buried among lease
schedules and accounting policy, and recall drops.

The filter is deliberately generous on recall and cheap to run. Anything that
survives goes to the model, which does the actual judgement about whether a
sentence describes a real counterparty.
"""

from __future__ import annotations

import re

import pandas as pd

# Sections where counterparties are named. Item 1 carries the business
# description, 1A the dependency risks, and 7 the concentration discussion.
SECTION_STARTS = [
    r"item\s*1\s*[\.\-–—:]?\s*business",
    r"item\s*1a\s*[\.\-–—:]?\s*risk\s*factors",
    r"item\s*7\s*[\.\-–—:]?\s*management",
]

# Phrases that actually precede a named counterparty. Weighted: a concentration
# disclosure is far more likely to name one than a generic mention of customers.
STRONG = re.compile(
    r"(accounted for|represented|comprised)\s+(approximately\s+)?\d{1,2}(\.\d+)?%"
    r"|(\d{1,2}(\.\d+)?%\s+of\s+(our\s+|the\s+company'?s?\s+)?(total\s+)?(net\s+)?(revenue|sales))"
    r"|largest\s+customer|major\s+customers?|significant\s+customers?|principal\s+customers?"
    r"|key\s+customers?|primary\s+customers?"
    r"|sole[\-\s]source|single[\-\s]source|sole\s+supplier"
    r"|contract\s+manufacturer|foundry\s+partner"
    r"|customer\s+concentration|concentration\s+of\s+credit\s+risk",
    re.I,
)

# Outsourced-manufacturing vocabulary belongs here and was missing. NVIDIA names
# six of its assembly and test suppliers in a sentence built on "subcontractors"
# and no other relationship word -- Advanced Semiconductor Engineering, Hon Hai,
# Siliconware and three more. Without these terms the sentence scores zero.
WEAK = re.compile(
    r"\bcustomers?\b|\bsuppliers?\b|\bvendors?\b|\bresellers?\b|\bdistributors?\b"
    r"|\blicensees?\b|\bpartners?\b|\bpurchases?\s+from\b|\bsells?\s+to\b"
    r"|\bsupplied\s+by\b|\bdepend(s|ent)?\s+(up)?on\b|\brel(y|ies|iant)\s+(up)?on\b"
    r"|\bsubcontractors?\b|\bcontract\s+manufactur|\bfoundr(y|ies)\b|\boutsourc"
    r"|\bassembl(y|ers?)\b|\bfabricat|\bmanufactured?\s+(for|by)\b|\bsource[ds]?\s+from\b",
    re.I,
)

# Candidate company names: runs of capitalised tokens, optionally closed by a
# corporate suffix. Deliberately over-inclusive -- the model decides what is
# really a counterparty; this only has to get the passage in front of it.
CAPITALISED_RUN = re.compile(
    r"\b[A-Z][a-zA-Z&.\-']*(?:\s+(?:[A-Z][a-zA-Z&.\-']*|of|and|de|van|der))*\b"
)

CORPORATE_SUFFIX = re.compile(
    r"\b(inc|corp|corporation|company|co|ltd|limited|llc|lp|plc|nv|ag|sa|gmbh|ab|as"
    r"|holdings?|group|technologies|technology|systems|semiconductor|electronics"
    r"|industries|laboratories|labs|pharmaceuticals|motors|networks|solutions"
    r"|partners|international|stores|bank|airlines)\b\.?",
    re.I,
)

# Capitalised tokens that are not companies. Sentence-initial words and section
# furniture dominate the raw matches, and without this the scorer cannot tell a
# passage naming Samsung from one beginning "However, our customers...".
NOT_A_COMPANY = {
    "the", "a", "an", "we", "our", "us", "it", "its", "this", "these", "those", "there",
    "in", "on", "at", "for", "to", "from", "by", "with", "as", "if", "and", "or", "but",
    "however", "additionally", "furthermore", "moreover", "although", "while", "because",
    "item", "note", "notes", "part", "table", "contents", "form", "annual", "report",
    "risk", "factors", "business", "management", "discussion", "analysis", "overview",
    "company", "corporation", "inc", "llc", "ltd",
    "we", "customers", "customer", "suppliers", "supplier", "oem", "oems", "odm", "odms",
    "tier", "aib", "aibs", "gaap", "sec", "fasb", "asc", "ifrs", "u.s", "us", "usa",
    "united", "states", "america", "european", "union", "china", "japan", "korea",
    "taiwan", "india", "canada", "mexico", "germany", "france", "brazil",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "fiscal", "year", "quarter",
    "no", "yes", "none", "not", "all", "any", "each", "such", "certain", "other",
}

# Generic plurals: dense in boilerplate, never a counterparty.
GENERIC = re.compile(
    r"\b(customers|suppliers|vendors|resellers|distributors|retailers|oems?|odms?"
    r"|manufacturers|partners|clients|third parties|end users|consumers)\b",
    re.I,
)

PROXIMITY_CHARS = 300

# Below this share of the document, a section match is treated as spurious.
MIN_SECTION_SHARE = 0.15

BOILERPLATE = re.compile(
    r"forward[\-\s]looking statements|table of contents|see note \d|incorporated by reference",
    re.I,
)


def sections(text: str) -> str:
    """Keep Items 1, 1A and 7; fall back to the whole document if unparsed.

    Item headings appear twice in most filings -- once in the table of contents,
    once at the section itself -- so the *last* match is taken as the real start.
    """
    lowered = text.lower()
    spans: list[tuple[int, int]] = []
    for pattern in SECTION_STARTS:
        matches = list(re.finditer(pattern, lowered))
        if not matches:
            continue
        start = matches[-1].start()
        spans.append((start, min(start + 400_000, len(text))))

    if not spans:
        return text

    spans.sort()
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    kept = "\n\n".join(text[start:end] for start, end in merged)

    # A heading match that captures almost nothing is a cross-reference ("see
    # Item 1A"), not the section itself. Filing layouts vary too much to detect
    # this case by case, so an implausibly small span falls back to the whole
    # document: the paragraph scorer is selective enough to absorb the extra.
    if len(kept) < MIN_SECTION_SHARE * len(text):
        return text
    return kept


SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def paragraphs(text: str, min_chars: int = 120, target_chars: int = 1200) -> list[str]:
    """Split into scoreable units without trusting the filer's line breaks.

    Blank-line splitting looked obvious and failed on roughly a third of filers:
    Walmart's 10-K renders as a single 65,000-character block with no blank line
    in it, so every unit fell outside any sane length bound and nothing was
    selected. Sentence windows are independent of how a filer's HTML happens to
    wrap, which makes the unit size comparable across the whole corpus -- and the
    score below normalises by length, so units have to be comparable for the
    threshold to mean the same thing everywhere.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    sentences = SENTENCE_END.split(flat)

    out, buffer = [], ""
    for sentence in sentences:
        buffer = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(buffer) >= target_chars:
            out.append(buffer)
            buffer = ""
    if buffer:
        out.append(buffer)

    return [p for p in out if len(p) >= min_chars and not BOILERPLATE.search(p)]


def company_candidates(paragraph: str) -> list[tuple[int, str]]:
    """Positions and text of capitalised runs that could be company names."""
    out = []
    for match in CAPITALISED_RUN.finditer(paragraph):
        text = match.group().strip()
        if len(text) < 3:
            continue
        tokens = [t for t in re.split(r"\s+", text) if t]
        meaningful = [t for t in tokens if t.strip(".,&-'").lower() not in NOT_A_COMPANY]
        if not meaningful:
            continue
        # A single capitalised word at a sentence boundary is usually just a
        # sentence start; require either more tokens or a corporate suffix.
        if len(meaningful) == 1 and not CORPORATE_SUFFIX.search(text):
            before = paragraph[max(0, match.start() - 2) : match.start()]
            if match.start() == 0 or before.strip().endswith((".", "!", "?", ":", ";")):
                continue
        out.append((match.start(), text))
    return out


def score(paragraph: str) -> float:
    """How likely this paragraph names a real counterparty.

    Scored on *named entities near relationship language*, not on keyword
    density. Density was the first attempt and it selected exactly the wrong
    passages: generic risk-factor prose is saturated with "customers",
    "suppliers" and "OEMs" while naming nobody, so it outscored the paragraph
    that actually named the foundry. Every counterparty of interest in the NVIDIA
    filing tested this way -- Taiwan Semiconductor, Samsung, Hon Hai -- was
    ranked out of the budget by boilerplate.
    """
    candidates = company_candidates(paragraph)
    if not candidates:
        return 0.0

    keyword_spans = [m.start() for m in WEAK.finditer(paragraph)]
    strong_spans = [m.start() for m in STRONG.finditer(paragraph)]
    if not keyword_spans and not strong_spans:
        return 0.0

    total = 0.0
    for position, _ in candidates:
        near_weak = any(abs(position - k) <= PROXIMITY_CHARS for k in keyword_spans)
        near_strong = any(abs(position - k) <= PROXIMITY_CHARS for k in strong_spans)
        if near_strong:
            total += 3.0
        elif near_weak:
            total += 1.0

    # Boilerplate tax: passages that are mostly generic plurals are describing a
    # category, not a counterparty.
    generic = len(GENERIC.findall(paragraph))
    return total / (1.0 + 0.5 * generic)


def select(text: str, budget_chars: int = 14000, min_score: float = 0.5) -> list[str]:
    """The highest-scoring passages, in document order, within a char budget."""
    candidates = [(score(p), i, p) for i, p in enumerate(paragraphs(sections(text)))]
    candidates = [c for c in candidates if c[0] >= min_score]
    candidates.sort(key=lambda c: -c[0])

    chosen, used = [], 0
    for value, index, paragraph in candidates:
        if used + len(paragraph) > budget_chars:
            continue
        chosen.append((index, paragraph))
        used += len(paragraph)
    return [paragraph for _, paragraph in sorted(chosen)]


def summarize(text: str) -> dict:
    """Diagnostics for one filing, for tuning the filter without an API call."""
    selected = select(text)
    return {
        "chars_in": len(text),
        "chars_sections": len(sections(text)),
        "n_paragraphs": len(paragraphs(sections(text))),
        "n_selected": len(selected),
        "chars_selected": sum(len(p) for p in selected),
        "compression": round(len(text) / max(sum(len(p) for p in selected), 1), 1),
    }


def frame(texts: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"key": k, **summarize(v)} for k, v in texts.items()])


def prescreen(selected: list[str], lookup, resolver, source_ticker: str) -> tuple[bool, int]:
    """Can this filing possibly yield a usable edge?

    A claim only becomes an edge if its counterparty resolves to a listed
    ticker. If no candidate name sits near relationship language, the model has
    nothing to find and the call is fourteen seconds spent to learn nothing.

    Proximity is the part that matters. A first version accepted any resolving
    name anywhere in the selected passages and skipped only 4% of filings --
    every filing mentions *some* capitalised name that happens to collide with a
    ticker. Requiring the name to sit within `PROXIMITY_CHARS` of a customer or
    supplier phrase is the same condition under which a real relationship would
    actually be described.

    Still generous: one qualifying candidate passes the whole filing, and a
    quantified concentration disclosure passes it even without one, since the
    model reads names the regex is not built to catch.
    """
    hits = 0
    for paragraph in selected:
        keywords = [m.start() for m in WEAK.finditer(paragraph)]
        keywords += [m.start() for m in STRONG.finditer(paragraph)]
        if not keywords:
            continue
        for position, name in company_candidates(paragraph):
            if not any(abs(position - k) <= PROXIMITY_CHARS for k in keywords):
                continue
            ticker, _tier = resolver(name, lookup)
            if ticker and ticker != source_ticker:
                hits += 1
    if hits:
        return True, hits
    return any(STRONG.search(p) for p in selected), 0
