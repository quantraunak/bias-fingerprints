"""Construct documents whose ground truth is known because we planted it.

Measuring how extraction recall depends on the number of items to be extracted
needs documents at many cardinalities with reliable labels. Annotating them is
weeks of human work, and the corpus only has two annotated filings above nine
items -- which is why the observation that motivated this question rests on two
filings that disagree.

Building the documents instead removes the annotator entirely. Every sentence
here is real filing prose: the relationship sentences are evidence quotes that
already passed the verbatim-substring check against their source filings, and
the filler is drawn from filings the extractor read and returned zero links for.
A document is then a shuffle of `n` relationship sentences into filler, padded
to a fixed length, and its label is the list of counterparties that were
planted.

That buys the thing observational data cannot give: **control**. Cardinality
varies while total length, sentence count and item positions are held fixed or
varied deliberately, so an effect of cardinality cannot be an effect of length
wearing a disguise.

What it costs is external validity. A constructed document is not a 10-K: the
relationship sentences come from many different filers, so the document has no
consistent narrator, and a model may behave differently on real prose. The
curve measured here has to be anchored against the annotated real filings
before it can be claimed to describe real extraction, and `anchor()` is what
does that.
"""

from __future__ import annotations

import random
import re

import pandas as pd

# A planted sentence has to name its counterparty and stand alone. Too short and
# it carries no relationship language; too long and it is a paragraph that may
# smuggle in a second counterparty and corrupt the label.
MIN_EVIDENCE = 60
MAX_EVIDENCE = 400
MIN_FILLER = 80
MAX_FILLER = 300

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")

# A transplanted sentence is only valid ground truth if the relationship it
# describes belongs to the *filer*. "Bill Me Later's operations depend on
# lending services provided by CIT Bank" is a true sentence about a real
# relationship, and once it sits in a document attributed to some other company
# the relationship is between two third parties -- which the extraction prompt
# explicitly instructs the model to ignore. Labelling it as a planted item
# scores a correct refusal as a miss. Requiring first-person framing is what
# makes the transplant preserve the label.
FIRST_PERSON = re.compile(r"\b(we|our|us)\b|\bthe Company\b", re.I)

# Categories the extraction prompt rejects by rule (E3, A4). A sentence
# announcing an acquisition is not a supplier relationship, so a model that
# declines to extract it is right and the label would be wrong.
EXCLUDED_KIND = re.compile(
    r"\bacquir\w*|\bacquisition\b|\bmerger\b|\bdivestiture\b"
    r"|\bcompetitors?\b|\bpurchased the (intellectual|assets)",
    re.I,
)

# Filler must be free of anything that reads as a counterparty, or it plants
# items the label does not know about and inflates apparent false positives.
# 53% of raw filing sentences carry a capitalised multi-word name.
CAPITALISED_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def evidence_pool(claims: pd.DataFrame) -> pd.DataFrame:
    """Relationship sentences usable as plantable items with known labels.

    Requires the counterparty's leading token to appear in its own evidence
    quote. A claim whose quote does not name the counterparty cannot serve as
    ground truth for that counterparty, however good the claim is.
    """
    frame = claims.dropna(subset=["evidence", "counterparty"]).copy()
    frame["evidence"] = frame["evidence"].str.strip()
    frame = frame[frame["evidence"].str.len().between(MIN_EVIDENCE, MAX_EVIDENCE)]
    named = frame.apply(
        lambda r: str(r.counterparty).split()[0].lower() in str(r.evidence).lower(),
        axis=1,
    )
    frame = frame[named].drop_duplicates("evidence")
    # The relationship has to be the filer's own, and of a kind the prompt
    # accepts. Both were found empirically: without them the model correctly
    # refused three of the first four single-item documents and the curve was
    # measuring the construction rather than the model.
    frame = frame[frame.evidence.map(lambda e: bool(FIRST_PERSON.search(e)))]
    frame = frame[~frame.evidence.map(lambda e: bool(EXCLUDED_KIND.search(e)))]
    # One sentence per counterparty: repeated mentions of the same company would
    # make the planted count and the distinct-counterparty count disagree.
    return frame.drop_duplicates("counterparty").reset_index(drop=True)


def filler_pool(texts: list[str]) -> list[str]:
    """Sentences from filings the extractor read and found nothing in.

    Real prose, and empirically free of extractable relationships -- the model
    itself is the filter, which is a stronger guarantee than a keyword screen.
    """
    out = []
    for text in texts:
        for sentence in SENTENCE_SPLIT.split(re.sub(r"\s+", " ", text)):
            sentence = sentence.strip()
            if not (MIN_FILLER <= len(sentence) <= MAX_FILLER):
                continue
            # A capitalised multi-word name in filler is an unlabelled item.
            if CAPITALISED_NAME.search(sentence):
                continue
            out.append(sentence)
    seen, unique = set(), []
    for sentence in out:
        if sentence not in seen:
            seen.add(sentence)
            unique.append(sentence)
    return unique


def build(
    n_items: int,
    evidence: pd.DataFrame,
    filler: list[str],
    rng: random.Random,
    target_chars: int = 4000,
    spread: str = "uniform",
) -> tuple[str, list[str]]:
    """One document with `n_items` planted relationships, and its label.

    `target_chars` is held constant across cardinalities so that a difference in
    recall between n=2 and n=20 cannot be attributed to document length. Filler
    absorbs the difference: a document with more planted sentences carries
    proportionally less filler.

    `spread` controls where the items sit. 'uniform' scatters them evenly,
    'front' and 'back' concentrate them in the first or last third. Position is
    the axis the lost-in-the-middle literature varies, so keeping it a parameter
    lets the two effects be separated rather than confounded.
    """
    if n_items > len(evidence):
        raise ValueError(f"asked for {n_items} items, pool holds {len(evidence)}")

    picked = evidence.sample(n=n_items, random_state=rng.randrange(1 << 30))
    items = list(picked.evidence)
    labels = list(picked.counterparty)

    used = sum(len(s) for s in items)
    body = list(items)
    pad = []
    while used + sum(len(s) for s in pad) < target_chars and filler:
        pad.append(rng.choice(filler))

    if spread == "uniform":
        slots = _uniform_slots(len(body), len(pad), rng)
    elif spread in ("front", "back"):
        slots = _clustered_slots(len(body), len(pad), spread, rng)
    else:
        raise ValueError(f"unknown spread {spread!r}")

    document, item_iter, pad_iter = [], iter(body), iter(pad)
    for is_item in slots:
        document.append(next(item_iter) if is_item else next(pad_iter))
    return " ".join(document), labels


def _uniform_slots(n_items: int, n_pad: int, rng: random.Random) -> list[bool]:
    slots = [True] * n_items + [False] * n_pad
    rng.shuffle(slots)
    return slots


def _clustered_slots(n_items: int, n_pad: int, where: str, rng: random.Random) -> list[bool]:
    total = n_items + n_pad
    third = max(1, total // 3)
    zone = list(range(third)) if where == "front" else list(range(total - third, total))
    positions = set(rng.sample(zone, min(n_items, len(zone))))
    # Overflow when the zone cannot hold every item; the remainder goes adjacent
    # rather than being dropped, which would silently change the cardinality.
    spill = n_items - len(positions)
    if spill > 0:
        rest = [i for i in range(total) if i not in positions]
        positions.update(rng.sample(rest, spill))
    return [i in positions for i in range(total)]


def recovered(extracted: list[str], planted: list[str], lookup, resolve_one) -> int:
    """How many planted counterparties the model returned.

    Matching is on resolved ticker where possible and normalised name
    otherwise, matching how `evaluate.score` compares predictions to gold, so
    the synthetic numbers and the real-filing numbers mean the same thing.
    """
    def key(name: str):
        ticker, _ = resolve_one(name, lookup)
        return ticker if ticker else _normalise(name)

    got = {key(name) for name in extracted if str(name).strip()}
    want = {key(name) for name in planted}
    return len(got & want)


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).strip()
