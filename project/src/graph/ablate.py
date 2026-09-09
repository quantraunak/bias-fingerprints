"""Vary how many relationships a real filing discloses, by removing them.

The synthetic-document approach failed for a reason worth recording: shuffling
real sentences from many filers into one document produces text that is not a
10-K, and the model does not read it like one. At a single planted item, recall
on constructed documents was 1/6 against 0.66-1.00 for the same model on real
filings. The construction dominated whatever cardinality effect might exist.

Ablation keeps the document real. Start from one filing, locate the sentences
carrying each disclosed relationship, and delete a subset -- replacing each with
neutral prose from the *same* filing so the length, the voice and the section
structure are preserved. Every version is the real document minus some
disclosures, which is a thing that could have existed: a filer with fewer
relationships to report.

Ground truth is the set of relationships left in. That is exact by
construction, and no annotation is required beyond knowing which sentences
carried a relationship in the first place.

Two limits, stated because they bound the claim:

* **The reference set may be model-derived.** Using one model's extractions as
  the item set makes a measurement of *that* model circular. It is sound for
  measuring a different model, and the gold-annotated filings provide a
  non-circular anchor at lower cardinalities.
* **Replacement prose is not neutral in the way filler is.** It comes from the
  same filing, so it may carry topic and tone that a deleted sentence did not.
  It is chosen to contain no resolvable company name, which is the property
  that matters, but it is not information-free.
"""

from __future__ import annotations

import random
import re

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")

# Replacement prose must not introduce a counterparty of its own, or the
# ablation would delete one item and add another.
MIN_REPLACEMENT = 60
MAX_REPLACEMENT = 400


def locate(passage_list: list[str], evidence: str) -> tuple[int, str] | None:
    """Find (paragraph index, sentence) containing an evidence quote.

    The quote may span a sentence boundary or sit inside a longer sentence, so
    the containing sentence is returned rather than the quote itself: deleting
    a fragment would leave an ungrammatical remainder, which is a different
    edit from removing a disclosure.
    """
    for index, paragraph in enumerate(passage_list):
        if evidence not in paragraph:
            continue
        for sentence in SENTENCE_SPLIT.split(paragraph):
            if evidence in sentence or sentence in evidence:
                return index, sentence
        return index, evidence
    return None


def neutral_sentences(passage_list: list[str], carriers: set[str],
                      has_company) -> list[str]:
    """Sentences from this filing that carry no counterparty.

    `has_company` is injected rather than imported so the caller supplies the
    same resolver the rest of the pipeline uses, and so this module stays
    testable without a registry.
    """
    out = []
    for paragraph in passage_list:
        for sentence in SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not (MIN_REPLACEMENT <= len(sentence) <= MAX_REPLACEMENT):
                continue
            if sentence in carriers or has_company(sentence):
                continue
            out.append(sentence)
    seen, unique = set(), []
    for sentence in out:
        if sentence not in seen:
            seen.add(sentence)
            unique.append(sentence)
    return unique


def ablate(passage_list: list[str], items: list[dict], keep: int,
           replacements: list[str], rng: random.Random) -> tuple[list[str], list[str]]:
    """Keep `keep` of the disclosed relationships, remove the rest.

    `items` are dicts with `counterparty` and `evidence`. Returns the modified
    passages and the counterparties still disclosed.

    Removal replaces each carrier sentence with a same-filing sentence of
    similar length rather than deleting it outright, so total length stays
    within a few percent across the whole range and a cardinality effect cannot
    be a length effect.
    """
    if keep > len(items):
        raise ValueError(f"asked to keep {keep} of {len(items)} items")

    located = []
    for item in items:
        found = locate(passage_list, item["evidence"])
        if found is not None:
            located.append((item, found))
    if keep > len(located):
        raise ValueError(f"only {len(located)} of {len(items)} items are locatable")

    order = list(range(len(located)))
    rng.shuffle(order)
    kept_indices = set(order[:keep])

    out = list(passage_list)
    retained = []
    pool = list(replacements)
    rng.shuffle(pool)

    for position, (item, (index, sentence)) in enumerate(located):
        if position in kept_indices:
            retained.append(item["counterparty"])
            continue
        if not pool:
            # Nothing left to swap in; drop the sentence rather than leave the
            # relationship in place, and accept the small length loss.
            out[index] = out[index].replace(sentence, "", 1)
            continue
        # Prefer a replacement of similar length so the edit is length-neutral.
        replacement = min(pool, key=lambda s: abs(len(s) - len(sentence)))
        pool.remove(replacement)
        out[index] = out[index].replace(sentence, replacement, 1)

    return out, retained
