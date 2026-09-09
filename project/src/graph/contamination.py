"""Measure how much of an extraction came from the document rather than the model.

An LLM reading a 2013 filing was trained on text through 2025. When it reports
that the filer buys from Taiwan Semiconductor, there is no way to tell from the
output whether it read that in the passage or recalled it from training. Both
produce a well-formed claim with a plausible evidence quote, and if the claim
happens to be true the usual quality metrics score it as a success.

That matters for anyone building a *historical* dataset with a model, because
the contamination is not random. A model knows the famous relationships and the
recent ones, so leakage concentrates on large, well-covered firms -- which
biases the density of the resulting graph in exactly the direction that would
confound a cross-sectional study built on it.

The test here is an ablation. Replace the counterparty's identifying tokens with
invented ones, keeping the industry words and the sentence intact, and re-run
the extraction. Three outcomes:

* **read** -- the model reports the invented name. It was reading the document.
* **silent** -- the model reports nothing. It needed the real name to recognise
  that a relationship was being described, which is a softer form of the same
  dependence.
* **phantom** -- the model reports the *real* name, which no longer appears in
  the text it was shown. This is unambiguous: the name came from the weights.

Only the distinctive tokens are swapped. "Taiwan Semiconductor Manufacturing
Company" becomes "Vantessa Semiconductor Manufacturing Company", not
"Vantessa Korlan Bexil Company" -- the second would remove the industry cue as
well as the identity, and a recall drop could then be blamed on an incoherent
sentence rather than on lost identity. `resolve.distinctive` already computes
which tokens carry identity, using registry document frequency, so the split is
measured rather than hand-listed.
"""

from __future__ import annotations

import random
import re

from src.graph import resolve

# Invented roots are built from these rather than sampled from a word list, so
# no generated name can accidentally be a real company. Every candidate is
# still checked against the registry before use.
ONSETS = ["v", "k", "z", "th", "br", "dr", "fl", "gl", "kr", "pl", "qu", "sv", "tr", "vr"]
NUCLEI = ["a", "e", "i", "o", "u", "ae", "ea", "io", "ou"]
CODAS = ["n", "l", "r", "s", "th", "ndal", "ric", "mor", "vex", "tan", "dis", "quin"]

WORD = re.compile(r"[A-Za-z][A-Za-z&.'\-]*")


MIN_INVENTED = 5


def invent(rng: random.Random) -> str:
    """A pronounceable token that is not a word and not a company.

    Kept at least `MIN_INVENTED` characters: a short invention like "Qual"
    reads as a clipped form of a real company and reintroduces the identity
    the swap exists to remove.
    """
    while True:
        token = rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)
        if len(token) >= MIN_INVENTED:
            return token.capitalize()


def mint_alias(name: str, lookup: dict, rng: random.Random, attempts: int = 12) -> str | None:
    """An invented stand-in for `name` with its identifying tokens replaced.

    Returns None when the name has no distinctive token to swap -- a
    counterparty called only "the Company" cannot be de-identified, and
    pretending otherwise would put a meaningless row in the results.
    """
    tokens = WORD.findall(name)
    if not tokens:
        return None

    carriers = resolve.distinctive({resolve.normalize(t) for t in tokens if t}, lookup)
    carriers.discard("")
    if not carriers:
        return None

    for _ in range(attempts):
        swapped = [
            invent(rng) if resolve.normalize(t) in carriers else t
            for t in tokens
        ]
        candidate = " ".join(swapped)
        # An invented name that happens to resolve would silently reintroduce a
        # real identity and invert the whole measurement.
        ticker, _tier = resolve.resolve_one(candidate, lookup)
        if ticker is None:
            return candidate
    return None


def surface_forms(name: str, lookup: dict) -> list[str]:
    """Ways the name may appear in prose, longest first.

    A filing writes "Taiwan Semiconductor Manufacturing Company" once and
    "Taiwan Semiconductor" or "TSMC" thereafter. Replacing only the full form
    would leave the identity in the text and understate leakage.
    """
    tokens = WORD.findall(name)
    if not tokens:
        return []
    carriers = resolve.distinctive({resolve.normalize(t) for t in tokens if resolve.normalize(t)},
                                  lookup)
    carriers.discard("")

    def identifies(form: str) -> bool:
        """A form is only safe to replace if it carries an identifying token.

        Without this, a suffix word normalises to the empty string, counts as
        distinctive because the registry has never seen it, and every
        occurrence of "Company" in the filing gets rewritten.
        """
        return any(resolve.normalize(t) in carriers for t in WORD.findall(form))

    forms = {name}
    if len(tokens) > 1:
        forms.add(" ".join(tokens))
        for cut in range(len(tokens) - 1, 1, -1):
            forms.add(" ".join(tokens[:cut]))
    for token in tokens:
        if resolve.normalize(token) in carriers and len(token) > 3:
            forms.add(token)
    kept = [f for f in forms if f.strip() and identifies(f)]

    # The acronym is exempt from `identifies`: it is derived from the whole
    # name, so none of its characters survive normalisation as an identifying
    # token, and the guard above would drop it. "TSMC" is precisely the surface
    # form that has to be replaced -- leaving it in the text leaves the
    # identity in the text, and every leakage number would be understated.
    initials = "".join(t[0] for t in tokens if t and t[0].isupper())
    if len(initials) >= 3:
        kept.append(initials)

    return sorted(set(kept), key=len, reverse=True)


def swap(passages: list[str], name: str, alias: str, lookup: dict) -> tuple[list[str], int]:
    """Replace every surface form of `name` with the matching form of `alias`."""
    alias_tokens = WORD.findall(alias)
    out, replaced = list(passages), 0

    for form in surface_forms(name, lookup):
        form_tokens = WORD.findall(form)
        if not form_tokens:
            continue
        # Map a truncated surface form onto the same slice of the alias, so
        # "Taiwan Semiconductor" and the full name stay mutually consistent.
        if len(form_tokens) <= len(alias_tokens):
            replacement = " ".join(alias_tokens[: len(form_tokens)])
        else:
            replacement = " ".join(alias_tokens)
        # Multi-token forms are distinctive enough to match case-insensitively.
        # A single token is not: "Industry" is rare in the EDGAR registry, so it
        # reads as identifying, but it is common in filing prose and matching it
        # case-insensitively would rewrite "the semiconductor industry"
        # throughout the document. In a company name it is capitalised; in prose
        # it is not, and that is the distinction worth using.
        flags = re.I if len(form_tokens) > 1 else 0
        pattern = re.compile(
            r"\b" + r"\s+".join(re.escape(t) for t in form_tokens) + r"\b", flags
        )
        for i, passage in enumerate(out):
            out[i], n = pattern.subn(replacement, passage)
            replaced += n
    return out, replaced


def classify(counterparties: list[str], name: str, alias: str, lookup: dict) -> str:
    """How the model responded to a swapped passage.

    'phantom' is the finding: the model named an entity that was not in the
    text it was given. 'read' means it reported the invented name, so it was
    working from the document. 'silent' means it reported neither.
    """
    real_ticker, _ = resolve.resolve_one(name, lookup)
    alias_key = resolve.normalize(alias)
    name_key = resolve.normalize(name)

    for extracted in counterparties:
        key = resolve.normalize(extracted)
        if key and (key == alias_key or key in alias_key or alias_key in key):
            return "read"
    for extracted in counterparties:
        key = resolve.normalize(extracted)
        if not key:
            continue
        if key == name_key or key in name_key or name_key in key:
            return "phantom"
        if real_ticker is not None and resolve.resolve_one(extracted, lookup)[0] == real_ticker:
            return "phantom"
    return "silent"
