"""Guards on the entity-swap ablation.

The swap is the instrument: every contamination number is a comparison between
extraction on real text and extraction on swapped text, so a defect here does
not produce a wrong answer, it produces a *plausible* wrong answer. Two failure
directions, and they bias in opposite ways:

* **Identity left in the text.** If any surface form of the real name survives
  the swap -- the truncated form, the acronym -- then a model reading it is
  scored as reading the document when it was handed the identity. Leakage is
  understated.
* **Over-replacement.** If a generic word is treated as identifying, the swap
  rewrites unrelated prose. "the semiconductor industry" becoming "the Gloumor
  industry" damages the passage, recall drops for reasons unrelated to
  identity, and leakage is overstated.

The registry is a fixture rather than a live download, so the suite is
deterministic and runs offline.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from src.graph import contamination, resolve


@pytest.fixture
def reference() -> pd.DataFrame:
    rows = [
        ("TSM", "TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD"),
        ("INTC", "INTEL CORP"),
        ("BA", "BOEING CO"),
        ("AMAT", "APPLIED MATERIALS INC /DE"),
        # Filler so "semiconductor" and "manufacturing" carry the document
        # frequency that makes them generic, as they do in the real registry.
        ("SEM1", "AMKOR SEMICONDUCTOR INC"),
        ("SEM2", "TOWER SEMICONDUCTOR LTD"),
        ("SEM3", "ALPHA SEMICONDUCTOR CORP"),
        ("SEM4", "MAGNACHIP SEMICONDUCTOR CORP"),
        ("MFG1", "GLOBAL MANUFACTURING CORP"),
        ("MFG2", "SANMINA MANUFACTURING INC"),
        ("MFG3", "KIMBALL MANUFACTURING CO"),
        ("MFG4", "PRECISION MANUFACTURING GROUP"),
    ]
    frame = pd.DataFrame(rows, columns=["ticker", "title"])
    frame["cik"] = range(1, len(frame) + 1)
    frame["normalized"] = frame["title"].map(resolve.normalize)
    frame["tokens"] = frame["normalized"].map(lambda s: frozenset(s.split()))
    return frame


@pytest.fixture
def lookup(reference):
    return resolve.build_lookup(reference)


@pytest.fixture
def rng():
    return random.Random(7)


TSMC = "Taiwan Semiconductor Manufacturing Company"


# --------------------------------------------------------------------------
# minting a stand-in
# --------------------------------------------------------------------------

def test_a_minted_alias_does_not_resolve_to_a_real_company(lookup, rng):
    """An alias that resolves silently reintroduces the identity being removed."""
    for _ in range(40):
        alias = contamination.mint_alias(TSMC, lookup, rng)
        assert alias is not None
        assert resolve.resolve_one(alias, lookup)[0] is None


def test_industry_words_survive_and_only_identity_is_swapped(lookup, rng):
    """The ablation must remove identity without removing the context cue.

    Swapping "Semiconductor Manufacturing" too would let a recall drop be
    explained by an incoherent sentence rather than by lost identity.
    """
    alias = contamination.mint_alias(TSMC, lookup, rng)
    assert "Semiconductor" in alias and "Manufacturing" in alias
    assert "Taiwan" not in alias


def test_a_name_with_nothing_identifying_cannot_be_swapped(lookup, rng):
    """"the Company" cannot be de-identified, and a row for it would be noise."""
    assert contamination.mint_alias("The Company", lookup, rng) is None
    assert contamination.mint_alias("", lookup, rng) is None


def test_minting_is_reproducible_from_a_seed(lookup):
    first = contamination.mint_alias(TSMC, lookup, random.Random(3))
    second = contamination.mint_alias(TSMC, lookup, random.Random(3))
    assert first == second


def test_invented_tokens_are_long_enough_to_not_read_as_a_real_clip(lookup):
    """A short invention like "Qual" reads as a clipped real company name."""
    generated = {contamination.invent(random.Random(i)) for i in range(60)}
    assert all(len(t) >= contamination.MIN_INVENTED for t in generated)


# --------------------------------------------------------------------------
# surface forms: every way the identity appears in prose
# --------------------------------------------------------------------------

def test_the_acronym_is_a_surface_form(lookup):
    """Filings write the name once and the acronym thereafter.

    The acronym normalises to nothing identifying, so the generic-word guard
    would drop it, and leaving "TSMC" in the text leaves the identity in the
    text.
    """
    assert "TSMC" in contamination.surface_forms(TSMC, lookup)


def test_truncated_forms_are_surface_forms(lookup):
    forms = contamination.surface_forms(TSMC, lookup)
    assert "Taiwan Semiconductor" in forms


def test_a_bare_corporate_suffix_is_never_a_surface_form(lookup):
    """The bug this guards: suffixes normalise to "" and read as distinctive.

    Left unguarded, every occurrence of "Company" in a 10-K is rewritten.
    """
    forms = contamination.surface_forms(TSMC, lookup)
    assert "Company" not in forms and "Corporation" not in forms


def test_forms_are_ordered_longest_first(lookup):
    """Replacing the short form first would strand the rest of the long one."""
    forms = contamination.surface_forms(TSMC, lookup)
    assert forms == sorted(forms, key=len, reverse=True)


# --------------------------------------------------------------------------
# the swap itself
# --------------------------------------------------------------------------

PASSAGE = (
    "We depend on Taiwan Semiconductor Manufacturing Company for wafers. "
    "Taiwan Semiconductor is our sole foundry, and TSMC has served us for years. "
    "The semiconductor industry is cyclical."
)


def test_no_surface_form_of_the_real_name_survives(lookup, rng):
    """The invariant the whole measurement rests on."""
    alias = contamination.mint_alias(TSMC, lookup, rng)
    swapped, count = contamination.swap([PASSAGE], TSMC, alias, lookup)
    text = swapped[0]
    assert count == 3
    for leak in ("Taiwan", "TSMC"):
        assert leak not in text


def test_generic_prose_is_left_alone(lookup, rng):
    """"the semiconductor industry" is not a counterparty and must not move."""
    alias = contamination.mint_alias(TSMC, lookup, rng)
    swapped, _ = contamination.swap([PASSAGE], TSMC, alias, lookup)
    assert "The semiconductor industry is cyclical." in swapped[0]


def test_a_lowercase_common_word_is_not_swapped(lookup, rng):
    """Single-token forms match case-sensitively for exactly this reason.

    "Industry" is rare in the registry and so reads as identifying, but it is
    common in filing prose; a case-insensitive match would rewrite all of it.
    """
    name = "Hon Hai Precision Industry"
    alias = contamination.mint_alias(name, lookup, rng)
    passage = "We use Hon Hai Precision Industry. The electronics industry is cyclical."
    swapped, _ = contamination.swap([passage], name, alias, lookup)
    assert "The electronics industry is cyclical." in swapped[0]


def test_the_swap_is_consistent_across_forms(lookup, rng):
    """The truncated form must map onto the same alias, not a second identity."""
    alias = contamination.mint_alias(TSMC, lookup, rng)
    swapped, _ = contamination.swap([PASSAGE], TSMC, alias, lookup)
    head = alias.split()[0]
    assert swapped[0].count(head) == 3


def test_a_passage_not_naming_the_counterparty_is_untouched(lookup, rng):
    alias = contamination.mint_alias(TSMC, lookup, rng)
    other = ["Our customers are diversified across many end markets."]
    swapped, count = contamination.swap(other, TSMC, alias, lookup)
    assert swapped == other and count == 0


# --------------------------------------------------------------------------
# classification: the three outcomes
# --------------------------------------------------------------------------

def test_reporting_the_invented_name_is_reading(lookup, rng):
    alias = contamination.mint_alias(TSMC, lookup, rng)
    assert contamination.classify([alias], TSMC, alias, lookup) == "read"


def test_reporting_the_real_name_is_a_phantom(lookup, rng):
    """The finding: a name the model was never shown came out of the weights."""
    alias = contamination.mint_alias(TSMC, lookup, rng)
    assert contamination.classify([TSMC], TSMC, alias, lookup) == "phantom"
    assert contamination.classify(["Taiwan Semiconductor"], TSMC, alias, lookup) == "phantom"


def test_a_phantom_is_caught_through_resolution_not_just_spelling(lookup, rng):
    """"TSMC" and the full name are the same identity and both count."""
    alias = contamination.mint_alias(TSMC, lookup, rng)
    assert contamination.classify(["TSMC"], TSMC, alias, lookup) == "phantom"


def test_reporting_neither_is_silence(lookup, rng):
    alias = contamination.mint_alias(TSMC, lookup, rng)
    assert contamination.classify(["Intel Corporation"], TSMC, alias, lookup) == "silent"
    assert contamination.classify([], TSMC, alias, lookup) == "silent"


def test_reading_wins_over_a_phantom_when_both_appear(lookup, rng):
    """A model that names the alias did read; a stray real name is secondary."""
    alias = contamination.mint_alias(TSMC, lookup, rng)
    assert contamination.classify([alias, TSMC], TSMC, alias, lookup) == "read"
