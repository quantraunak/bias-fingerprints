"""Guards on counterparty resolution.

Each test corresponds to a resolution defect found in corpus output. They matter
more than they look: an unresolved counterparty costs one edge, but a wrongly
resolved one puts an unrelated return series on a real relationship, and the
resulting signal is still smooth and plausible enough that no downstream
diagnostic will announce it.

The reference registry is a fixture rather than a live download, so the suite is
deterministic and runs offline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.graph import resolve


@pytest.fixture
def reference() -> pd.DataFrame:
    """A registry shaped like EDGAR's, carrying each case under test."""
    rows = [
        ("TSM", "TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD"),
        ("DE", "DEERE & CO"),
        ("STM", "STMicroelectronics N.V."),
        ("BA", "BOEING CO"),
        ("AMAT", "APPLIED MATERIALS INC /DE"),
        ("LRCX", "LAM RESEARCH CORP"),
        ("CHCO", "City Holding Co"),
        ("INTC", "INTEL CORP"),
        # Filler registrants so the generic tokens under test carry the document
        # frequency they carry in the real registry -- "city" appears in six
        # registered names there, "semiconductor" in twelve, "manufacturing" in
        # five. The rule is a statement about the registry's token distribution,
        # so a fixture that does not reproduce that distribution cannot test it.
        ("CTY1", "CITY OFFICE REIT INC"),
        ("CTY2", "CITY DEVELOPMENTS LTD"),
        ("CTY3", "MIDWEST CITY BANCORP"),
        ("CTY4", "CITY BANK TEXAS"),
        ("SEM1", "AMKOR SEMICONDUCTOR INC"),
        ("SEM2", "TOWER SEMICONDUCTOR LTD"),
        ("SEM3", "ALPHA SEMICONDUCTOR CORP"),
        ("SEM4", "MAGNACHIP SEMICONDUCTOR CORP"),
        ("MFG1", "GLOBAL MANUFACTURING CORP"),
        ("MFG2", "SANMINA MANUFACTURING INC"),
        ("MFG3", "KIMBALL MANUFACTURING CO"),
    ]
    frame = pd.DataFrame(rows, columns=["ticker", "title"])
    frame["cik"] = range(1, len(frame) + 1)
    frame["normalized"] = frame["title"].map(resolve.normalize)
    frame["tokens"] = frame["normalized"].map(lambda s: frozenset(s.split()))
    return frame


@pytest.fixture
def lookup(reference):
    return resolve.build_lookup(reference)


def test_dotted_acronym_suffix_is_stripped():
    """`N.V.` shattered into the tokens "n" and "v", which survived the suffix
    filter, so STMicroelectronics never matched a filing naming it."""
    assert resolve.normalize("STMicroelectronics N.V.") == "stmicroelectronics"
    assert resolve.normalize("Fiat S.p.A.") == "fiat"


def test_parenthetical_short_form_is_stripped():
    """A filing introduces a short form on first mention and drops it after.
    Keeping it split one company into two nodes, and scored a correct extraction
    as both a false positive and a false negative at once."""
    assert resolve.normalize("European Aeronautic Defence and Space Company (EADS)") == \
           resolve.normalize("European Aeronautic Defence and Space Company")
    assert resolve.normalize('Tianjin Haiguang Advanced Technology Investment Co., Ltd. ("THATIC")') == \
           resolve.normalize("Tianjin Haiguang Advanced Technology Investment Co., Ltd.")


def test_state_of_incorporation_marker_is_stripped():
    assert resolve.normalize("NORTHROP GRUMMAN CORP /DE/") == "northrop grumman"


def test_generic_shared_tokens_do_not_resolve(lookup):
    """SMIC resolved to TSM: both reduce to {semiconductor, manufacturing} plus
    one distinguishing token, and a subset rule that only counted tokens put a
    Taiwanese foundry's returns on a Chinese foundry's edge."""
    ticker, tier = resolve.resolve_one("Semiconductor Manufacturing International Corporation", lookup)
    assert ticker is None, f"resolved to {ticker} via {tier}"


def test_generic_single_token_reference_does_not_resolve(lookup):
    """"City Holding Co" reduces to the single generic token "city", so every
    two-word phrase starting with City matched it -- the corpus produced
    "City Council" -> CHCO."""
    assert resolve.resolve_one("City Council", lookup)[0] is None


def test_distinctive_single_token_reference_does_resolve(lookup):
    """The fix for City Council must not cost Deere. "Deere & Co" also reduces
    to one token, but that token names exactly one registrant, which is the
    difference a token count cannot see."""
    assert resolve.resolve_one("John Deere", lookup) == ("DE", "subset")


def test_exact_names_resolve(lookup):
    for name, expected in [
        ("Applied Materials", "AMAT"),
        ("Lam Research Corporation", "LRCX"),
        ("Boeing Company", "BA"),
        ("STMicroelectronics, Inc.", "STM"),
    ]:
        assert resolve.resolve_one(name, lookup)[0] == expected, name


def test_aliases_win_over_matching(lookup):
    """Names whose filing form never matches their registered form, and foreign
    counterparties with no US line, are decided by hand and must stay decided."""
    assert resolve.resolve_one("TSMC", lookup) == ("TSM", "alias")
    assert resolve.resolve_one("Samsung Electronics", lookup) == (None, "alias")


def test_ambiguous_single_tokens_are_refused(lookup):
    """A single common word is not an identification."""
    assert resolve.resolve_one("Delta", lookup)[0] is None
    assert resolve.resolve_one("Apple", lookup)[0] is None


def test_government_bodies_do_not_resolve(lookup):
    """The extractor emits regulators and agencies; none is a tradable node, and
    resolution is the stage that has to refuse them."""
    for name in ["U.S. Army", "Federal Reserve Board", "U.S. Government"]:
        assert resolve.resolve_one(name, lookup)[0] is None, name


def test_anonymous_references_are_structural_not_prefixed():
    """A prefix rule cannot separate "one customer" from "Single Touch Systems".

    The first version rejected any name starting with a determiner or ranking
    adjective, which also rejected real registrants beginning with "single",
    "primary" or "top". What actually distinguishes the two is that every token
    of an anonymous reference is a quantifier, a ranking word or a role noun.
    """
    from src.graph import extract

    for anonymous in ["one customer", "two customers", "top three customers",
                      "a major distributor", "our largest customer", "Customer A",
                      "certain suppliers", "third parties"]:
        assert extract.is_anonymous(anonymous), anonymous

    for real in ["Intel Corporation", "Boeing", "Single Touch Systems",
                 "Primary Health Properties", "Advanced Semiconductor Engineering, Inc.",
                 "Hon Hai Precision Industry Co., Ltd."]:
        assert not extract.is_anonymous(real), real
