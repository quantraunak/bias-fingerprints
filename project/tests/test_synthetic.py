"""Guards on constructed documents.

The whole method rests on one claim: the label is correct because we planted it.
Every defect here breaks that claim quietly. A document that contains fewer
items than its label says produces an artificially low recall; one whose length
varies with cardinality reintroduces exactly the confound the construction
exists to remove, and the resulting curve would be a length effect wearing a
cardinality label.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from src.graph import synthetic


@pytest.fixture
def claims() -> pd.DataFrame:
    rows = [
        ("Intel Corporation", "We purchase microprocessors from Intel Corporation under a long term supply agreement covering multiple product generations."),
        ("Samsung Electronics", "Samsung Electronics supplies memory components used across our consumer hardware product lines and has done so since 2011."),
        ("Boeing Company", "Sales to the Boeing Company accounted for a significant share of our aerospace segment revenue during the fiscal year."),
        ("Cisco Systems", "We provide contract manufacturing services to Cisco Systems for a range of networking equipment assembled in Asia."),
        # rejected: quote does not name the counterparty
        ("Ghost Industries", "This sentence describes a supplier relationship but never names the company involved in it anywhere at all."),
        # rejected: too short
        ("Tiny Corp", "Tiny Corp is a supplier."),
    ]
    return pd.DataFrame(rows, columns=["counterparty", "evidence"])


@pytest.fixture
def pool(claims):
    return synthetic.evidence_pool(claims)


@pytest.fixture
def filler():
    return [
        f"Our results of operations may fluctuate for reasons described in this section number {i} of the filing."
        for i in range(400)
    ]


@pytest.fixture
def rng():
    return random.Random(7)


# --------------------------------------------------------------------------
# the pools
# --------------------------------------------------------------------------

def test_a_quote_that_does_not_name_its_counterparty_is_rejected(pool):
    """It cannot serve as ground truth for a company it never mentions."""
    assert "Ghost Industries" not in set(pool.counterparty)


def test_short_quotes_are_rejected(pool):
    """Too short to carry relationship language the model could act on."""
    assert "Tiny Corp" not in set(pool.counterparty)


def test_usable_quotes_are_kept(pool):
    assert len(pool) == 4


def test_each_counterparty_appears_once(claims):
    """Repeated mentions would desynchronise planted count from distinct names."""
    duplicated = pd.concat([claims, claims])
    assert synthetic.evidence_pool(duplicated).counterparty.is_unique


def test_filler_is_split_into_sentences():
    text = ("This is a filler sentence of adequate length describing general business conditions. "
            "Here is a second filler sentence which is also of entirely adequate length for the pool. Short one.")
    out = synthetic.filler_pool([text])
    assert len(out) == 2
    assert all(synthetic.MIN_FILLER <= len(s) <= synthetic.MAX_FILLER for s in out)


# --------------------------------------------------------------------------
# construction: the label must be exactly what is in the document
# --------------------------------------------------------------------------

def test_every_planted_item_appears_in_the_document(pool, filler, rng):
    """If an item is labelled but absent, recall is understated by construction."""
    document, labels = synthetic.build(3, pool, filler, rng)
    assert len(labels) == 3
    for name in labels:
        quote = pool.loc[pool.counterparty == name, "evidence"].iloc[0]
        assert quote in document


def test_the_label_has_no_duplicates(pool, filler, rng):
    _, labels = synthetic.build(4, pool, filler, rng)
    assert len(set(labels)) == len(labels)


def test_length_is_held_constant_across_cardinalities(pool, filler, rng):
    """The confound the construction exists to remove.

    If a 4-item document is systematically longer than a 1-item document, a
    measured cardinality effect could be a length effect instead, and the whole
    experiment would be uninterpretable.
    """
    lengths = [len(synthetic.build(n, pool, filler, rng, target_chars=4000)[0])
               for n in (1, 2, 3, 4)]
    assert max(lengths) - min(lengths) < 0.15 * min(lengths)


def test_asking_for_more_items_than_the_pool_holds_raises(pool, filler, rng):
    """Silently returning fewer would mislabel the cardinality."""
    with pytest.raises(ValueError):
        synthetic.build(99, pool, filler, rng)


def test_construction_is_reproducible_from_a_seed(pool, filler):
    first = synthetic.build(3, pool, filler, random.Random(11))
    second = synthetic.build(3, pool, filler, random.Random(11))
    assert first == second


def test_zero_items_is_a_valid_document(pool, filler, rng):
    """The control condition: filler only, so precision stays measurable."""
    document, labels = synthetic.build(0, pool, filler, rng)
    assert labels == [] and len(document) > 1000


# --------------------------------------------------------------------------
# position: kept separable from cardinality
# --------------------------------------------------------------------------

@pytest.mark.parametrize("where", ["front", "back"])
def test_clustered_placement_puts_items_in_the_named_third(pool, filler, rng, where):
    """Position is the axis the lost-in-the-middle literature varies.

    Keeping it a parameter is what lets a cardinality effect be distinguished
    from a position effect rather than confounded with one.
    """
    document, labels = synthetic.build(2, pool, filler, rng, spread=where)
    positions = []
    for name in labels:
        quote = pool.loc[pool.counterparty == name, "evidence"].iloc[0]
        positions.append(document.index(quote) / len(document))
    assert all(p < 0.5 for p in positions) if where == "front" else all(p > 0.4 for p in positions)


def test_an_unknown_spread_is_rejected(pool, filler, rng):
    with pytest.raises(ValueError):
        synthetic.build(2, pool, filler, rng, spread="sideways")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _resolve_one(name, lookup):
    return lookup.get(synthetic._normalise(name)), "exact"


LOOKUP = {"intel": "INTC", "intel corporation": "INTC", "samsung electronics": "SSNLF"}


def test_recovery_counts_matching_counterparties():
    got = synthetic.recovered(["Intel Corporation", "Samsung Electronics"],
                              ["Intel Corporation", "Samsung Electronics"], LOOKUP, _resolve_one)
    assert got == 2


def test_recovery_matches_on_resolved_ticker_not_spelling():
    """"Intel" and "Intel Corporation" are one company, and one recovery."""
    assert synthetic.recovered(["Intel"], ["Intel Corporation"], LOOKUP, _resolve_one) == 1


def test_extra_names_do_not_inflate_recovery(pool):
    assert synthetic.recovered(["Intel Corporation", "Unrelated Co"],
                               ["Intel Corporation"], LOOKUP, _resolve_one) == 1


def test_nothing_extracted_is_zero_recovery():
    assert synthetic.recovered([], ["Intel Corporation"], LOOKUP, _resolve_one) == 0
