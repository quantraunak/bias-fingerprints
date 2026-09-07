"""Guards on the edge panel and the link-lagged signal.

These two modules produce the result the study reports, and until now neither
had a test. The invariants below are the ones whose violation would not show up
as an error: an edge visible a day before it was filed, or a counterparty return
drawn from inside the prediction window, both yield a signal that looks
perfectly well behaved and is worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.graph import build, signal


@pytest.fixture
def calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", "2021-06-30")


@pytest.fixture
def claims() -> pd.DataFrame:
    """Two issuers, three claims, filed on different dates."""
    return pd.DataFrame([
        {"source_ticker": "AAA", "counterparty_ticker": "BBB", "relation": "customer",
         "filed": pd.Timestamp("2020-02-14"), "revenue_pct": 20.0, "confidence": "high"},
        {"source_ticker": "AAA", "counterparty_ticker": "CCC", "relation": "supplier",
         "filed": pd.Timestamp("2020-08-14"), "revenue_pct": None, "confidence": "high"},
        {"source_ticker": "DDD", "counterparty_ticker": "BBB", "relation": "partner",
         "filed": pd.Timestamp("2020-02-14"), "revenue_pct": None, "confidence": "high"},
    ])


def test_edges_are_not_visible_before_they_were_filed(claims):
    """The whole point-in-time claim rests on this one comparison."""
    edges = build.edges(claims)
    for row in edges.itertuples():
        assert row.valid_from == row.filed if hasattr(row, "filed") else True
    assert build.active(edges, pd.Timestamp("2020-02-13")).empty
    assert len(build.active(edges, pd.Timestamp("2020-02-14"))) == 2
    # The August claim is invisible in July and visible in September.
    assert "CCC" not in set(build.active(edges, pd.Timestamp("2020-07-01")).target)
    assert "CCC" in set(build.active(edges, pd.Timestamp("2020-09-01")).target)


def test_a_later_filing_supersedes_the_earlier_one(claims):
    """Relationships lapse by not being restated, which is how filings work.

    An issuer's next 10-K bounds every edge the previous one disclosed, so a
    counterparty that stops being named stops being an edge without the filing
    ever saying it ended.
    """
    edges = build.edges(claims)
    aaa_bbb = edges[(edges.source == "AAA") & (edges.target == "BBB")].iloc[0]
    assert aaa_bbb.valid_to == pd.Timestamp("2020-08-14")


def test_final_filing_expires_rather_than_living_forever(claims):
    edges = build.edges(claims)
    ddd = edges[edges.source == "DDD"].iloc[0]
    assert ddd.valid_to == pd.Timestamp("2020-02-14") + pd.Timedelta(days=build.MAX_AGE_DAYS)


def test_self_loops_are_dropped():
    """A self-loop feeds a firm's own lagged return back as its predictor."""
    frame = pd.DataFrame([{
        "source_ticker": "AAA", "counterparty_ticker": "AAA", "relation": "customer",
        "filed": pd.Timestamp("2020-02-14"), "revenue_pct": None, "confidence": "high"}])
    assert build.edges(frame).empty


def test_competitor_links_are_not_edges():
    """A competitor's good month carries no signed prediction for the filer."""
    frame = pd.DataFrame([{
        "source_ticker": "AAA", "counterparty_ticker": "BBB", "relation": "competitor",
        "filed": pd.Timestamp("2020-02-14"), "revenue_pct": None, "confidence": "high"}])
    assert build.edges(frame).empty


def test_signal_never_uses_a_return_from_after_the_rebalance(claims, calendar):
    """The second of the two look-ahead routes into this signal.

    A counterparty return drawn from inside the prediction window would make the
    signal predict a return it has already seen. Changing the future must leave
    the signal untouched.
    """
    edges = build.edges(claims)
    as_of = pd.Timestamp("2020-06-30")
    dates = pd.DatetimeIndex([as_of])

    returns = pd.DataFrame(0.001, index=calendar, columns=["AAA", "BBB", "CCC", "DDD"])
    baseline = signal.link_signal(edges, returns, dates)

    poisoned = returns.copy()
    poisoned.loc[poisoned.index > as_of] = 5.0
    after = signal.link_signal(edges, poisoned, dates)

    pd.testing.assert_series_equal(baseline, after)
    assert not baseline.empty, "fixture produced no signal; the test would be vacuous"


def test_signal_is_the_weighted_mean_of_counterparty_returns(calendar):
    """A hand-checkable value, so a silent change in the aggregation is caught."""
    edges = build.edges(pd.DataFrame([
        {"source_ticker": "AAA", "counterparty_ticker": "BBB", "relation": "customer",
         "filed": pd.Timestamp("2020-01-02"), "revenue_pct": None, "confidence": "high"},
        {"source_ticker": "AAA", "counterparty_ticker": "CCC", "relation": "customer",
         "filed": pd.Timestamp("2020-01-02"), "revenue_pct": None, "confidence": "high"},
    ]))
    as_of = pd.Timestamp("2020-06-30")
    returns = pd.DataFrame(0.0, index=calendar, columns=["AAA", "BBB", "CCC"])
    window = calendar[(calendar <= as_of)][-21:]
    returns.loc[window, "BBB"] = 0.01
    returns.loc[window, "CCC"] = -0.01

    got = signal.link_signal(edges, returns, pd.DatetimeIndex([as_of])).loc[(as_of, "AAA")]
    expected = ((1.01 ** 21 - 1) + (0.99 ** 21 - 1)) / 2
    assert got == pytest.approx(expected, rel=1e-9)


def test_placebo_preserves_degree_and_drops_self_loops(claims):
    """If the shuffle changed the degree distribution it would not be a null."""
    edges = build.edges(claims)
    shuffled = signal.placebo(edges, seed=0)
    assert (shuffled.source == shuffled.target).sum() == 0
    assert sorted(shuffled.source) == sorted(edges.source)


def test_second_order_excludes_self_and_first_order_neighbours():
    """A name reachable at one hop must not also be counted at two."""
    first = {"AAA": [("BBB", 1.0)], "BBB": [("CCC", 1.0), ("AAA", 1.0)]}
    second = build.second_order(first)
    assert dict(second["AAA"]) == {"CCC": 1.0}
