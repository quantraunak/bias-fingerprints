"""Guards against the specific defects that produced the previous results.

Each test corresponds to a bug that shipped and looked plausible while wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval import metrics
from src.factors.transforms import lag_trading_days
from src.model import splits
from src.model.target import cross_sectional_rank, forward_returns
from src.model.walkforward import month_end_trading_days


@pytest.fixture
def calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", "2020-12-31")


def test_rebalance_dates_are_real_trading_days(calendar):
    """The original engine used calendar month-ends; 30% were not trading days,
    the cross-section came back empty, and the month's P&L was discarded."""
    frame = pd.DataFrame(1.0, index=calendar, columns=["A"])
    rebalance = month_end_trading_days(frame.index)

    assert len(rebalance) > 0
    assert rebalance.isin(frame.index).all(), "every rebalance date must exist in the price index"

    calendar_month_ends = frame.resample("ME").last().index
    missing = (~calendar_month_ends.isin(frame.index)).sum()
    assert missing > 0, "fixture should contain calendar month-ends that are not trading days"


def test_lag_is_in_trading_days_not_calendar_days(calendar):
    """252 calendar days is 173 trading days: the old momentum was 8 months, not 12."""
    frame = pd.DataFrame({"A": np.arange(len(calendar), dtype=float)}, index=calendar)
    lagged = lag_trading_days(frame, 252)
    assert lagged["A"].iloc[252] == frame["A"].iloc[0]
    assert lagged["A"].iloc[-1] == frame["A"].iloc[-253]


def test_forward_returns_do_not_leak_backwards(calendar):
    prices = pd.DataFrame({"A": np.linspace(100, 200, len(calendar))}, index=calendar)
    forward = forward_returns(prices, 21)
    # The last 21 rows cannot be known and must be NaN, not filled.
    assert forward["A"].tail(21).isna().all()
    expected = prices["A"].iloc[21] / prices["A"].iloc[0] - 1.0
    assert forward["A"].iloc[0] == pytest.approx(expected)


def test_training_window_is_purged_and_embargoed(calendar):
    predict_on = calendar[-1]
    window = splits.training_window(predict_on, calendar, horizon_days=21, embargo_days=21, train_years=5)
    gap = len(calendar[(calendar > window.train_end) & (calendar <= predict_on)])
    assert gap == 42, "training must stop horizon + embargo trading days before the prediction date"


def test_target_removes_the_market_component():
    """Ranking within a date is invariant to a common shift, which is the whole point:
    28% of the raw label's variance was the market move, which cannot be ranked on."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame(np.random.RandomState(0).randn(5, 20), index=dates)
    shocked = frame.add(pd.Series([0.05, -0.03, 0.10, 0.0, -0.08], index=dates), axis=0)
    pd.testing.assert_frame_equal(cross_sectional_rank(frame), cross_sectional_rank(shocked))


def test_uniqueness_weights_discount_overlapping_labels():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=200).repeat(3))
    weights = splits.uniqueness_weights(dates, horizon_days=21)
    assert len(weights) == len(dates)
    assert weights.mean() == pytest.approx(1.0)
    # In the steady state every label overlaps a full horizon of neighbours, so
    # weights are flat; only the first horizon has fewer overlaps and is upweighted.
    assert weights[0] > weights[len(weights) // 2]
    assert weights[len(weights) // 2] == pytest.approx(weights[-1])


def test_cagr_uses_elapsed_time_not_row_count():
    """The old metric annualised by 252/len(returns) on a series missing 29% of its
    days, which reported 5.9% where the elapsed-time growth rate was 4.1%."""
    full = pd.bdate_range("2016-01-01", "2020-12-31")
    returns = pd.Series(0.0004, index=full)
    complete = metrics.summarize(returns)

    holed = returns[::3]  # same span, a third of the observations
    sparse = metrics.summarize(holed)

    assert complete["coverage"] == pytest.approx(1.0, abs=0.02)
    assert sparse["coverage"] < 0.4
    # Same elapsed window, so the growth rate must not be inflated by dropping days.
    assert sparse["cagr"] < complete["cagr"]
