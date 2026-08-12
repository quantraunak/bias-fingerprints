import pandas as pd
import pytest

from src.labels.forward_returns import compute_forward_returns, last_label_date


def test_forward_returns_alignment():
    prices = pd.DataFrame(
        {"AAA": [100, 110, 121]}, index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    )
    fwd = compute_forward_returns(prices, horizon_days=1)
    assert abs(fwd.loc["2020-01-01", "AAA"] - 0.10) < 1e-9
    assert abs(fwd.loc["2020-01-02", "AAA"] - 0.10) < 1e-9
    assert pd.isna(fwd.loc["2020-01-03", "AAA"])


def test_last_label_date_purges_peeking_labels():
    idx = pd.bdate_range("2020-01-01", periods=30)
    dt = idx[25]
    label_end = last_label_date(idx, dt, horizon_days=5)
    assert label_end == idx[20]
    assert label_end + pd.offsets.BDay(5) == dt

    assert last_label_date(idx, idx[3], horizon_days=5) is None
