"""End-to-end backtest smoke test on synthetic panel."""

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest


def _synthetic_panel(n_tickers: int = 30, n_days: int = 800) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(7)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)] + ["SPY"]
    rets = rng.normal(0.0003, 0.015, size=(n_days, len(tickers)))
    rets[:, -1] = rng.normal(0.0004, 0.01, size=n_days)
    prices = 100 * (1 + pd.DataFrame(rets, index=idx, columns=tickers)).cumprod()
    volumes = pd.DataFrame(rng.randint(1_000_000, 5_000_000, size=prices.shape), index=idx, columns=tickers)
    return prices, volumes


def test_backtest_produces_oos_returns():
    prices, volumes = _synthetic_panel()
    cfg = BacktestConfig(
        start="2016-01-01",
        end=str(prices.index[-1].date()),
        oos_start="2018-01-01",
        rebalance_freq="ME",
        train_lookback_months=24,
        horizon_days=21,
        long_short_quantile=0.15,
        beta_window=126,
        cov_window=126,
        min_train_samples=500,
        risk_aversion=5.0,
        turnover_penalty=0.1,
        commission_bps=1.0,
        slippage_bps=5.0,
        max_weight=0.1,
        gross_leverage=1.5,
        beta_tolerance=0.05,
        solver="CLARABEL",
        normalize="zscore",
        winsorize_limits=(0.05, 0.95),
        sector_neutral=False,
        benchmark="SPY",
        model_config={"type": "ridge", "min_train_samples": 500},
        vol_target=None,
    )

    out = run_backtest(prices, volumes, cfg)
    rets = out["daily_returns"]
    assert len(rets) > 100
    assert rets.index.min() >= pd.Timestamp("2018-01-01")
    assert abs(out["holdings"].sum(axis=1).mean()) < 0.05

    betas = [e["portfolio_beta"] for e in out["rebalance_costs"]]
    assert all(abs(b) <= cfg.beta_tolerance + 1e-3 for b in betas)
