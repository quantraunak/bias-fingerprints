from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.features.factors import compute_factors
from src.features.preprocess import build_feature_matrix
from src.labels.forward_returns import compute_forward_returns
from src.models.model_factory import build_ranker
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.costs import turnover_cost
from src.portfolio.optimizer import OptimizerConfig, optimize_weights
from src.portfolio.risk_model import compute_effective_leverage, shrink_covariance


@dataclass
class BacktestConfig:
    start: str
    end: str
    oos_start: str | None
    rebalance_freq: str
    train_lookback_months: int
    horizon_days: int
    long_short_quantile: float
    beta_window: int
    cov_window: int
    min_train_samples: int
    risk_aversion: float
    turnover_penalty: float
    commission_bps: float
    slippage_bps: float
    max_weight: float
    gross_leverage: float
    beta_tolerance: float
    solver: str | None
    normalize: str
    winsorize_limits: tuple[float, float]
    sector_neutral: bool
    model_config: dict = field(default_factory=dict)
    cov_method: str = "ledoit_wolf"
    vol_target: float | None = None
    vol_lookback: int = 126
    vol_ewma_span: int | None = 126
    min_leverage: float = 0.4
    max_leverage: float = 2.0
    dd_threshold: float | None = 0.08
    dd_min_scale: float = 0.35


def _estimate_betas(returns: pd.DataFrame, market: pd.Series) -> pd.Series:
    aligned = returns.join(market.rename("market"), how="inner")
    if aligned.empty:
        return pd.Series(0.0, index=returns.columns)
    market_col = aligned["market"]
    var = market_col.var()
    if var == 0.0 or pd.isna(var):
        return pd.Series(0.0, index=returns.columns)
    betas = {}
    for col in returns.columns:
        series = aligned[col].dropna()
        common = series.index.intersection(market_col.dropna().index)
        if common.empty:
            betas[col] = 0.0
        else:
            betas[col] = series.loc[common].cov(market_col.loc[common]) / var
    return pd.Series(betas)


def run_backtest(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    config: BacktestConfig,
    sector_map: pd.Series | None = None,
) -> dict[str, Any]:
    prices = prices.loc[config.start : config.end].dropna(how="all", axis=1)
    volumes = volumes.loc[config.start : config.end].reindex(prices.index)

    returns = prices.pct_change(fill_method=None)
    oos_ts = pd.Timestamp(config.oos_start) if config.oos_start else None

    factors = compute_factors(prices, volumes)
    sectors = sector_map if config.sector_neutral else None
    features = build_feature_matrix(
        factors,
        winsorize_limits=config.winsorize_limits,
        normalize=config.normalize,
        sector_map=sectors,
    )

    fwd = compute_forward_returns(prices, config.horizon_days).stack(future_stack=True)
    fwd.index.names = ["date", "ticker"]

    rebal_dates = prices.resample(config.rebalance_freq).last().index
    rebal_dates = rebal_dates[(rebal_dates >= prices.index.min()) & (rebal_dates <= prices.index.max())]

    holdings: dict[pd.Timestamp, pd.Series] = {}
    gross_returns: list[pd.Series] = []
    net_returns: list[pd.Series] = []
    turnovers: list[float] = []
    rebalance_costs: list[dict] = []
    feature_importances: list[dict] = []
    leverage_history: list[tuple[pd.Timestamp, float]] = []

    prev_weights = pd.Series(dtype=float)
    cumulative_net = pd.Series(dtype=float)

    for i, dt in enumerate(rebal_dates[:-1]):
        next_dt = rebal_dates[i + 1]
        if oos_ts is not None and next_dt < oos_ts:
            continue

        train_start = dt - pd.DateOffset(months=config.train_lookback_months)
        train_mask = (features.index.get_level_values(0) >= train_start) & (
            features.index.get_level_values(0) < dt
        )
        test_mask = features.index.get_level_values(0) == dt

        X_train = features.loc[train_mask]
        y_train = fwd.reindex(X_train.index).dropna()
        ok = X_train.notna().all(axis=1)
        idx = y_train.index.intersection(X_train.index[ok])
        X_train, y_train = X_train.loc[idx], y_train.loc[idx]
        X_test = features.loc[test_mask]
        if X_test.empty:
            continue

        if len(y_train) < config.min_train_samples:
            pred = X_test.mean(axis=1)
        else:
            model = build_ranker(config.model_config)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fi = getattr(model, "feature_importance_", None)
            if fi is not None:
                feature_importances.append({"date": dt, **fi.to_dict()})

        pred = pred.dropna()
        if pred.empty:
            continue

        n = len(pred)
        q = max(1, int(n * config.long_short_quantile))
        selected = pd.concat([pred.nlargest(q), pred.nsmallest(q)])
        if isinstance(selected.index, pd.MultiIndex):
            selected = selected.copy()
            selected.index = selected.index.get_level_values("ticker")
        selected = selected[~selected.index.duplicated(keep="first")]
        sel_tickers = selected.index.tolist()

        ret_window = returns[sel_tickers].loc[:dt].iloc[-config.cov_window :]
        cov = shrink_covariance(ret_window) if config.cov_method == "ledoit_wolf" else ret_window.cov()

        market = returns["SPY"].loc[:dt].iloc[-config.beta_window :]
        betas = _estimate_betas(returns[sel_tickers].loc[:dt].iloc[-config.beta_window :], market)

        effective_leverage = compute_effective_leverage(
            cumulative_net,
            base_leverage=config.gross_leverage,
            target_vol=config.vol_target,
            vol_lookback=config.vol_lookback,
            min_leverage=config.min_leverage,
            max_leverage=config.max_leverage,
            ewma_span=config.vol_ewma_span,
            dd_threshold=config.dd_threshold,
            dd_min_scale=config.dd_min_scale,
        )
        leverage_history.append((dt, effective_leverage))

        w = optimize_weights(
            selected,
            cov,
            betas,
            prev_weights.reindex(sel_tickers).fillna(0.0),
            PortfolioConstraints(
                gross_leverage=effective_leverage,
                max_weight=config.max_weight,
                beta_tolerance=config.beta_tolerance,
            ),
            OptimizerConfig(
                risk_aversion=config.risk_aversion,
                turnover_penalty=config.turnover_penalty,
                solver=config.solver,
            ),
        )

        holdings[dt] = w
        prev_subset = prev_weights.reindex(sel_tickers).fillna(0.0)
        turnover = float((w - prev_subset).abs().sum())
        turnovers.append(turnover)
        cost = turnover_cost(prev_subset, w, config.commission_bps, config.slippage_bps)
        rebalance_costs.append({"date": dt, "turnover": turnover, "cost": cost})

        pnl_window = returns.loc[dt:next_dt].iloc[1:]
        if pnl_window.empty:
            continue

        gross = pnl_window[sel_tickers].mul(w, axis=1).sum(axis=1)
        net = gross.copy()
        net.iloc[0] -= cost
        gross_returns.append(gross)
        net_returns.append(net)
        cumulative_net = pd.concat(net_returns).sort_index() if net_returns else cumulative_net
        prev_weights = w

    gross_daily = pd.concat(gross_returns).sort_index() if gross_returns else pd.Series(dtype=float)
    net_daily = pd.concat(net_returns).sort_index() if net_returns else pd.Series(dtype=float)
    if not net_daily.empty:
        net_daily.index = pd.to_datetime(net_daily.index)
    if not gross_daily.empty:
        gross_daily.index = pd.to_datetime(gross_daily.index)

    return {
        "daily_returns": net_daily,
        "gross_returns": gross_daily,
        "holdings": pd.DataFrame(holdings).T.sort_index(),
        "turnovers": turnovers,
        "rebalance_costs": rebalance_costs,
        "feature_importances": pd.DataFrame(feature_importances) if feature_importances else pd.DataFrame(),
        "leverage_history": pd.DataFrame(leverage_history, columns=["date", "gross_leverage"]).set_index("date")
        if leverage_history
        else pd.DataFrame(),
    }
