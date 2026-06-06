from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    total = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return total ** (1 / years) - 1 if years > 0 else 0.0


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    return (returns.mean() / downside.std()) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return dd.min()


def calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    dd = max_drawdown(returns)
    if dd == 0:
        return 0.0
    return cagr(returns, periods_per_year) / abs(dd)


def tail_ratio(returns: pd.Series) -> float:
    """Right tail (95th pctile) / left tail (abs 5th pctile)."""
    if returns.empty:
        return 0.0
    right = np.percentile(returns, 95)
    left = abs(np.percentile(returns, 5))
    if left == 0:
        return 0.0
    return float(right / left)


def hit_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return (returns > 0).mean()


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    aligned = pd.concat([returns, market_returns], axis=1, join="inner").dropna()
    if aligned.empty or aligned.iloc[:, 1].var() == 0:
        return 0.0
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    periods_per_year: int = 252,
    seed: int = 7,
) -> tuple[float, float]:
    """Block-bootstrap confidence interval for the Sharpe ratio.

    Uses block resampling (block size ~ sqrt(n)) to preserve
    autocorrelation structure in returns.
    """
    rng = np.random.RandomState(seed)
    n = len(returns)
    if n < 20:
        return (0.0, 0.0)

    vals = returns.values
    block_size = max(5, int(np.sqrt(n)))
    sharpes = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        blocks = [vals[s : s + block_size] for s in starts]
        sample = np.concatenate(blocks)[:n]
        std = sample.std()
        sharpes[i] = (sample.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    alpha = (1 - ci) / 2
    return (float(np.percentile(sharpes, 100 * alpha)),
            float(np.percentile(sharpes, 100 * (1 - alpha))))


def cost_stress_sharpe(
    gross_returns: pd.Series,
    rebalance_costs: list[dict],
    commission_bps: float,
    slippage_scenarios_bps: list[float],
) -> dict[str, float]:
    """Recompute Sharpe under alternative slippage assumptions (same turnover)."""
    out = {}
    if gross_returns.empty or not rebalance_costs:
        return out

    for slip_bps in slippage_scenarios_bps:
        net = gross_returns.copy()
        for event in rebalance_costs:
            dt = pd.Timestamp(event["date"])
            cost = event["turnover"] * (commission_bps + slip_bps) / 10_000.0
            idx = net.index[net.index >= dt]
            if not idx.empty:
                net.loc[idx[0]] -= cost
        out[f"sharpe_slippage_{int(slip_bps)}bps"] = round(sharpe(net), 4)
    return out


def alpha_vs_market(returns: pd.Series, market_returns: pd.Series) -> dict[str, float]:
    aligned = pd.concat([returns, market_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {"alpha_ann": 0.0, "beta": 0.0, "r_squared": 0.0}
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    x_ = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(x_, y, rcond=None)
    alpha_daily, beta_hat = coef[0], coef[1]
    fitted = x_ @ coef
    ss_res = ((y - fitted) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "alpha_ann": float(alpha_daily * 252),
        "beta": float(beta_hat),
        "r_squared": float(r2),
    }


def summarize(
    returns: pd.Series,
    turnovers: list[float],
    holdings: pd.DataFrame,
    market_returns: pd.Series | None = None,
    gross_returns: pd.Series | None = None,
    rebalance_costs: list[dict] | None = None,
    commission_bps: float = 1.0,
    base_slippage_bps: float = 5.0,
    cost_stress_bps: list[float] | None = None,
) -> dict:
    sr_lo, sr_hi = bootstrap_sharpe_ci(returns)

    result = {
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "sharpe_ci_95": [round(sr_lo, 4), round(sr_hi, 4)],
        "sortino": sortino(returns),
        "calmar": calmar(returns),
        "max_drawdown": max_drawdown(returns),
        "annualized_volatility": annualized_volatility(returns),
        "tail_ratio": tail_ratio(returns),
        "hit_rate": hit_rate(returns),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "avg_gross_leverage": float(holdings.abs().sum(axis=1).mean()) if not holdings.empty else 0.0,
        "avg_net_exposure": float(holdings.sum(axis=1).mean()) if not holdings.empty else 0.0,
        "beta_vs_spy": beta(returns, market_returns) if market_returns is not None else None,
        "n_days": len(returns),
    }

    if market_returns is not None:
        result["market_regression"] = alpha_vs_market(returns, market_returns)

    if gross_returns is not None and rebalance_costs and cost_stress_bps:
        result["cost_stress"] = cost_stress_sharpe(
            gross_returns, rebalance_costs, commission_bps, cost_stress_bps
        )

    return result
