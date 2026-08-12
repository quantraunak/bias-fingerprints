"""Goal and wealth projection under uncertainty.

Two deliberate departures from standard financial-planning software:

1. Returns are drawn by STATIONARY BLOCK BOOTSTRAP from a fat-tailed,
   regime-switching generator rather than i.i.d. normals. Sequence-of-returns
   risk is the dominant driver of ruin for a spending household, and i.i.d.
   normal draws systematically understate it.

2. Every path is run AFTER TAX and AFTER FEES, with spending indexed to a
   stochastic inflation path. Pre-tax projections flatter the plan by roughly
   100-200bps a year, which compounds into a very different answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..types import Goal, Household


@dataclass
class ProjectionConfig:
    n_paths: int = 10_000
    years: int = 30
    block_size: int = 3  # years per bootstrap block; preserves autocorrelation
    seed: int = 42
    inflation_mean: float = 0.024
    inflation_vol: float = 0.013
    # Regime parameters: a calm regime and a crisis regime with persistence.
    crisis_prob: float = 0.14  # unconditional share of years in crisis
    crisis_persistence: float = 0.55  # P(crisis | crisis last year)
    crisis_return_shift: float = -0.22  # additive return shock in crisis
    crisis_vol_multiple: float = 1.9
    fee_drag: float = 0.0035  # all-in advisory + fund + trading costs
    tax_drag: float = 0.0  # supplied by the caller from the CMA


@dataclass
class ProjectionResult:
    terminal_wealth: np.ndarray  # real (today's) dollars
    paths: np.ndarray  # (n_paths, years+1) real dollars
    success_rate: float
    median_terminal: float
    p10_terminal: float
    p90_terminal: float
    max_drawdown_p95: float
    prob_double_real: float
    prob_ruin: float
    safe_spending_rate: float
    years_to_double_median: float | None
    goal_funding: pd.DataFrame

    def summary(self) -> dict:
        return {
            "success_rate": round(self.success_rate, 4),
            "prob_ruin": round(self.prob_ruin, 4),
            "median_terminal_real": round(self.median_terminal, 0),
            "p10_terminal_real": round(self.p10_terminal, 0),
            "p90_terminal_real": round(self.p90_terminal, 0),
            "prob_double_real": round(self.prob_double_real, 4),
            "years_to_double_median": self.years_to_double_median,
            "worst_drawdown_p95": round(self.max_drawdown_p95, 4),
            "safe_spending_rate": round(self.safe_spending_rate, 4),
            "goal_funding": self.goal_funding.to_dict(orient="records"),
        }


def _simulate_returns(
    expected_return: float,
    volatility: float,
    cfg: ProjectionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Regime-switching annual returns, then block-bootstrapped.

    Returns array of shape (n_paths, years).
    """
    n, T = cfg.n_paths, cfg.years

    # Markov chain over {calm, crisis}.
    p_cc = cfg.crisis_persistence
    # Solve for P(crisis|calm) that gives the target unconditional probability.
    p_nc = cfg.crisis_prob * (1 - p_cc) / max(1 - cfg.crisis_prob, 1e-9)
    p_nc = float(np.clip(p_nc, 0.0, 1.0))

    state = rng.random(n) < cfg.crisis_prob
    out = np.zeros((n, T))

    # Calm-regime parameters are backed out so the BLEND matches the CMA.
    p = cfg.crisis_prob
    calm_ret = (expected_return - p * (expected_return + cfg.crisis_return_shift)) / max(1 - p, 1e-9)

    # Total variance of a mixture is within-regime variance PLUS the variance
    # of the regime means. Omitting that second term is what makes most
    # regime-switching planners quietly run ~25% hotter than their stated
    # volatility, which in turn understates ruin probability.
    m = cfg.crisis_vol_multiple
    between_var = p * (1 - p) * cfg.crisis_return_shift**2
    within_target = volatility**2 - between_var
    if within_target <= 0:
        # Mean shift alone already exceeds the target; shrink it to fit.
        calm_vol = volatility * 0.25
    else:
        calm_vol = np.sqrt(within_target / ((1 - p) + p * m**2))

    for t in range(T):
        mu = np.where(state, expected_return + cfg.crisis_return_shift, calm_ret)
        sd = np.where(state, calm_vol * cfg.crisis_vol_multiple, calm_vol)
        # Student-t innovations (df=5) for fat tails, rescaled to unit variance.
        df = 5
        z = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2))
        out[:, t] = mu + sd * z
        switch = rng.random(n)
        state = np.where(state, switch < p_cc, switch < p_nc)

    # Stationary block bootstrap over the time axis to preserve multi-year runs.
    if cfg.block_size > 1:
        idx = np.zeros((n, T), dtype=int)
        for start in range(0, T, cfg.block_size):
            length = min(cfg.block_size, T - start)
            origin = rng.integers(0, T - length + 1, size=n)
            for k in range(length):
                idx[:, start + k] = origin + k
        out = np.take_along_axis(out, idx, axis=1)

    return out


def project(
    household: Household,
    expected_return: float,
    volatility: float,
    cfg: ProjectionConfig | None = None,
) -> ProjectionResult:
    """Run the household forward under the policy portfolio."""
    cfg = cfg or ProjectionConfig(years=household.horizon_years)
    rng = np.random.default_rng(cfg.seed)
    n, T = cfg.n_paths, cfg.years

    net_return = expected_return - cfg.fee_drag - cfg.tax_drag
    returns = _simulate_returns(net_return, volatility, cfg, rng)

    # Stochastic inflation, used to deflate to real dollars and index spending.
    infl = cfg.inflation_mean + cfg.inflation_vol * rng.standard_normal((n, T))
    cum_infl = np.cumprod(1 + infl, axis=1)

    W = np.zeros((n, T + 1))
    W[:, 0] = household.total_assets
    net_spend = max(household.annual_spending - household.annual_income, 0.0)

    # Goals become extra withdrawals in their year.
    goal_draw = np.zeros(T + 1)
    for g in household.goals:
        if 0 < g.year_offset <= T:
            goal_draw[g.year_offset] += g.amount

    for t in range(T):
        spend_nominal = net_spend * cum_infl[:, t]
        goal_nominal = goal_draw[t + 1] * cum_infl[:, t]
        start = W[:, t] - spend_nominal - goal_nominal
        start = np.maximum(start, 0.0)
        W[:, t + 1] = start * (1 + returns[:, t])
        W[:, t + 1] = np.maximum(W[:, t + 1], 0.0)

    real = W / np.concatenate([np.ones((n, 1)), cum_infl], axis=1)
    terminal = real[:, -1]

    # Drawdown on the real wealth path.
    peaks = np.maximum.accumulate(real, axis=1)
    dd = np.where(peaks > 0, real / np.maximum(peaks, 1e-9) - 1.0, 0.0)
    worst_dd = dd.min(axis=1)

    ruin = float((terminal <= 0).mean())
    success = float((terminal >= household.total_assets * 0.5).mean())
    doubled = float((terminal >= household.total_assets * 2).mean())

    # Median years until real wealth first doubles.
    target = household.total_assets * 2
    hit = np.argmax(real >= target, axis=1).astype(float)
    hit[real.max(axis=1) < target] = np.nan
    years_double = float(np.nanmedian(hit)) if np.any(~np.isnan(hit)) else None

    # Goal funding probabilities.
    rows = []
    for g in household.goals:
        if g.year_offset > T:
            continue
        need = g.amount * (cum_infl[:, min(g.year_offset, T) - 1] if g.inflation_linked else 1.0)
        available = W[:, g.year_offset]
        rows.append(
            {
                "goal": g.name,
                "priority": g.priority,
                "year": g.year_offset,
                "amount_today": g.amount,
                "prob_funded": round(float((available >= need).mean()), 4),
            }
        )
    goal_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["goal", "priority", "year", "amount_today", "prob_funded"]
    )

    return ProjectionResult(
        terminal_wealth=terminal,
        paths=real,
        success_rate=success,
        median_terminal=float(np.median(terminal)),
        p10_terminal=float(np.percentile(terminal, 10)),
        p90_terminal=float(np.percentile(terminal, 90)),
        max_drawdown_p95=float(np.percentile(worst_dd, 5)),
        prob_double_real=doubled,
        prob_ruin=ruin,
        safe_spending_rate=safe_spending_rate(household, expected_return, volatility, cfg),
        years_to_double_median=years_double,
        goal_funding=goal_df,
    )


def safe_spending_rate(
    household: Household,
    expected_return: float,
    volatility: float,
    cfg: ProjectionConfig | None = None,
    target_success: float = 0.90,
    preserve_real_capital: bool = True,
) -> float:
    """Largest constant real spending rate with `target_success` confidence.

    Bisection on the spending rate. For a multi-generational family office the
    constraint is usually capital preservation in real terms, not merely
    avoiding zero — hence `preserve_real_capital`.
    """
    cfg = cfg or ProjectionConfig(years=household.horizon_years, n_paths=3000)
    cfg = ProjectionConfig(**{**cfg.__dict__, "n_paths": min(cfg.n_paths, 3000)})
    rng = np.random.default_rng(cfg.seed + 1)
    n, T = cfg.n_paths, cfg.years

    returns = _simulate_returns(expected_return - cfg.fee_drag - cfg.tax_drag, volatility, cfg, rng)
    W0 = household.total_assets
    floor = W0 if preserve_real_capital else 0.0

    def success_at(rate: float) -> float:
        W = np.full(n, W0)
        for t in range(T):
            W = np.maximum(W - rate * W0, 0.0) * (1 + returns[:, t])
        return float((W >= floor).mean())

    lo, hi = 0.0, 0.12
    for _ in range(24):
        mid = (lo + hi) / 2
        if success_at(mid) >= target_success:
            lo = mid
        else:
            hi = mid
    return lo


def compare_policies(
    household: Household,
    policies: dict[str, tuple[float, float]],
    cfg: ProjectionConfig | None = None,
) -> pd.DataFrame:
    """Run several (expected_return, volatility) policies side by side."""
    rows = []
    for name, (ret, vol) in policies.items():
        r = project(household, ret, vol, cfg)
        rows.append(
            {
                "policy": name,
                "expected_return": round(ret, 4),
                "volatility": round(vol, 4),
                "median_terminal_real": round(r.median_terminal, 0),
                "p10_terminal_real": round(r.p10_terminal, 0),
                "prob_double_real": round(r.prob_double_real, 4),
                "prob_ruin": round(r.prob_ruin, 4),
                "worst_dd_p95": round(r.max_drawdown_p95, 4),
                "safe_spending_rate": round(r.safe_spending_rate, 4),
            }
        )
    return pd.DataFrame(rows)
