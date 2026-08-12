from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class OptimizerConfig:
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.1
    solver: str | None = "CLARABEL"


def optimize_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    betas: pd.Series,
    prev_weights: Optional[pd.Series],
    constraints,
    config: OptimizerConfig,
) -> pd.Series:
    mu = mu.dropna()

    if isinstance(mu.index, pd.MultiIndex):
        mu = mu.copy()
        mu.index = mu.index.get_level_values(-1)
        mu = mu[~mu.index.duplicated(keep="first")]

    tickers = mu.index.tolist()
    if tickers and isinstance(tickers[0], tuple):
        tickers = [t[-1] if isinstance(t, tuple) else t for t in tickers]
        mu.index = tickers

    if not tickers:
        raise ValueError("Optimizer received empty signal.")

    cov = cov.loc[tickers, tickers]
    betas = betas.reindex(tickers).fillna(0.0)

    n = len(tickers)
    cov = cov.fillna(0.0).values
    cov = cov + np.eye(n) * 1e-6

    w = cp.Variable(n)
    t = cp.Variable(n, nonneg=True)

    # Cross-sectionally standardize the signal. Raw model output is a forecast
    # of 21-day returns (order 1e-3); the quadratic risk term is order 1e-2,
    # so on the raw scale the penalty dominates and the optimizer returns a
    # zero portfolio regardless of how good the ranking is. Only the
    # cross-sectional ORDERING carries information, so normalizing to unit
    # dispersion is the scale-free way to make the two terms comparable.
    mu_std = float(mu.values.std())
    signal = (mu.values - mu.values.mean()) / mu_std if mu_std > 1e-12 else mu.values

    obj = signal @ w - config.risk_aversion * cp.quad_form(w, cov)
    if prev_weights is not None and len(prev_weights) == n:
        obj -= config.turnover_penalty * cp.norm1(w - prev_weights.values)

    L = constraints.gross_leverage
    cons = [
        cp.sum(w) == 0.0,
        t >= w,
        t >= -w,
        # NOTE: a minimum-gross constraint (sum(t) >= k) is vacuous here and
        # was silently producing zero-weight portfolios. `t` is only bounded
        # BELOW by |w| and carries no cost in the objective, so the solver
        # satisfies any such constraint by inflating t while driving w to
        # zero. Minimum gross is non-convex; the leverage target is enforced
        # by rescaling after the solve instead.
        cp.sum(t) <= L,
        t <= constraints.max_weight,
        cp.abs(betas.values @ w) <= constraints.beta_tolerance,
    ]

    problem = cp.Problem(cp.Maximize(obj), cons)
    solver = config.solver or "CLARABEL"
    problem.solve(solver=solver, verbose=False)

    if w.value is None:
        raise RuntimeError(f"Portfolio optimizer failed (status={problem.status}, solver={solver}).")

    weights = pd.Series(w.value, index=tickers)

    # Scale up to the leverage target. The optimizer respects a gross CAP but
    # nothing forces it to use the full budget, so an unscaled solution can sit
    # far below target and quietly under-deploy the strategy.
    gross = float(weights.abs().sum())
    if gross < 1e-8:
        raise RuntimeError(
            f"Optimizer returned a degenerate zero portfolio (gross={gross:.2e}). "
            "Check signal scaling and risk_aversion."
        )
    weights *= L / gross

    # Rescaling can push a name past the per-position cap; clip and renormalize
    # so both the box and the leverage target hold.
    cap = constraints.max_weight
    if weights.abs().max() > cap + 1e-9:
        weights = weights.clip(-cap, cap)
        gross = float(weights.abs().sum())
        if gross > 1e-8:
            weights *= min(L / gross, 1.0)

    return weights
