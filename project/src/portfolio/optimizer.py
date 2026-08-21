"""Mean-variance optimisation under dollar-, beta- and box-constraints.

Carried over from the previous engine, which had already been debugged; the
three findings below cost real time to locate and are kept as comments because
each one produced plausible-looking output while being wrong.

1. A minimum-gross constraint written as `t >= |w|, sum(t) >= k` is vacuous.
   `t` is only bounded below and carries no cost, so the solver satisfies it by
   inflating `t` while driving `w` to zero. Reported leverage read 2.0 against
   weights of order 1e-11. Minimum gross is non-convex and cannot be expressed
   this way.
2. Raw model output forecasts 21-day returns, order 1e-3, against a quadratic
   risk term of order 1e-2. On the raw scale the penalty dominates and the zero
   portfolio is optimal no matter how good the ranking is. Only the
   cross-sectional ordering carries information, so the signal is standardised
   to unit dispersion before it enters the objective.
3. A per-name cap below equal weight makes the box infeasible at every
   rebalance. The cap has to be checked against the number of names actually
   held.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Constraints:
    gross_leverage: float
    max_weight: float
    beta_tolerance: float
    vol_target: float  # annualised portfolio volatility budget


@dataclass(frozen=True)
class Objective:
    turnover_penalty: float
    solver: str = "CLARABEL"
    information_coefficient: float = 0.02
    horizon_days: int = 21


def expected_returns(
    scores: pd.Series, covariance: pd.DataFrame, ic: float, horizon_days: int = 21
) -> np.ndarray:
    """Grinold's forecasting rule, in annual units: alpha = IC * volatility * score.

    A model score is a ranking, not a return, and the two cannot be traded off
    against a covariance matrix until they share units. Grinold and Kahn's
    identity supplies the conversion: under a signal with an information
    coefficient of 0.012, a one-sigma score on a name whose 21-day volatility is
    10% forecasts 0.012 * 0.10 = 12bp of excess return over that month.

    The factor of `sqrt(periods_per_year)` is the part that is easy to get wrong,
    and getting it wrong is not cosmetic. Grinold's rule is stated at the
    forecast horizon, but the covariance in the constraint is annualised -- and
    an expected *return* annualises by multiplying by the number of periods
    while a *volatility* annualises by its square root. Writing
    `ic * annual_vol * score` mixes the two and under-forecasts alpha by
    sqrt(12) at a monthly horizon, which was enough to make the optimiser
    conclude the signal could not pay for its own trading costs and refuse to
    hold a meaningful book.

    Scaling by each name's own volatility also shrinks the forecast on quiet
    names, rather than treating a one-sigma score on a utility and on a biotech
    as the same opportunity.
    """
    standardized = scores.to_numpy()
    dispersion = standardized.std()
    if dispersion > 1e-12:
        standardized = (standardized - standardized.mean()) / dispersion
    annual_vol = np.sqrt(np.clip(np.diag(covariance.to_numpy()), 1e-8, None))
    periods_per_year = 252.0 / horizon_days
    return np.sqrt(periods_per_year) * ic * annual_vol * standardized


def optimize(
    signal: pd.Series,
    covariance: pd.DataFrame,
    betas: pd.Series,
    previous: pd.Series,
    constraints: Constraints,
    objective: Objective,
) -> pd.Series:
    tickers = signal.index.tolist()
    if constraints.max_weight * len(tickers) < constraints.gross_leverage:
        raise ValueError(
            f"Box infeasible: {len(tickers)} names capped at {constraints.max_weight} "
            f"cannot reach gross {constraints.gross_leverage}."
        )

    matrix = covariance.loc[tickers, tickers].fillna(0.0).to_numpy()
    matrix = matrix + np.eye(len(tickers)) * 1e-6
    beta_vector = betas.reindex(tickers).fillna(0.0).to_numpy()
    prior = previous.reindex(tickers).fillna(0.0).to_numpy()

    alpha = expected_returns(
        signal, covariance.loc[tickers, tickers], objective.information_coefficient, objective.horizon_days
    )

    weights = cp.Variable(len(tickers))
    absolute = cp.Variable(len(tickers), nonneg=True)

    # Trading costs are charged in the same annualised units as the alpha, so
    # the optimiser makes the real economic trade-off rather than an arbitrary
    # one. On this signal the two are genuinely comparable, and the penalty
    # correctly suppresses churn that the edge cannot pay for.
    utility = alpha @ weights - objective.turnover_penalty * cp.norm1(weights - prior)

    problem = cp.Problem(
        cp.Maximize(utility),
        [
            cp.sum(weights) == 0.0,                      # dollar neutral
            absolute >= weights,
            absolute >= -weights,
            cp.sum(absolute) <= constraints.gross_leverage,
            absolute <= constraints.max_weight,
            cp.abs(beta_vector @ weights) <= constraints.beta_tolerance,
            # Risk budget rather than a risk-aversion coefficient. With an alpha
            # this small, any hand-picked lambda either ignores risk entirely or
            # collapses the book to zero -- both were observed. Stating the
            # volatility the book is allowed to run is well posed, is what a
            # mandate actually specifies, and leaves nothing to tune.
            cp.quad_form(weights, matrix) <= constraints.vol_target ** 2,
        ],
    )
    problem.solve(solver=objective.solver, verbose=False)

    if weights.value is None:
        raise RuntimeError(f"Optimizer failed: status={problem.status}, solver={objective.solver}")

    return pd.Series(weights.value, index=tickers).round(10)
