"""Mean-variance optimisation under dollar-, beta-, industry- and box-constraints.

Three findings from the previous engine are kept as comments because each one
produced plausible-looking output while being wrong.

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

**Risk belongs in the objective, and the risk-aversion term is solved for.**
An earlier version moved risk to a constraint (`w'Sigma w <= target^2`) because
no hand-picked lambda worked -- every value either ignored risk or collapsed the
book to zero. That reasoning was right about hand-picking and wrong about the
conclusion. A linear objective over a box constraint has its optimum at a
vertex, so the solver piled onto `max_weight` for a handful of extreme names and
discarded the ordering information across the rest of the cross-section, which
is most of the signal. Bisecting on lambda until the realised volatility hits
the budget gives the same risk with a book spread across the whole
cross-section. There is nothing to tune: the target volatility is stated, and
lambda is whatever achieves it.
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
    industry_tolerance: float = 0.0  # 0 disables the industry constraint


@dataclass(frozen=True)
class Objective:
    turnover_penalty: float
    solver: str = "CLARABEL"
    information_coefficient: float = 0.02
    horizon_days: int = 21
    risk_in_objective: bool = True
    bisection_steps: int = 18


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


def _industry_matrix(tickers: list[str], industries: pd.Series | None) -> np.ndarray | None:
    """One row per industry, 1 where the name belongs to it.

    Constraining these rows to ~0 removes industry composition from the book.
    The beta constraint alone does not: a long-cheap-cyclicals, short-expensive-
    defensives tilt can be market-beta-neutral and still be an industry bet, and
    that is precisely the exposure the decile decomposition attributes 59% of the
    apparent edge to.
    """
    if industries is None:
        return None
    mapped = industries.reindex(tickers)
    if mapped.isna().all():
        return None
    labels = [g for g in mapped.dropna().unique()]
    if len(labels) < 2:
        return None
    rows = [(mapped == label).to_numpy(dtype=float) for label in labels]
    return np.vstack(rows)


def optimize(
    signal: pd.Series,
    covariance: pd.DataFrame,
    betas: pd.Series,
    previous: pd.Series,
    constraints: Constraints,
    objective: Objective,
    industries: pd.Series | None = None,
) -> pd.Series:
    tickers = signal.index.tolist()
    if constraints.max_weight * len(tickers) < constraints.gross_leverage:
        raise ValueError(
            f"Box infeasible: {len(tickers)} names capped at {constraints.max_weight} "
            f"cannot reach gross {constraints.gross_leverage}."
        )

    matrix = covariance.loc[tickers, tickers].fillna(0.0).to_numpy()
    matrix = matrix + np.eye(len(tickers)) * 1e-6
    # Ledoit-Wolf output is positive semi-definite by construction -- it is a
    # convex combination of a sample covariance and a scaled identity -- but
    # CVXPY re-certifies it numerically with ARPACK, which fails to converge on
    # matrices of a few hundred names. Asserting what shrinkage already
    # guarantees avoids a solver error that only appears once the cross-section
    # is wide enough to matter.
    psd_matrix = cp.psd_wrap(matrix)
    beta_vector = betas.reindex(tickers).fillna(0.0).to_numpy()
    prior = previous.reindex(tickers).fillna(0.0).to_numpy()
    industry_rows = _industry_matrix(tickers, industries)

    alpha = expected_returns(
        signal, covariance.loc[tickers, tickers], objective.information_coefficient, objective.horizon_days
    )

    def build(risk_aversion: float | None):
        weights = cp.Variable(len(tickers))
        absolute = cp.Variable(len(tickers), nonneg=True)

        utility = alpha @ weights - objective.turnover_penalty * cp.norm1(weights - prior)
        limits = [
            cp.sum(weights) == 0.0,                      # dollar neutral
            absolute >= weights,
            absolute >= -weights,
            cp.sum(absolute) <= constraints.gross_leverage,
            absolute <= constraints.max_weight,
            cp.abs(beta_vector @ weights) <= constraints.beta_tolerance,
        ]
        if industry_rows is not None and constraints.industry_tolerance >= 0:
            limits.append(
                cp.abs(industry_rows @ weights) <= constraints.industry_tolerance
            )

        if risk_aversion is None:
            limits.append(cp.quad_form(weights, psd_matrix) <= constraints.vol_target ** 2)
        else:
            utility = utility - risk_aversion * cp.quad_form(weights, psd_matrix)

        problem = cp.Problem(cp.Maximize(utility), limits)
        problem.solve(solver=objective.solver, verbose=False)
        if weights.value is None:
            raise RuntimeError(f"Optimizer failed: status={problem.status}, solver={objective.solver}")
        return np.asarray(weights.value)

    if not objective.risk_in_objective:
        return pd.Series(build(None), index=tickers).round(10)

    solution = _bisect_to_vol_target(build, matrix, constraints.vol_target, objective.bisection_steps)
    return pd.Series(solution, index=tickers).round(10)


def _bisect_to_vol_target(
    build, matrix: np.ndarray, vol_target: float, steps: int
) -> np.ndarray:
    """Find the risk aversion whose optimum runs at the target volatility.

    Volatility falls monotonically in lambda, so bisection is well posed and
    needs no starting guess beyond a bracket. The bracket is widened rather than
    assumed: alpha and covariance scales vary across rebalance dates, and a fixed
    bracket silently returns its own endpoint on the dates where it does not
    contain the answer.
    """
    def realised(weights: np.ndarray) -> float:
        return float(np.sqrt(max(weights @ matrix @ weights, 0.0)))

    low, high = 1e-6, 1.0
    solution = build(high)
    for _ in range(40):
        if realised(solution) <= vol_target:
            break
        high *= 4.0
        solution = build(high)

    best = solution
    for _ in range(steps):
        mid = np.sqrt(low * high)  # geometric: lambda spans orders of magnitude
        candidate = build(mid)
        if realised(candidate) > vol_target:
            low = mid
        else:
            high = mid
            best = candidate
    return best


def transfer_coefficient(alpha: np.ndarray, weights: np.ndarray, matrix: np.ndarray) -> float:
    """Risk-adjusted correlation between the forecast and the implemented book.

    Grinold and Kahn's measure of how much of a signal survives implementation.
    A low value says the constraints, not the signal, are determining the book --
    which is an implementation problem and fixable, and worth separating from a
    signal problem, which is not.
    """
    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return float("nan")
    scaled_alpha = np.linalg.solve(chol, alpha)
    scaled_weights = chol.T @ weights
    denominator = np.linalg.norm(scaled_alpha) * np.linalg.norm(scaled_weights)
    if denominator < 1e-12:
        return float("nan")
    return float(scaled_alpha @ scaled_weights / denominator)
