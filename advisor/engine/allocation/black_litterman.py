"""Black-Litterman posterior expected returns.

Raw mean-variance optimization on hand-forecast returns produces absurd corner
portfolios — small input errors get amplified by the inverse covariance matrix.
Black-Litterman fixes this by starting from the market's own implied returns
and moving only where you have an explicit, quantified view.

    pi     = delta * Sigma * w_mkt              (reverse optimization)
    E[R]   = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1
             [(tau*Sigma)^-1 pi + P' Omega^-1 Q]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class View:
    """A quantified market view.

    Absolute view:  {"em": 1.0}, expected_return=0.09
    Relative view:  {"intl_dev": 1.0, "us_large": -1.0}, expected_return=0.015
                    ("international beats US large by 150bps")

    `confidence` in (0, 1]: 1.0 means the view is as certain as the prior.
    """

    weights: dict[str, float]
    expected_return: float
    confidence: float = 0.5
    rationale: str = ""


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.6,
) -> pd.Series:
    """Reverse-optimize the market portfolio into the returns that justify it.

    Everything in this module lives in EXCESS-return space (over cash). The
    caller adds the cash rate back to get total returns.
    """
    w = market_weights.reindex(cov.index).fillna(0.0)
    pi = risk_aversion * cov.values @ w.values
    return pd.Series(pi, index=cov.index, name="equilibrium_excess_return")


def calibrate_risk_aversion(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    market_excess_return: float,
) -> float:
    """Solve for the delta that makes the equilibrium consistent with the CMA.

    delta = market excess return / market variance. Without this step the
    equilibrium returns float at an arbitrary level set by whatever risk
    aversion was hard-coded, and the whole portfolio expected return is wrong.
    """
    w = market_weights.reindex(cov.index).fillna(0.0)
    var = float(w.values @ cov.values @ w.values)
    if var <= 0:
        return 2.6
    return float(np.clip(market_excess_return / var, 0.5, 12.0))


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    views: list[View],
    risk_aversion: float = 2.6,
    tau: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (posterior expected returns, posterior covariance)."""
    assets = list(cov.index)
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion)
    Sigma = cov.values
    tau_sigma = tau * Sigma

    if not views:
        return pi, cov.copy()

    P = np.zeros((len(views), len(assets)))
    Q = np.zeros(len(views))
    omega_diag = np.zeros(len(views))

    for i, v in enumerate(views):
        for asset, w in v.weights.items():
            if asset not in cov.index:
                raise KeyError(f"View references unknown asset '{asset}'")
            P[i, assets.index(asset)] = w
        Q[i] = v.expected_return
        # Idzorek-style: confidence scales the view's variance relative to the
        # prior variance of that same portfolio.
        prior_var = float(P[i] @ tau_sigma @ P[i].T)
        c = min(max(v.confidence, 1e-3), 1.0)
        omega_diag[i] = prior_var * (1.0 / c - 1.0) + 1e-10

    Omega = np.diag(omega_diag)
    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(Omega)

    posterior_cov_returns = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
    mu = posterior_cov_returns @ (inv_tau_sigma @ pi.values + P.T @ inv_omega @ Q)

    # Posterior covariance of returns includes parameter uncertainty.
    post_sigma = Sigma + posterior_cov_returns

    return (
        pd.Series(mu, index=cov.index, name="posterior_return"),
        pd.DataFrame(post_sigma, index=cov.index, columns=cov.columns),
    )


def default_market_weights() -> pd.Series:
    """Approximate global multi-asset market cap weights, extended to include
    alternatives at their realistic investable size."""
    return pd.Series(
        {
            "us_large": 0.310,
            "us_small": 0.045,
            "intl_dev": 0.135,
            "em": 0.055,
            "core_bond": 0.230,
            "tips": 0.030,
            "hy_credit": 0.040,
            "private_equity": 0.055,
            "private_credit": 0.030,
            "real_estate": 0.045,
            "market_neutral": 0.010,
            "trend": 0.005,
            "cash": 0.010,
        }
    )
