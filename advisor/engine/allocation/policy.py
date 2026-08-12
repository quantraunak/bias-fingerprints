"""Strategic asset allocation under real family-office constraints.

The optimizer maximizes AFTER-TAX return per unit of risk, subject to the two
constraints that actually bind for a $10-50M household:

  1. Liquidity. Illiquid assets must not exceed what the family can fund
     through a drawdown while still meeting spending and unfunded capital calls.
  2. Drawdown tolerance. The stated max-drawdown limit is converted to a
     volatility budget; a policy the family abandons at the bottom has a
     realized return of zero regardless of its expected return.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from ..cma import AssetAssumption, after_tax_return, build_cma, covariance_matrix
from ..types import AssetClass, Household, TaxProfile
from .black_litterman import (
    View,
    black_litterman,
    calibrate_risk_aversion,
    default_market_weights,
)

# Illiquid sleeves. Capacity here is what forces the liquidity constraint.
ILLIQUID = {"private_equity", "private_credit", "real_estate"}

# Governance bounds. A mean-variance optimizer left unbounded will happily put
# a third of a family's wealth in whichever asset has the most flattering
# Sharpe estimate. These caps encode what an investment committee would
# actually approve, and they are the difference between a usable policy and a
# spreadsheet artifact.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "us_large": (0.10, 0.45),
    "us_small": (0.00, 0.12),
    "intl_dev": (0.05, 0.25),
    "em": (0.00, 0.12),
    "core_bond": (0.02, 0.30),
    "tips": (0.00, 0.15),
    "hy_credit": (0.00, 0.08),
    "private_equity": (0.00, 0.20),
    "private_credit": (0.00, 0.12),
    "real_estate": (0.00, 0.12),
    "market_neutral": (0.00, 0.15),
    "trend": (0.00, 0.10),
    "cash": (0.01, 0.12),
}


@dataclass
class PolicyConstraints:
    """Bounds are per asset class; anything unspecified gets (0, max_single)."""

    max_single: float = 0.35
    min_cash: float = 0.01
    max_illiquid: float | None = None  # None -> derived from liquidity needs
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))
    # Tracking-error-style anchor: keep the policy within reach of the market
    # portfolio so it stays defensible.
    max_deviation_from_market: float = 0.30

    def bound_for(self, asset: str) -> tuple[float, float]:
        return self.bounds.get(asset, (0.0, self.max_single))


@dataclass
class PolicyResult:
    weights: pd.Series
    expected_return: float  # pre-tax
    after_tax_return: float
    volatility: float
    sharpe: float
    illiquid_pct: float
    max_drawdown_estimate: float
    vol_budget: float
    binding_constraints: list[str]
    equilibrium_returns: pd.Series
    posterior_returns: pd.Series

    def summary(self) -> dict:
        return {
            "expected_return": round(self.expected_return, 4),
            "after_tax_return": round(self.after_tax_return, 4),
            "volatility": round(self.volatility, 4),
            "sharpe": round(self.sharpe, 3),
            "illiquid_pct": round(self.illiquid_pct, 4),
            "max_drawdown_estimate": round(self.max_drawdown_estimate, 4),
            "vol_budget": round(self.vol_budget, 4),
            "binding_constraints": self.binding_constraints,
            "weights": {k: round(v, 4) for k, v in self.weights.items() if v > 0.0005},
        }


def volatility_budget(max_drawdown_tolerance: float, horizon_years: int = 30) -> float:
    """Convert a drawdown tolerance into an annual volatility ceiling.

    For a diversified portfolio the expected worst drawdown over a multi-decade
    horizon runs roughly 2.2-2.6x annual volatility (fat tails and serial
    correlation push it above the lognormal result). We use 2.4x, which puts a
    25% tolerance at roughly 10.4% vol.
    """
    multiple = 2.4 if horizon_years >= 20 else 2.1
    return max(max_drawdown_tolerance / multiple, 0.02)


def liquidity_capacity(household: Household) -> float:
    """Max illiquid share of assets that still leaves the family solvent.

    Reserve must cover: spending through a drawdown, unfunded capital calls,
    and a haircut on liquid assets in a stress (public equities -35%).
    """
    total = household.total_assets
    if total <= 0:
        return 0.0

    net_spend = max(household.annual_spending - household.annual_income, 0.0)
    spending_reserve = net_spend * max(household.liquidity_reserve_years, 1.0)
    call_reserve = household.unfunded_commitments
    # In stress, liquid assets are worth less exactly when calls arrive.
    stress_haircut = 0.35

    required_liquid = (spending_reserve + call_reserve) / (1 - stress_haircut)
    capacity = 1.0 - (required_liquid / total)
    # Governance cap: even with infinite capacity, do not exceed 45% illiquid.
    return float(np.clip(capacity, 0.0, 0.45))


def optimize_policy(
    household: Household,
    views: list[View] | None = None,
    constraints: PolicyConstraints | None = None,
    cma: dict[AssetClass, AssetAssumption] | None = None,
    risk_aversion: float = 2.6,
    tau: float = 0.05,
) -> PolicyResult:
    """Solve for the strategic asset allocation."""
    cma = cma or build_cma()
    constraints = constraints or PolicyConstraints()
    views = views or []
    tax: TaxProfile = household.tax

    cov = covariance_matrix(cma)
    assets = list(cov.index)
    mkt = default_market_weights().reindex(assets).fillna(0.0)
    mkt = mkt / mkt.sum()

    # Everything below runs in EXCESS-return space, then cash is added back.
    cash_rate = cma[AssetClass.CASH].expected_return
    cma_excess = pd.Series(
        {a: cma[AssetClass(a)].expected_return - cash_rate for a in assets}
    )

    # Anchor the equilibrium to the CMA's own view of the market portfolio, so
    # the level of expected returns is set by the building-block forecasts
    # rather than by an arbitrary risk-aversion constant.
    market_excess = float((cma_excess * mkt).sum())
    delta = calibrate_risk_aversion(cov, mkt, market_excess)

    # The CMA enters as a set of absolute views on the equilibrium prior. This
    # is what makes the two forecasting systems one coherent posterior instead
    # of two competing numbers: the market sets the cross-sectional shape, the
    # building-block CMA pulls each asset toward its fundamental estimate.
    cma_views = [
        View(
            weights={a: 1.0},
            expected_return=float(cma_excess[a]),
            confidence=0.35,
            rationale=f"Building-block CMA for {a}",
        )
        for a in assets
    ]
    all_views = cma_views + views

    posterior_excess, posterior_cov = black_litterman(cov, mkt, all_views, delta, tau)
    equilibrium = implied_prior = black_litterman(cov, mkt, [], delta, tau)[0] + cash_rate
    posterior_mu = posterior_excess + cash_rate

    # Convert to after-tax expected returns. Taxable share of the balance sheet
    # determines how much the tax drag actually bites at the household level.
    taxable_share = household.taxable_assets / household.total_assets if household.total_assets else 1.0
    mu_at = {}
    for a in assets:
        ac = AssetClass(a)
        assumption = cma[ac]
        drag = assumption.expected_return - after_tax_return(assumption, tax, True)
        mu_at[a] = posterior_mu[a] - drag * taxable_share
    mu = pd.Series(mu_at).reindex(assets)

    vol_cap = volatility_budget(household.max_drawdown_tolerance, household.horizon_years)
    illiquid_cap = (
        constraints.max_illiquid
        if constraints.max_illiquid is not None
        else liquidity_capacity(household)
    )

    n = len(assets)
    Sigma = posterior_cov.values
    Sigma = (Sigma + Sigma.T) / 2 + np.eye(n) * 1e-8

    w = cp.Variable(n)
    lo = np.array([constraints.bound_for(a)[0] for a in assets])
    hi = np.array([constraints.bound_for(a)[1] for a in assets])
    illiquid_mask = np.array([1.0 if a in ILLIQUID else 0.0 for a in assets])
    cash_idx = assets.index("cash")

    cons = [
        cp.sum(w) == 1.0,
        w >= lo,
        w <= hi,
        w[cash_idx] >= constraints.min_cash,
        illiquid_mask @ w <= illiquid_cap,
        cp.quad_form(w, Sigma) <= vol_cap**2,
        cp.norm1(w - mkt.values) <= constraints.max_deviation_from_market * 2,
    ]

    # Maximize after-tax return; the vol budget is a hard constraint, so a small
    # risk penalty only breaks ties between equally-returning portfolios.
    objective = cp.Maximize(mu.values @ w - 0.25 * cp.quad_form(w, Sigma))
    problem = cp.Problem(objective, cons)
    problem.solve(solver=cp.CLARABEL)

    if w.value is None:
        # Relax the vol budget rather than fail: report it as binding.
        cons[-2] = cp.quad_form(w, Sigma) <= (vol_cap * 1.5) ** 2
        problem = cp.Problem(objective, cons)
        problem.solve(solver=cp.CLARABEL)
    if w.value is None:
        raise RuntimeError("Policy optimization infeasible even after relaxation.")

    weights = pd.Series(np.clip(w.value, 0, None), index=assets)
    weights = weights / weights.sum()

    port_var = float(weights.values @ Sigma @ weights.values)
    port_vol = float(np.sqrt(port_var))
    pretax = float((posterior_mu.reindex(assets) * weights).sum())
    aftertax = float((mu * weights).sum())
    illiquid_pct = float(illiquid_mask @ weights.values)

    binding = []
    if port_vol > vol_cap * 0.98:
        binding.append("volatility_budget")
    if illiquid_pct > illiquid_cap * 0.98:
        binding.append("liquidity_capacity")
    for i, a in enumerate(assets):
        if weights[a] > hi[i] * 0.99 and hi[i] < constraints.max_single:
            binding.append(f"max_weight:{a}")

    return PolicyResult(
        weights=weights.sort_values(ascending=False),
        expected_return=pretax,
        after_tax_return=aftertax,
        volatility=port_vol,
        sharpe=(pretax - cash_rate) / port_vol if port_vol else 0.0,
        illiquid_pct=illiquid_pct,
        max_drawdown_estimate=-port_vol * 2.4,
        vol_budget=vol_cap,
        binding_constraints=binding,
        equilibrium_returns=equilibrium,
        posterior_returns=posterior_mu,
    )


def efficient_frontier(
    household: Household,
    n_points: int = 12,
    views: list[View] | None = None,
    cma: dict[AssetClass, AssetAssumption] | None = None,
) -> pd.DataFrame:
    """Trace after-tax return vs volatility across drawdown tolerances."""
    rows = []
    base_dd = household.max_drawdown_tolerance
    for dd in np.linspace(0.08, 0.45, n_points):
        household.max_drawdown_tolerance = float(dd)
        try:
            r = optimize_policy(household, views=views, cma=cma)
        except RuntimeError:
            continue
        rows.append(
            {
                "drawdown_tolerance": round(float(dd), 4),
                "volatility": round(r.volatility, 4),
                "expected_return": round(r.expected_return, 4),
                "after_tax_return": round(r.after_tax_return, 4),
                "sharpe": round(r.sharpe, 3),
                "illiquid_pct": round(r.illiquid_pct, 4),
                "equity_pct": round(
                    float(r.weights.reindex(["us_large", "us_small", "intl_dev", "em"]).fillna(0).sum()), 4
                ),
            }
        )
    household.max_drawdown_tolerance = base_dd
    return pd.DataFrame(rows)


def gap_analysis(household: Household, target: pd.Series) -> pd.DataFrame:
    """Current vs policy weights, with the tax cost of closing each gap."""
    current = household.current_allocation()
    total = household.total_assets
    all_assets = sorted(set(current.index) | set(target.index))

    # Embedded gain by asset class drives the cost of selling down an overweight.
    gains: dict[str, float] = {}
    values: dict[str, float] = {}
    for h in household.holdings:
        k = h.asset_class.value
        gains[k] = gains.get(k, 0.0) + h.unrealized_gain
        values[k] = values.get(k, 0.0) + h.market_value

    rows = []
    for a in all_assets:
        cur = float(current.get(a, 0.0))
        tgt = float(target.get(a, 0.0))
        drift = tgt - cur
        dollars = drift * total
        gain_pct = (gains.get(a, 0.0) / values[a]) if values.get(a, 0.0) > 0 else 0.0
        tax_cost = 0.0
        if dollars < 0:  # selling
            tax_cost = abs(dollars) * max(gain_pct, 0.0) * household.tax.ltcg_rate
        rows.append(
            {
                "asset_class": a,
                "current": round(cur, 4),
                "policy": round(tgt, 4),
                "gap": round(drift, 4),
                "trade_dollars": round(dollars, 0),
                "embedded_gain_pct": round(gain_pct, 4),
                "tax_cost_to_close": round(tax_cost, 0),
            }
        )
    df = pd.DataFrame(rows).set_index("asset_class")
    return df.reindex(df["gap"].abs().sort_values(ascending=False).index)
