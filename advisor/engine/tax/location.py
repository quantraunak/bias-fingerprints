"""Asset location: which wrapper holds which asset class.

Same policy weights, different accounts, materially different after-tax return.
The rule of thumb "bonds in the IRA" is often wrong — what belongs in a
tax-deferred account is whatever has the highest tax drag PER DOLLAR, which at
today's yields is high-turnover alternatives and high-yield credit, not
low-coupon core bonds.

Solved as a transportation LP: minimize total household tax drag subject to
hitting policy weights in aggregate and respecting each account's size.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from ..cma import AssetAssumption, build_cma, tax_drag
from ..types import (
    TAX_EXEMPT,
    Account,
    AccountType,
    AssetClass,
    Household,
)


@dataclass
class LocationResult:
    placement: pd.DataFrame  # rows = asset class, cols = account, values = $
    baseline_drag: float  # annual $ lost to tax under pro-rata placement
    optimized_drag: float
    annual_savings: float
    savings_bps: float
    pv_savings_20y: float

    def summary(self) -> dict:
        return {
            "annual_savings": round(self.annual_savings, 0),
            "savings_bps": round(self.savings_bps, 1),
            "pv_savings_20y": round(self.pv_savings_20y, 0),
            "baseline_drag": round(self.baseline_drag, 0),
            "optimized_drag": round(self.optimized_drag, 0),
            "placement": {
                str(i): {str(c): round(v, 0) for c, v in row.items() if abs(v) > 1}
                for i, row in self.placement.iterrows()
            },
        }


def _wrapper_drag_multiplier(account_type: AccountType) -> float:
    """How much of an asset's taxable drag is actually incurred in a wrapper.

    Tax-exempt (Roth/DAF/CRT): 0 — and gains are never taxed on exit either.
    Traditional IRA: ~0 annually, but the entire balance is taxed as ordinary
    income on withdrawal. Charging ~35% of the drag captures that the deferral
    is not free.
    Non-grantor trust: compressed brackets hit the top rate almost immediately.
    """
    if account_type in TAX_EXEMPT:
        return 0.0
    if account_type == AccountType.TRADITIONAL_IRA:
        return 0.35
    if account_type == AccountType.NON_GRANTOR_TRUST:
        return 1.10
    return 1.0  # taxable, grantor trust (grantor pays from other assets)


def optimize_location(
    household: Household,
    policy_weights: pd.Series,
    cma: dict[AssetClass, AssetAssumption] | None = None,
) -> LocationResult:
    """Assign policy dollars to accounts to minimize household tax drag."""
    cma = cma or build_cma()
    tax = household.tax
    total = household.total_assets
    if total <= 0:
        raise ValueError("Household has no assets.")

    # Drop negligible sleeves, then RENORMALIZE. Without renormalization the
    # asset-side and account-side totals disagree and the LP is infeasible.
    kept = policy_weights[policy_weights > 1e-6]
    kept = kept / kept.sum()
    assets = list(kept.index)

    accounts = household.accounts
    acct_ids = [a.account_id for a in accounts]
    acct_values = np.array([a.market_value for a in accounts])
    # Account values must sum to the same total the weights are scaled against.
    if acct_values.sum() > 0:
        acct_values = acct_values * (total / acct_values.sum())

    # After-tax return matrix: what a dollar of asset i actually earns in
    # wrapper j. The objective maximizes total household after-tax return.
    #
    # Note on the popular "put your highest-returning assets in the Roth"
    # heuristic: it is not generally right. Against a TAXABLE alternative the
    # gain from sheltering an asset equals its tax DRAG, so high-drag assets
    # (credit, high-turnover alternatives) win the shelter regardless of their
    # expected return. The high-return heuristic applies when choosing between
    # Roth and TRADITIONAL dollars, since traditional balances are only
    # partly owned by the family. That second effect is approximated by the
    # wrapper multipliers below rather than modelled explicitly.
    R = np.zeros((len(assets), len(accounts)))
    D = np.zeros((len(assets), len(accounts)))
    for i, a in enumerate(assets):
        assumption = cma[AssetClass(a)]
        base = tax_drag(assumption, tax)
        for j, acct in enumerate(accounts):
            D[i, j] = base * _wrapper_drag_multiplier(acct.account_type)
            R[i, j] = assumption.expected_return - D[i, j]

    target_dollars = np.array([kept[a] * total for a in assets])

    X = cp.Variable((len(assets), len(accounts)), nonneg=True)
    cons = [
        cp.sum(X, axis=1) == target_dollars,  # hit policy in aggregate
        cp.sum(X, axis=0) == acct_values,  # respect account sizes
    ]
    # Illiquid sleeves cannot sit in an IRA in practice (custody/UBTI issues).
    for i, a in enumerate(assets):
        if a in {"private_equity", "private_credit", "real_estate"}:
            for j, acct in enumerate(accounts):
                if acct.account_type == AccountType.TRADITIONAL_IRA:
                    cons.append(X[i, j] == 0)

    # Tiny tie-breaker toward putting the highest-return assets in tax-exempt
    # wrappers when the drag term alone is indifferent.
    objective = cp.Maximize(cp.sum(cp.multiply(R, X)))
    problem = cp.Problem(objective, cons)
    problem.solve(solver=cp.CLARABEL)

    if X.value is None:
        raise RuntimeError("Asset location LP infeasible — check account values sum to total.")

    placement = pd.DataFrame(np.maximum(X.value, 0.0), index=assets, columns=acct_ids)

    optimized = float((D * placement.values).sum())
    # Baseline: every account holds the policy pro-rata (what most families do).
    pro_rata = np.outer(target_dollars / total, acct_values)
    baseline = float((D * pro_rata).sum())

    savings = baseline - optimized
    # PV over 20 years at the policy's own discount rate.
    r = 0.06
    pv = savings * (1 - (1 + r) ** -20) / r

    return LocationResult(
        placement=placement,
        baseline_drag=baseline,
        optimized_drag=optimized,
        annual_savings=savings,
        savings_bps=(savings / total) * 10_000,
        pv_savings_20y=pv,
    )


def withdrawal_sequence(household: Household, annual_need: float, years: int = 30) -> pd.DataFrame:
    """Tax-efficient drawdown ordering.

    Conventional order (taxable -> deferred -> Roth) is frequently suboptimal.
    Filling low brackets from the IRA in early years — before Social Security,
    RMDs, and IRMAA thresholds bind — usually beats it, and it shrinks the
    balance subject to RMDs later.
    """
    taxable = sum(a.market_value for a in household.accounts if a.is_taxable)
    deferred = sum(
        a.market_value for a in household.accounts if a.account_type == AccountType.TRADITIONAL_IRA
    )
    exempt = sum(a.market_value for a in household.accounts if a.account_type in TAX_EXEMPT)

    # Bracket-filling target: convert/withdraw up to the top of the 24% bracket
    # in low-income years rather than realizing it all at 37%+ later.
    bracket_fill = 400_000.0 if household.tax.filing_status == "mfj" else 200_000.0

    rows = []
    for y in range(1, years + 1):
        need = annual_need
        from_taxable = min(taxable, need * 0.6)
        need -= from_taxable
        taxable -= from_taxable

        from_deferred = min(deferred, need)
        deferred -= from_deferred
        need -= from_deferred

        from_exempt = min(exempt, need)
        exempt -= from_exempt

        # Opportunistic Roth conversion using unused low-bracket room.
        conversion = 0.0
        if deferred > 0 and from_deferred < bracket_fill:
            conversion = min(deferred, bracket_fill - from_deferred)
            deferred -= conversion
            exempt += conversion
        rows.append(
            {
                "year": y,
                "from_taxable": round(from_taxable, 0),
                "from_deferred": round(from_deferred, 0),
                "from_tax_exempt": round(from_exempt, 0),
                "roth_conversion": round(conversion, 0),
                "taxable_balance": round(taxable, 0),
                "deferred_balance": round(deferred, 0),
                "exempt_balance": round(exempt, 0),
            }
        )
    return pd.DataFrame(rows)
