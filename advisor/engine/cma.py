"""Forward-looking capital market assumptions.

Historical average returns are a bad forecast — they are highest exactly when
valuations are richest. Every expected return here is built bottom-up from
observable inputs (yields, payout, growth, valuation drift) so the assumptions
can be defended line by line in an investment committee.

Equities   Grinold-Kroner:  E[r] = DY + (1+g)(1+i) - 1 - dilution + ΔPE
Bonds      Starting YTM is the dominant predictor of 10y return (R² ~ 0.90)
Privates   Public proxy + illiquidity/leverage premium - fees, vol de-smoothed
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .types import AssetClass


@dataclass
class MarketInputs:
    """Observable starting conditions. Update these quarterly — they are the
    only thing that should move the CMA."""

    inflation: float = 0.024  # 10y breakeven
    cash_rate: float = 0.037  # 3m T-bill
    ust10_yield: float = 0.042
    tips_real_yield: float = 0.019
    ig_spread: float = 0.009
    hy_spread: float = 0.031
    us_dividend_yield: float = 0.013
    us_buyback_yield: float = 0.019
    us_real_eps_growth: float = 0.021
    us_cape: float = 34.0
    # Structural fair value, NOT the 1900- mean. Lower accounting-quality
    # adjustments, higher index margins and lower real rates all justify a
    # permanently higher multiple than history; assuming full reversion to 16x
    # has been the single most costly CMA error of the last two decades.
    us_cape_fair: float = 28.0
    intl_dividend_yield: float = 0.031
    intl_real_eps_growth: float = 0.014
    intl_cape: float = 17.5
    intl_cape_fair: float = 17.0
    em_dividend_yield: float = 0.029
    em_real_eps_growth: float = 0.028
    em_cape: float = 14.0
    em_cape_fair: float = 15.0
    reversion_years: int = 15  # years over which valuation drifts to fair


@dataclass
class AssetAssumption:
    asset_class: AssetClass
    expected_return: float  # geometric, nominal, net of manager fees
    volatility: float
    yield_component: float  # portion arriving as taxable income each year
    income_is_ordinary: bool  # True -> taxed at ordinary rates
    turnover: float  # fraction of NAV realizing gains annually
    liquidity_days: int  # days to liquidate without material impact
    fee: float = 0.0  # already deducted from expected_return; shown for audit

    @property
    def sharpe(self) -> float:
        return self.expected_return / self.volatility if self.volatility else 0.0


def _equity_return(
    dividend_yield: float,
    buyback_yield: float,
    real_growth: float,
    inflation: float,
    cape: float,
    cape_fair: float,
    years: int,
) -> float:
    """Grinold-Kroner with a valuation drift term."""
    income = dividend_yield + buyback_yield
    nominal_growth = (1 + real_growth) * (1 + inflation) - 1
    # Annualized multiple change as CAPE drifts toward fair value.
    valuation_drift = (cape_fair / cape) ** (1 / years) - 1
    return income + nominal_growth + valuation_drift


def build_cma(inputs: MarketInputs | None = None) -> dict[AssetClass, AssetAssumption]:
    """Produce the full assumption set from market inputs."""
    m = inputs or MarketInputs()
    y = m.reversion_years

    us_large = _equity_return(
        m.us_dividend_yield, m.us_buyback_yield, m.us_real_eps_growth,
        m.inflation, m.us_cape, m.us_cape_fair, y,
    )
    intl = _equity_return(
        m.intl_dividend_yield, 0.005, m.intl_real_eps_growth,
        m.inflation, m.intl_cape, m.intl_cape_fair, y,
    )
    em = _equity_return(
        m.em_dividend_yield, 0.002, m.em_real_eps_growth,
        m.inflation, m.em_cape, m.em_cape_fair, y,
    )

    # Small cap: size premium is thin after quality screens; +60bps over large,
    # but materially higher vol. Do not over-promise it.
    us_small = us_large + 0.006

    core_bond_ytm = m.ust10_yield + 0.4 * m.ig_spread
    # HY: spread less expected credit losses (~200bps through-cycle at this spread).
    hy = m.ust10_yield + m.hy_spread - 0.020

    # Private equity: levered small/mid-cap equity. Honest build is public
    # small-cap + leverage effect + operational/selection alpha - fees.
    # Assume top-half-of-market access (not top-decile — that is not investable
    # for a $10-50M family office without a strong platform).
    pe_gross = us_small + 0.045
    pe_fees = 0.020 + 0.20 * max(pe_gross - 0.08, 0.0)  # mgmt + carry over 8% pref
    pe_net = pe_gross - pe_fees

    # Private credit: SOFR + spread, less losses, less fees.
    pc_gross = m.cash_rate + 0.055
    pc_net = pc_gross - 0.012 - 0.015  # credit losses, fees

    # Core real estate: cap rate + inflation-linked NOI growth - capex drag.
    real_estate = 0.052 + m.inflation - 0.008

    # Market-neutral sleeve: cash + alpha. Calibrated to a realistic net Sharpe
    # of ~0.6 AFTER fees and capacity haircuts, NOT to backtest Sharpe.
    mn_vol = 0.08
    market_neutral = m.cash_rate + 0.60 * mn_vol

    # Managed futures / trend: small positive premium, valued for its crisis
    # convexity rather than its standalone return.
    trend = m.cash_rate + 0.030

    rows = [
        AssetAssumption(AssetClass.US_LARGE, us_large, 0.158, m.us_dividend_yield, False, 0.05, 1, 0.0004),
        AssetAssumption(AssetClass.US_SMALL, us_small, 0.205, 0.013, False, 0.10, 2, 0.0008),
        AssetAssumption(AssetClass.INTL_DEV, intl, 0.170, m.intl_dividend_yield, False, 0.08, 2, 0.0006),
        AssetAssumption(AssetClass.EM, em, 0.215, m.em_dividend_yield, False, 0.12, 3, 0.0012),
        AssetAssumption(AssetClass.CORE_BOND, core_bond_ytm, 0.058, core_bond_ytm, True, 0.30, 1, 0.0005),
        AssetAssumption(AssetClass.TIPS, m.tips_real_yield + m.inflation, 0.062, 0.020, True, 0.25, 1, 0.0006),
        AssetAssumption(AssetClass.HY_CREDIT, hy, 0.098, hy, True, 0.40, 3, 0.0035),
        AssetAssumption(AssetClass.PRIVATE_EQUITY, pe_net, 0.235, 0.0, False, 0.0, 2555, pe_fees),
        AssetAssumption(AssetClass.PRIVATE_CREDIT, pc_net, 0.105, pc_gross - 0.012, True, 0.0, 365, 0.015),
        AssetAssumption(AssetClass.REAL_ESTATE, real_estate, 0.145, 0.038, True, 0.0, 180, 0.011),
        AssetAssumption(AssetClass.MARKET_NEUTRAL, market_neutral, mn_vol, 0.0, True, 4.0, 30, 0.020),
        AssetAssumption(AssetClass.TREND, trend, 0.125, 0.0, True, 6.0, 5, 0.015),
        AssetAssumption(AssetClass.CASH, m.cash_rate, 0.006, m.cash_rate, True, 0.0, 0, 0.0002),
    ]
    return {r.asset_class: r for r in rows}


# Correlations. Private-market entries are DE-SMOOTHED: reported private returns
# are appraisal-based and understate true correlation to public equity, which is
# the single most common error in family-office risk models.
_CORR_SPEC: dict[tuple[AssetClass, AssetClass], float] = {
    (AssetClass.US_LARGE, AssetClass.US_SMALL): 0.89,
    (AssetClass.US_LARGE, AssetClass.INTL_DEV): 0.84,
    (AssetClass.US_LARGE, AssetClass.EM): 0.72,
    (AssetClass.US_LARGE, AssetClass.CORE_BOND): 0.10,
    (AssetClass.US_LARGE, AssetClass.TIPS): 0.12,
    (AssetClass.US_LARGE, AssetClass.HY_CREDIT): 0.72,
    (AssetClass.US_LARGE, AssetClass.PRIVATE_EQUITY): 0.80,
    (AssetClass.US_LARGE, AssetClass.PRIVATE_CREDIT): 0.55,
    (AssetClass.US_LARGE, AssetClass.REAL_ESTATE): 0.55,
    (AssetClass.US_LARGE, AssetClass.MARKET_NEUTRAL): 0.05,
    (AssetClass.US_LARGE, AssetClass.TREND): -0.10,
    (AssetClass.US_LARGE, AssetClass.CASH): 0.0,
    (AssetClass.US_SMALL, AssetClass.INTL_DEV): 0.76,
    (AssetClass.US_SMALL, AssetClass.EM): 0.68,
    (AssetClass.US_SMALL, AssetClass.CORE_BOND): 0.05,
    (AssetClass.US_SMALL, AssetClass.TIPS): 0.08,
    (AssetClass.US_SMALL, AssetClass.HY_CREDIT): 0.70,
    (AssetClass.US_SMALL, AssetClass.PRIVATE_EQUITY): 0.85,
    (AssetClass.US_SMALL, AssetClass.PRIVATE_CREDIT): 0.55,
    (AssetClass.US_SMALL, AssetClass.REAL_ESTATE): 0.52,
    (AssetClass.US_SMALL, AssetClass.MARKET_NEUTRAL): 0.05,
    (AssetClass.US_SMALL, AssetClass.TREND): -0.08,
    (AssetClass.US_SMALL, AssetClass.CASH): 0.0,
    (AssetClass.INTL_DEV, AssetClass.EM): 0.80,
    (AssetClass.INTL_DEV, AssetClass.CORE_BOND): 0.12,
    (AssetClass.INTL_DEV, AssetClass.TIPS): 0.15,
    (AssetClass.INTL_DEV, AssetClass.HY_CREDIT): 0.68,
    (AssetClass.INTL_DEV, AssetClass.PRIVATE_EQUITY): 0.70,
    (AssetClass.INTL_DEV, AssetClass.PRIVATE_CREDIT): 0.48,
    (AssetClass.INTL_DEV, AssetClass.REAL_ESTATE): 0.50,
    (AssetClass.INTL_DEV, AssetClass.MARKET_NEUTRAL): 0.05,
    (AssetClass.INTL_DEV, AssetClass.TREND): -0.06,
    (AssetClass.INTL_DEV, AssetClass.CASH): 0.0,
    (AssetClass.EM, AssetClass.CORE_BOND): 0.10,
    (AssetClass.EM, AssetClass.TIPS): 0.18,
    (AssetClass.EM, AssetClass.HY_CREDIT): 0.65,
    (AssetClass.EM, AssetClass.PRIVATE_EQUITY): 0.62,
    (AssetClass.EM, AssetClass.PRIVATE_CREDIT): 0.42,
    (AssetClass.EM, AssetClass.REAL_ESTATE): 0.45,
    (AssetClass.EM, AssetClass.MARKET_NEUTRAL): 0.04,
    (AssetClass.EM, AssetClass.TREND): -0.05,
    (AssetClass.EM, AssetClass.CASH): 0.0,
    (AssetClass.CORE_BOND, AssetClass.TIPS): 0.82,
    (AssetClass.CORE_BOND, AssetClass.HY_CREDIT): 0.28,
    (AssetClass.CORE_BOND, AssetClass.PRIVATE_EQUITY): 0.05,
    (AssetClass.CORE_BOND, AssetClass.PRIVATE_CREDIT): 0.20,
    (AssetClass.CORE_BOND, AssetClass.REAL_ESTATE): 0.18,
    (AssetClass.CORE_BOND, AssetClass.MARKET_NEUTRAL): 0.05,
    (AssetClass.CORE_BOND, AssetClass.TREND): 0.25,
    (AssetClass.CORE_BOND, AssetClass.CASH): 0.15,
    (AssetClass.TIPS, AssetClass.HY_CREDIT): 0.30,
    (AssetClass.TIPS, AssetClass.PRIVATE_EQUITY): 0.06,
    (AssetClass.TIPS, AssetClass.PRIVATE_CREDIT): 0.22,
    (AssetClass.TIPS, AssetClass.REAL_ESTATE): 0.28,
    (AssetClass.TIPS, AssetClass.MARKET_NEUTRAL): 0.05,
    (AssetClass.TIPS, AssetClass.TREND): 0.20,
    (AssetClass.TIPS, AssetClass.CASH): 0.12,
    (AssetClass.HY_CREDIT, AssetClass.PRIVATE_EQUITY): 0.62,
    (AssetClass.HY_CREDIT, AssetClass.PRIVATE_CREDIT): 0.78,
    (AssetClass.HY_CREDIT, AssetClass.REAL_ESTATE): 0.48,
    (AssetClass.HY_CREDIT, AssetClass.MARKET_NEUTRAL): 0.08,
    (AssetClass.HY_CREDIT, AssetClass.TREND): -0.12,
    (AssetClass.HY_CREDIT, AssetClass.CASH): 0.02,
    (AssetClass.PRIVATE_EQUITY, AssetClass.PRIVATE_CREDIT): 0.55,
    (AssetClass.PRIVATE_EQUITY, AssetClass.REAL_ESTATE): 0.52,
    (AssetClass.PRIVATE_EQUITY, AssetClass.MARKET_NEUTRAL): 0.05,
    (AssetClass.PRIVATE_EQUITY, AssetClass.TREND): -0.08,
    (AssetClass.PRIVATE_EQUITY, AssetClass.CASH): 0.0,
    (AssetClass.PRIVATE_CREDIT, AssetClass.REAL_ESTATE): 0.45,
    (AssetClass.PRIVATE_CREDIT, AssetClass.MARKET_NEUTRAL): 0.08,
    (AssetClass.PRIVATE_CREDIT, AssetClass.TREND): -0.05,
    (AssetClass.PRIVATE_CREDIT, AssetClass.CASH): 0.10,
    (AssetClass.REAL_ESTATE, AssetClass.MARKET_NEUTRAL): 0.06,
    (AssetClass.REAL_ESTATE, AssetClass.TREND): -0.02,
    (AssetClass.REAL_ESTATE, AssetClass.CASH): 0.05,
    (AssetClass.MARKET_NEUTRAL, AssetClass.TREND): 0.10,
    (AssetClass.MARKET_NEUTRAL, AssetClass.CASH): 0.10,
    (AssetClass.TREND, AssetClass.CASH): 0.05,
}


def correlation_matrix() -> pd.DataFrame:
    classes = list(AssetClass)
    n = len(classes)
    C = np.eye(n)
    idx = {c: i for i, c in enumerate(classes)}
    for (a, b), rho in _CORR_SPEC.items():
        C[idx[a], idx[b]] = rho
        C[idx[b], idx[a]] = rho
    C = _nearest_psd(C)
    labels = [c.value for c in classes]
    return pd.DataFrame(C, index=labels, columns=labels)


def _nearest_psd(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clip negative eigenvalues and renormalize to unit diagonal.

    Hand-specified correlation matrices are almost never PSD; without this the
    optimizer silently produces garbage weights.
    """
    C = (C + C.T) / 2
    vals, vecs = np.linalg.eigh(C)
    if vals.min() >= eps:
        return C
    vals = np.clip(vals, eps, None)
    C = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def covariance_matrix(cma: dict[AssetClass, AssetAssumption] | None = None) -> pd.DataFrame:
    cma = cma or build_cma()
    corr = correlation_matrix()
    vols = pd.Series({c.value: a.volatility for c, a in cma.items()}).reindex(corr.index)
    return corr.mul(vols, axis=0).mul(vols, axis=1)


def expected_returns(cma: dict[AssetClass, AssetAssumption] | None = None) -> pd.Series:
    cma = cma or build_cma()
    return pd.Series({c.value: a.expected_return for c, a in cma.items()})


def after_tax_return(a: AssetAssumption, tax, wrapper_is_taxable: bool) -> float:
    """Expected return net of annual tax drag inside a given wrapper.

    Drag comes from two places: income distributed each year, and gains
    realized by portfolio turnover. Unrealized appreciation compounds tax-free
    until sale — which is precisely why asset location and low turnover pay.
    """
    if not wrapper_is_taxable:
        return a.expected_return

    income_rate = tax.ordinary_rate if a.income_is_ordinary else tax.ltcg_rate
    income_drag = a.yield_component * income_rate

    appreciation = max(a.expected_return - a.yield_component, 0.0)
    realized_frac = min(a.turnover, 1.0)
    # Turnover above 100%/yr realizes short-term gains.
    gain_rate = tax.stcg_rate if a.turnover > 1.0 else tax.ltcg_rate
    gain_drag = appreciation * realized_frac * gain_rate

    return a.expected_return - income_drag - gain_drag


def tax_drag(a: AssetAssumption, tax) -> float:
    """Annual return lost to taxes if this asset sits in a taxable account.
    This number ranks candidates for asset location."""
    return a.expected_return - after_tax_return(a, tax, wrapper_is_taxable=True)


def cma_table(cma: dict[AssetClass, AssetAssumption] | None = None, tax=None) -> pd.DataFrame:
    """Audit-friendly view of the whole assumption set."""
    from .types import TaxProfile

    cma = cma or build_cma()
    tax = tax or TaxProfile()
    rows = []
    for c, a in cma.items():
        d = asdict(a)
        d["asset_class"] = c.value
        d["sharpe"] = round(a.sharpe, 3)
        d["tax_drag"] = round(tax_drag(a, tax), 4)
        d["after_tax_return"] = round(after_tax_return(a, tax, True), 4)
        rows.append(d)
    return pd.DataFrame(rows).set_index("asset_class")
