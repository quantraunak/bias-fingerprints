"""Core domain types for a single-family-office household.

Everything downstream (allocation, tax, planning) reads these. Money is in
nominal USD, rates are decimals (0.0325 == 3.25%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import pandas as pd


class AssetClass(str, Enum):
    """The SAA opportunity set. Keep this list stable — CMA and covariance
    matrices are keyed off it."""

    US_LARGE = "us_large"
    US_SMALL = "us_small"
    INTL_DEV = "intl_dev"
    EM = "em"
    CORE_BOND = "core_bond"
    TIPS = "tips"
    HY_CREDIT = "hy_credit"
    PRIVATE_EQUITY = "private_equity"
    PRIVATE_CREDIT = "private_credit"
    REAL_ESTATE = "real_estate"
    MARKET_NEUTRAL = "market_neutral"
    TREND = "trend"
    CASH = "cash"


class AccountType(str, Enum):
    """Tax wrapper. Drives asset location and after-tax return math."""

    TAXABLE = "taxable"
    TRADITIONAL_IRA = "traditional_ira"
    ROTH = "roth"
    GRANTOR_TRUST = "grantor_trust"  # IDGT/SLAT — income taxed to grantor
    NON_GRANTOR_TRUST = "non_grantor_trust"  # compressed brackets
    DAF = "daf"  # donor-advised fund: tax-exempt
    CRT = "crt"  # charitable remainder trust: tax-exempt inside


TAX_DEFERRED = {AccountType.TRADITIONAL_IRA}
TAX_EXEMPT = {AccountType.ROTH, AccountType.DAF, AccountType.CRT}
TAXABLE_WRAPPERS = {
    AccountType.TAXABLE,
    AccountType.GRANTOR_TRUST,
    AccountType.NON_GRANTOR_TRUST,
}


@dataclass
class Lot:
    """A single tax lot. Lot-level detail is what makes tax alpha real —
    average cost basis systematically understates harvestable losses."""

    ticker: str
    shares: float
    cost_basis_per_share: float
    acquired: pd.Timestamp
    qsbs_eligible: bool = False  # IRC §1202 qualified small business stock

    @property
    def cost_basis(self) -> float:
        return self.shares * self.cost_basis_per_share

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized(self, price: float) -> float:
        return self.market_value(price) - self.cost_basis

    def is_long_term(self, asof: pd.Timestamp) -> bool:
        return (asof - self.acquired).days > 365


@dataclass
class Holding:
    """A position, optionally decomposed into lots."""

    ticker: str
    asset_class: AssetClass
    market_value: float
    price: float = 100.0
    lots: list[Lot] = field(default_factory=list)
    illiquid: bool = False
    # True only for an individual company. A 30% position in a total-market ETF
    # is not concentration risk; a 30% position in one stock is. Conflating the
    # two is the most common error in automated portfolio review.
    single_name: bool = False
    # For concentrated single stock: annualized vol and beta drive hedge pricing.
    vol: float | None = None
    beta: float | None = None

    @property
    def cost_basis(self) -> float:
        if self.lots:
            return sum(l.cost_basis for l in self.lots)
        return self.market_value  # unknown basis -> assume no embedded gain

    @property
    def unrealized_gain(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def embedded_gain_pct(self) -> float:
        if self.market_value <= 0:
            return 0.0
        return self.unrealized_gain / self.market_value


@dataclass
class Account:
    account_id: str
    account_type: AccountType
    holdings: list[Holding] = field(default_factory=list)
    # Securities-based line of credit available against this account.
    margin_eligible: bool = False

    @property
    def market_value(self) -> float:
        return sum(h.market_value for h in self.holdings)

    @property
    def is_taxable(self) -> bool:
        return self.account_type in TAXABLE_WRAPPERS


@dataclass
class TaxProfile:
    """Marginal rates actually paid. Defaults are 2026 federal top brackets
    for MFJ plus a high-tax state; override per household."""

    federal_ordinary: float = 0.37
    federal_ltcg: float = 0.20
    niit: float = 0.038  # net investment income tax
    state_ordinary: float = 0.133  # e.g. CA top marginal
    state_ltcg: float = 0.133  # most states tax cap gains as ordinary
    state_deductible: bool = False  # SALT cap makes this False for most
    estate_tax_rate: float = 0.40
    estate_exemption_per_person: float = 15_000_000.0  # 2026 OBBBA level
    filing_status: Literal["mfj", "single"] = "mfj"

    @property
    def ordinary_rate(self) -> float:
        """All-in marginal rate on interest / non-qualified income."""
        return self.federal_ordinary + self.niit + self.state_ordinary

    @property
    def ltcg_rate(self) -> float:
        """All-in marginal rate on long-term gains and qualified dividends."""
        return self.federal_ltcg + self.niit + self.state_ltcg

    @property
    def stcg_rate(self) -> float:
        return self.ordinary_rate

    @property
    def harvest_credit_rate(self) -> float:
        """Value of $1 of harvested loss. Losses offset STCG first (highest
        rate); a blended rate is the honest assumption for a mixed portfolio."""
        return 0.5 * self.stcg_rate + 0.5 * self.ltcg_rate


@dataclass
class Goal:
    """A funding requirement with a date and a priority tier."""

    name: str
    amount: float  # today's dollars
    year_offset: int  # years from now
    priority: Literal["essential", "important", "aspirational"] = "important"
    inflation_linked: bool = True


@dataclass
class Household:
    name: str
    accounts: list[Account] = field(default_factory=list)
    tax: TaxProfile = field(default_factory=TaxProfile)
    goals: list[Goal] = field(default_factory=list)

    # Spending and liquidity
    annual_spending: float = 400_000.0
    annual_income: float = 0.0  # outside earned income / business distributions
    liquidity_reserve_years: float = 2.0

    # Risk posture
    max_drawdown_tolerance: float = 0.25  # peak-to-trough the family can hold through
    horizon_years: int = 30
    n_beneficiaries: int = 2

    # Unfunded private-markets commitments (capital calls still to come)
    unfunded_commitments: float = 0.0

    @property
    def total_assets(self) -> float:
        return sum(a.market_value for a in self.accounts)

    @property
    def taxable_assets(self) -> float:
        return sum(a.market_value for a in self.accounts if a.is_taxable)

    @property
    def liquid_assets(self) -> float:
        return sum(
            h.market_value
            for a in self.accounts
            for h in a.holdings
            if not h.illiquid
        )

    @property
    def holdings(self) -> list[Holding]:
        return [h for a in self.accounts for h in a.holdings]

    @property
    def unrealized_gain(self) -> float:
        return sum(h.unrealized_gain for h in self.holdings)

    def current_allocation(self) -> pd.Series:
        """Weights by asset class across the whole balance sheet."""
        totals: dict[str, float] = {}
        for h in self.holdings:
            totals[h.asset_class.value] = totals.get(h.asset_class.value, 0.0) + h.market_value
        s = pd.Series(totals, dtype=float)
        total = s.sum()
        return s / total if total > 0 else s

    def concentration(self) -> pd.Series:
        """Weights of INDIVIDUAL COMPANIES, largest first, as a share of net
        worth. The #1 risk in this cohort. Funds and cash are excluded — they
        are not single-name exposure."""
        totals: dict[str, float] = {}
        for h in self.holdings:
            if not h.single_name:
                continue
            totals[h.ticker] = totals.get(h.ticker, 0.0) + h.market_value
        if not totals:
            return pd.Series(dtype=float)
        s = pd.Series(totals, dtype=float).sort_values(ascending=False)
        return s / self.total_assets if self.total_assets > 0 else s

    def spending_rate(self) -> float:
        net_spend = max(self.annual_spending - self.annual_income, 0.0)
        return net_spend / self.total_assets if self.total_assets > 0 else 0.0
