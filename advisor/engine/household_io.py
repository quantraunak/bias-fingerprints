"""Serialization for households — JSON in, dataclasses out."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .types import (
    Account,
    AccountType,
    AssetClass,
    Goal,
    Holding,
    Household,
    Lot,
    TaxProfile,
)


def household_from_dict(d: dict) -> Household:
    accounts = []
    for a in d.get("accounts", []):
        holdings = []
        for h in a.get("holdings", []):
            lots = [
                Lot(
                    ticker=h["ticker"],
                    shares=l["shares"],
                    cost_basis_per_share=l["cost_basis_per_share"],
                    acquired=pd.Timestamp(l["acquired"]),
                    qsbs_eligible=l.get("qsbs_eligible", False),
                )
                for l in h.get("lots", [])
            ]
            holdings.append(
                Holding(
                    ticker=h["ticker"],
                    asset_class=AssetClass(h["asset_class"]),
                    market_value=float(h["market_value"]),
                    price=float(h.get("price", 100.0)),
                    lots=lots,
                    illiquid=bool(h.get("illiquid", False)),
                    single_name=bool(h.get("single_name", False)),
                    vol=h.get("vol"),
                    beta=h.get("beta"),
                )
            )
        accounts.append(
            Account(
                account_id=a["account_id"],
                account_type=AccountType(a["account_type"]),
                holdings=holdings,
                margin_eligible=bool(a.get("margin_eligible", False)),
            )
        )

    tax = TaxProfile(**d.get("tax", {}))
    goals = [Goal(**g) for g in d.get("goals", [])]

    return Household(
        name=d.get("name", "Household"),
        accounts=accounts,
        tax=tax,
        goals=goals,
        annual_spending=float(d.get("annual_spending", 400_000)),
        annual_income=float(d.get("annual_income", 0)),
        liquidity_reserve_years=float(d.get("liquidity_reserve_years", 2.0)),
        max_drawdown_tolerance=float(d.get("max_drawdown_tolerance", 0.25)),
        horizon_years=int(d.get("horizon_years", 30)),
        n_beneficiaries=int(d.get("n_beneficiaries", 2)),
        unfunded_commitments=float(d.get("unfunded_commitments", 0.0)),
    )


def household_to_dict(h: Household) -> dict:
    return {
        "name": h.name,
        "annual_spending": h.annual_spending,
        "annual_income": h.annual_income,
        "liquidity_reserve_years": h.liquidity_reserve_years,
        "max_drawdown_tolerance": h.max_drawdown_tolerance,
        "horizon_years": h.horizon_years,
        "n_beneficiaries": h.n_beneficiaries,
        "unfunded_commitments": h.unfunded_commitments,
        "tax": h.tax.__dict__,
        "goals": [g.__dict__ for g in h.goals],
        "accounts": [
            {
                "account_id": a.account_id,
                "account_type": a.account_type.value,
                "margin_eligible": a.margin_eligible,
                "holdings": [
                    {
                        "ticker": x.ticker,
                        "asset_class": x.asset_class.value,
                        "market_value": x.market_value,
                        "price": x.price,
                        "illiquid": x.illiquid,
                        "single_name": x.single_name,
                        "vol": x.vol,
                        "beta": x.beta,
                        "lots": [
                            {
                                "shares": l.shares,
                                "cost_basis_per_share": l.cost_basis_per_share,
                                "acquired": l.acquired.date().isoformat(),
                                "qsbs_eligible": l.qsbs_eligible,
                            }
                            for l in x.lots
                        ],
                    }
                    for x in a.holdings
                ],
            }
            for a in h.accounts
        ],
    }


def load_household(path: str | Path) -> Household:
    return household_from_dict(json.loads(Path(path).read_text()))


def save_household(h: Household, path: str | Path) -> None:
    Path(path).write_text(json.dumps(household_to_dict(h), indent=2))


def sample_household() -> Household:
    """A representative $28M family office.

    Deliberately built with the pathologies this cohort actually has: a large
    low-basis founder position, too much cash, no private markets, no
    tax-aware structure, and an estate that will breach the exemption.
    """
    concentrated = Holding(
        ticker="NVDA",
        asset_class=AssetClass.US_LARGE,
        market_value=8_400_000,
        price=180.0,
        single_name=True,
        vol=0.44,
        beta=1.62,
        lots=[
            Lot("NVDA", 30_000, 14.0, pd.Timestamp("2017-03-15")),
            Lot("NVDA", 12_000, 48.0, pd.Timestamp("2020-06-01")),
            Lot("NVDA", 4_666, 132.0, pd.Timestamp("2024-02-20")),
        ],
    )

    taxable = Account(
        "taxable-brokerage",
        AccountType.TAXABLE,
        margin_eligible=True,
        holdings=[
            concentrated,
            Holding(
                "VTI", AssetClass.US_LARGE, 5_200_000, 285.0,
                lots=[
                    Lot("VTI", 12_000, 190.0, pd.Timestamp("2019-08-01")),
                    Lot("VTI", 6_245, 305.0, pd.Timestamp("2025-01-15")),
                ],
            ),
            Holding(
                "VXUS", AssetClass.INTL_DEV, 1_600_000, 62.0,
                lots=[Lot("VXUS", 25_806, 68.0, pd.Timestamp("2024-07-10"))],
            ),
            Holding(
                "BND", AssetClass.CORE_BOND, 2_100_000, 73.0,
                lots=[Lot("BND", 28_767, 79.0, pd.Timestamp("2021-11-01"))],
            ),
            Holding("CASH", AssetClass.CASH, 3_300_000, 1.0,
                    lots=[Lot("CASH", 3_300_000, 1.0, pd.Timestamp("2025-06-01"))]),
        ],
    )

    ira = Account(
        "rollover-ira",
        AccountType.TRADITIONAL_IRA,
        holdings=[
            Holding("AGG", AssetClass.CORE_BOND, 1_900_000, 98.0),
            Holding("VTI", AssetClass.US_LARGE, 1_400_000, 285.0),
        ],
    )

    roth = Account(
        "roth-ira",
        AccountType.ROTH,
        holdings=[Holding("VTI", AssetClass.US_LARGE, 900_000, 285.0)],
    )

    slat = Account(
        "family-slat",
        AccountType.GRANTOR_TRUST,
        holdings=[
            Holding("VTI", AssetClass.US_LARGE, 1_800_000, 285.0,
                    lots=[Lot("VTI", 6_315, 210.0, pd.Timestamp("2022-04-01"))]),
            Holding(
                "PE-FUND-II", AssetClass.PRIVATE_EQUITY, 1_400_000, 1.0, illiquid=True,
                lots=[Lot("PE-FUND-II", 1_400_000, 1.0, pd.Timestamp("2022-09-01"))],
            ),
        ],
    )

    return Household(
        name="Sample Family Office",
        accounts=[taxable, ira, roth, slat],
        tax=TaxProfile(),  # CA resident, top bracket
        goals=[
            Goal("Second home purchase", 3_500_000, 3, "important"),
            Goal("Children's education", 1_200_000, 8, "essential"),
            Goal("Foundation seed funding", 5_000_000, 15, "aspirational"),
        ],
        annual_spending=750_000,
        annual_income=250_000,
        liquidity_reserve_years=2.0,
        max_drawdown_tolerance=0.28,
        horizon_years=30,
        n_beneficiaries=3,
        unfunded_commitments=1_100_000,
    )
