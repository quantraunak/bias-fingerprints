"""The advice engine.

Runs every analysis module against a household and returns a single ranked list
of recommendations, each carrying a quantified dollar impact and a confidence
level. Ranking is by expected present value, NOT by how interesting the idea is
— which is why tax and structure recommendations usually outrank manager
selection.

Confidence tiers:
  high     mechanical — the benefit follows from the tax code and arithmetic
  medium   depends on market behaviour but is structurally robust
  low      depends on forecasts or a manager's skill persisting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .allocation.black_litterman import View
from .allocation.policy import PolicyResult, gap_analysis, optimize_policy
from .alpha.sleeve import sleeve_report
from .cma import build_cma, cma_table
from .planning.montecarlo import ProjectionConfig, compare_policies, project
from .tax.concentration import ConcentrationInputs, compare, diversification_breakeven
from .tax.estate import estate_report
from .tax.harvest import HarvestConfig, harvest_capacity_now, simulate_harvesting
from .tax.location import optimize_location, withdrawal_sequence
from .types import Household

Confidence = Literal["high", "medium", "low"]

# How much of a claimed benefit survives its own uncertainty. "High" is
# arithmetic — the tax code and a fee schedule. "Low" depends on a manager or a
# premium persisting, and history says most of that does not show up.
CONFIDENCE_WEIGHT: dict[str, float] = {"high": 1.0, "medium": 0.65, "low": 0.35}
Category = Literal["structure", "tax", "allocation", "risk", "alpha", "cost", "liquidity"]


@dataclass
class Recommendation:
    title: str
    category: Category
    annual_value: float  # expected $/yr benefit
    pv_value: float  # present value over the planning horizon
    confidence: Confidence
    horizon: str
    rationale: str
    actions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    @property
    def weighted_pv(self) -> float:
        """Present value discounted for how much the estimate can be trusted."""
        return self.pv_value * CONFIDENCE_WEIGHT.get(self.confidence, 0.5)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "category": self.category,
            "annual_value": round(self.annual_value, 0),
            "pv_value": round(self.pv_value, 0),
            "weighted_pv": round(self.weighted_pv, 0),
            "confidence": self.confidence,
            "horizon": self.horizon,
            "rationale": self.rationale,
            "actions": self.actions,
            "risks": self.risks,
            "evidence": self.evidence,
        }


def _pv(annual: float, years: int, rate: float = 0.06) -> float:
    if rate <= 0:
        return annual * years
    return annual * (1 - (1 + rate) ** -years) / rate


class AdviceEngine:
    """Runs the full analysis stack for one household."""

    def __init__(
        self,
        household: Household,
        views: list[View] | None = None,
        cma: dict | None = None,
    ):
        self.h = household
        self.views = views or []
        self.cma = cma or build_cma()
        self._policy: PolicyResult | None = None

    # -- individual analyses ------------------------------------------------

    @property
    def policy(self) -> PolicyResult:
        if self._policy is None:
            self._policy = optimize_policy(self.h, views=self.views, cma=self.cma)
        return self._policy

    def diagnostics(self) -> dict:
        """What is wrong with the portfolio as it stands today."""
        h = self.h
        conc = h.concentration()
        top = conc.iloc[0] if len(conc) else 0.0
        top_name = conc.index[0] if len(conc) else None
        alloc = h.current_allocation()
        illiquid = sum(x.market_value for x in h.holdings if x.illiquid) / h.total_assets

        flags = []
        if top > 0.20:
            flags.append(
                f"Single-name concentration: {top_name} is {top:.0%} of net worth. "
                "Above 20% the portfolio's outcome is driven by one company's fate."
            )
        if h.spending_rate() > 0.045:
            flags.append(
                f"Spending rate {h.spending_rate():.1%} exceeds the sustainable "
                "range for a perpetual portfolio (3.0-4.0% after tax and fees)."
            )
        if illiquid > 0.45:
            flags.append(f"Illiquid assets {illiquid:.0%} of the balance sheet — capital-call risk.")
        cash_w = float(alloc.get("cash", 0.0))
        if cash_w > 0.10:
            flags.append(
                f"Cash drag: {cash_w:.0%} in cash. Every 5% of idle cash costs roughly "
                f"{(self.policy.expected_return - self.cma[list(self.cma)[-1]].expected_return) * 0.05:.2%} "
                "of portfolio return per year."
            )
        if h.unrealized_gain / h.total_assets > 0.40:
            flags.append(
                f"Embedded gain is {h.unrealized_gain / h.total_assets:.0%} of assets — "
                "any repositioning must be tax-aware, not wholesale."
            )
        return {
            "total_assets": round(h.total_assets, 0),
            "liquid_assets": round(h.liquid_assets, 0),
            "unrealized_gain": round(h.unrealized_gain, 0),
            "spending_rate": round(h.spending_rate(), 4),
            "top_holding": top_name,
            "top_holding_pct": round(float(top), 4),
            "illiquid_pct": round(float(illiquid), 4),
            "current_allocation": {k: round(v, 4) for k, v in alloc.items()},
            "flags": flags,
        }

    def harvesting(self) -> dict:
        h = self.h
        taxable = h.taxable_assets
        # Only the liquid, publicly-traded taxable equity sleeve is harvestable.
        equity_share = float(
            self.policy.weights.reindex(["us_large", "us_small", "intl_dev", "em"]).fillna(0).sum()
        )
        harvestable_base = taxable * equity_share
        res = simulate_harvesting(
            harvestable_base,
            h.tax,
            HarvestConfig(years=min(h.horizon_years, 15), n_paths=120, n_stocks=250),
        )
        immediate = harvest_capacity_now(h)
        return {
            "harvestable_base": round(harvestable_base, 0),
            "simulation": res.summary(),
            "immediate_opportunities": immediate.to_dict(orient="records") if not immediate.empty else [],
            "immediate_benefit": round(float(immediate["tax_benefit"].sum()), 0)
            if not immediate.empty
            else 0.0,
        }

    def location(self) -> dict:
        try:
            res = optimize_location(self.h, self.policy.weights, self.cma)
            return res.summary()
        except (RuntimeError, ValueError) as e:
            return {"error": str(e), "annual_savings": 0.0, "pv_savings_20y": 0.0}

    def concentration(self) -> list[dict]:
        out = []
        for holding in self.h.holdings:
            pct = holding.market_value / self.h.total_assets
            # Only individual companies above 10% of net worth. A large index
            # fund position is an allocation question, not a concentration one.
            if pct < 0.10 or holding.illiquid or not holding.single_name:
                continue
            inp = ConcentrationInputs(
                position_value=holding.market_value,
                cost_basis=holding.cost_basis,
                other_assets=self.h.total_assets - holding.market_value,
                stock_vol=holding.vol or 0.42,
                stock_beta=holding.beta or 1.15,
                diversified_return=self.policy.expected_return,
                diversified_vol=self.policy.volatility,
                horizon_years=min(self.h.horizon_years, 10),
                qsbs_eligible=any(l.qsbs_eligible for l in holding.lots),
            )
            table = compare(inp, self.h.tax)
            out.append(
                {
                    "ticker": holding.ticker,
                    "position_value": round(holding.market_value, 0),
                    "pct_of_net_worth": round(pct, 4),
                    "embedded_gain": round(holding.unrealized_gain, 0),
                    "breakeven": diversification_breakeven(inp, self.h.tax),
                    "strategies": table.to_dict(orient="records"),
                    "recommended": table.iloc[0]["strategy"],
                    "ce_gain_vs_hold": round(float(table.iloc[0]["ce_vs_hold"]), 0),
                }
            )
        return out

    def estate(self) -> dict:
        return estate_report(self.h, growth_rate=self.policy.expected_return)

    def alpha(self) -> dict:
        return sleeve_report(
            self.policy.expected_return, self.policy.volatility, self.h.tax
        )

    def projection(self) -> dict:
        cfg = ProjectionConfig(years=self.h.horizon_years, n_paths=4000)
        base = project(self.h, self.policy.expected_return, self.policy.volatility, cfg)

        current = self.h.current_allocation()
        cur_ret = float(
            sum(
                current.get(c.value, 0.0) * a.expected_return
                for c, a in self.cma.items()
            )
        )
        from .cma import covariance_matrix

        cov = covariance_matrix(self.cma)
        w = current.reindex(cov.index).fillna(0.0)
        cur_vol = float(np.sqrt(w.values @ cov.values @ w.values))

        comparison = compare_policies(
            self.h,
            {
                "Current portfolio": (cur_ret, cur_vol),
                "Policy portfolio": (self.policy.expected_return, self.policy.volatility),
                "Policy + tax alpha": (
                    self.policy.expected_return + 0.010,
                    self.policy.volatility,
                ),
            },
            cfg,
        )
        return {
            "policy": base.summary(),
            "current_return": round(cur_ret, 4),
            "current_vol": round(cur_vol, 4),
            "comparison": comparison.to_dict(orient="records"),
        }

    # -- ranked recommendations ---------------------------------------------

    def recommendations(self) -> list[Recommendation]:
        h = self.h
        total = h.total_assets
        Y = h.horizon_years
        recs: list[Recommendation] = []

        # 1. Concentration ---------------------------------------------------
        for c in self.concentration():
            if c["breakeven"]["required_alpha_to_justify_holding"] <= 0.02:
                continue
            best = c["strategies"][0]
            annual = c["ce_gain_vs_hold"] / max(Y, 1)
            recs.append(
                Recommendation(
                    title=f"Diversify the {c['ticker']} position ({c['pct_of_net_worth']:.0%} of net worth)",
                    category="risk",
                    annual_value=annual,
                    pv_value=c["ce_gain_vs_hold"],
                    confidence="high",
                    horizon="0-24 months",
                    rationale=(
                        f"{c['ticker']} would need to beat a diversified portfolio by "
                        f"{c['breakeven']['required_alpha_to_justify_holding']:.1%} per year, every year, "
                        f"to justify holding it — that is the hurdle created by "
                        f"{c['breakeven']['tax_cost_pct_of_position']:.1%} of embedded tax cost plus the "
                        "risk penalty of single-name volatility. No defensible forecast for an "
                        f"individual stock clears it. Best-ranked path: {best['strategy']}."
                    ),
                    actions=[
                        f"Execute: {best['strategy']} — {best['notes']}",
                        "Adopt a 10b5-1 plan if the holder is an insider, to remove blackout and optics risk.",
                        "Harvest losses elsewhere in the portfolio in the same tax year to offset the gain.",
                        "Set a hard concentration ceiling in the IPS (recommend 10%) with automatic trimming.",
                    ],
                    risks=[
                        "Selling into a decline crystallizes tax on a position that may recover.",
                        "Exchange funds and collars carry lock-ups, counterparty and §1259 constructive-sale risk.",
                    ],
                    evidence=c,
                )
            )

        # 2. Estate ----------------------------------------------------------
        est = self.estate()
        if est["exposure"]["total_estate_tax"] > 0:
            # The estate saving lands in YEAR Y, not today. Every other
            # recommendation here is a present value, so this has to be
            # discounted before it can sit in the same ranking — otherwise a
            # future-dollar number swamps the list and the total exceeds the
            # family's entire net worth.
            future_savings = est["total_potential_savings"]
            savings = future_savings / (1.06**Y)
            recs.append(
                Recommendation(
                    title="Move assets out of the estate before they appreciate further",
                    category="structure",
                    annual_value=savings / max(Y, 1),
                    pv_value=savings,
                    confidence="high",
                    horizon="0-12 months",
                    rationale=(
                        f"The estate is projected to reach ${est['exposure']['projected_estate']:,.0f} "
                        f"in {Y} years, generating ${est['exposure']['total_estate_tax']:,.0f} of transfer "
                        f"tax at a {est['exposure']['effective_rate_on_estate']:.0%} effective rate. Every "
                        "year of delay leaves more appreciation inside the taxable estate — this is the "
                        "single largest number on the balance sheet and it is entirely structural."
                    ),
                    actions=[s["strategy"] for s in est["strategies"][:3]]
                    + ["Engage estate counsel and a qualified appraiser before year end."],
                    risks=[
                        "Gifted assets carry over basis — heirs lose the step-up on transferred appreciation.",
                        "Irrevocable structures cannot be undone; size them so the family retains sufficient access.",
                        "Legislative risk cuts both ways; the exemption is a use-it-or-lose-it asset.",
                    ],
                    evidence=est,
                )
            )

        # 3. Tax-loss harvesting / direct indexing ---------------------------
        harv = self.harvesting()
        annual_harvest = harv["simulation"]["net_of_fee_alpha"] * harv["harvestable_base"]
        if annual_harvest > 0:
            recs.append(
                Recommendation(
                    title="Convert the public equity sleeve to direct indexing with systematic loss harvesting",
                    category="tax",
                    annual_value=annual_harvest,
                    pv_value=_pv(annual_harvest, Y),
                    confidence="high",
                    horizon="0-6 months",
                    rationale=(
                        f"Lot-level simulation produces {harv['simulation']['first_year_tax_alpha']:.2%} of "
                        f"tax alpha in year one and {harv['simulation']['avg_annual_tax_alpha']:.2%} on average, "
                        f"net {harv['simulation']['net_of_fee_alpha']:.2%} after the SMA fee differential. "
                        "The benefit does not depend on forecasting anything — only on dispersion existing, "
                        "which it always does. It decays as the portfolio appreciates, so starting earlier is "
                        "strictly better."
                    ),
                    actions=[
                        "Move the taxable index exposure into a direct-indexing SMA (fee target ≤25bps).",
                        "Transfer existing low-basis ETFs in kind where possible; do not liquidate to fund it.",
                        "Set a 5% loss threshold with a 31-day wash-sale block and a correlated replacement list.",
                        "Route harvested losses against the gains realized in the concentration unwind.",
                    ],
                    risks=[
                        f"Harvesting defers rather than eliminates: terminal embedded gain rises to "
                        f"{harv['simulation']['terminal_embedded_gain_pct']:.0%}, a "
                        f"${harv['simulation']['deferral_liability']:,.0f} latent liability.",
                        "Tracking error to the index of roughly 50-150bps annually.",
                        "Losses only pay if the family has gains or income to absorb them.",
                    ],
                    evidence=harv,
                )
            )

        # 4. Asset location --------------------------------------------------
        loc = self.location()
        if loc.get("annual_savings", 0) > 0:
            recs.append(
                Recommendation(
                    title="Re-locate assets across tax wrappers",
                    category="tax",
                    annual_value=loc["annual_savings"],
                    pv_value=loc["pv_savings_20y"],
                    confidence="high",
                    horizon="0-3 months",
                    rationale=(
                        f"Holding the same policy weights in different wrappers saves "
                        f"${loc['annual_savings']:,.0f}/yr ({loc['savings_bps']:.0f}bps) with zero change in "
                        "market risk. High-turnover and high-yield assets belong in deferred accounts; "
                        "low-turnover equity belongs in taxable where it can be harvested and stepped up."
                    ),
                    actions=[
                        "Move private credit, high yield and the market-neutral sleeve into deferred accounts.",
                        "Hold direct-indexed equity in taxable to preserve harvesting and step-up.",
                        "Place the highest-expected-return assets in Roth/DAF where growth is never taxed.",
                    ],
                    risks=["Requires available capacity in each wrapper; rebalance only with new flows where possible."],
                    evidence=loc,
                )
            )

        # 5. Allocation gap --------------------------------------------------
        gaps = gap_analysis(h, self.policy.weights)
        proj = self.projection()
        return_gain = self.policy.expected_return - proj["current_return"]
        vol_change = self.policy.volatility - proj["current_vol"]
        if abs(return_gain) > 0.002 or vol_change < -0.005:
            annual = max(return_gain, 0.0) * total
            recs.append(
                Recommendation(
                    title="Reposition to the policy allocation",
                    category="allocation",
                    annual_value=annual,
                    pv_value=_pv(annual, Y),
                    confidence="medium",
                    horizon="6-18 months",
                    rationale=(
                        f"The policy portfolio raises expected return from {proj['current_return']:.2%} to "
                        f"{self.policy.expected_return:.2%} while moving volatility from "
                        f"{proj['current_vol']:.2%} to {self.policy.volatility:.2%}, inside the "
                        f"{self.policy.vol_budget:.1%} budget implied by a "
                        f"{h.max_drawdown_tolerance:.0%} drawdown tolerance. "
                        f"Binding constraints: {', '.join(self.policy.binding_constraints) or 'none'}."
                    ),
                    actions=[
                        "Direct all new cash flow to underweight sleeves before selling anything.",
                        "Close gaps with the lowest tax cost first — see the gap table.",
                        "Phase private-markets commitments across vintages rather than in a single year.",
                    ],
                    risks=[
                        "Expected returns are forecasts; the allocation should be robust to being wrong.",
                        "Repositioning triggers realized gains — sequence it with harvested losses.",
                    ],
                    evidence={
                        "gaps": gaps.reset_index().to_dict(orient="records"),
                        "projection": proj,
                    },
                )
            )

        # 6. Private markets -------------------------------------------------
        priv_target = float(
            self.policy.weights.reindex(["private_equity", "private_credit", "real_estate"])
            .fillna(0)
            .sum()
        )
        priv_current = float(
            h.current_allocation()
            .reindex(["private_equity", "private_credit", "real_estate"])
            .fillna(0)
            .sum()
        )
        if priv_target - priv_current > 0.03:
            illiq_premium = 0.025
            annual = (priv_target - priv_current) * total * illiq_premium
            recs.append(
                Recommendation(
                    title="Build the private-markets allocation on a vintage-year schedule",
                    category="allocation",
                    annual_value=annual,
                    pv_value=_pv(annual, Y),
                    confidence="low",
                    horizon="3-5 years",
                    rationale=(
                        f"Policy calls for {priv_target:.0%} in private markets against {priv_current:.0%} today. "
                        "The illiquidity premium is real but manager dispersion in private markets is enormous — "
                        "the gap between top- and bottom-quartile PE funds exceeds 15%/yr, versus roughly 2% in "
                        "public equity. At this asset level the binding constraint is access, not conviction."
                    ),
                    actions=[
                        "Commit across 4-5 vintage years rather than one, to avoid vintage concentration.",
                        "Favour evergreen/interval structures and secondaries for faster deployment and shorter J-curves.",
                        "Model capital calls explicitly against the liquidity reserve before committing.",
                        "Insist on fee transparency — layered fund-of-fund fees erase the illiquidity premium.",
                    ],
                    risks=[
                        "Manager selection risk dominates; median private equity has not beaten public equity net of fees.",
                        "Capital calls arrive in stress, exactly when liquid assets are impaired.",
                        "Reported volatility is appraisal-smoothed and understates true risk by roughly half.",
                    ],
                    evidence={"target": priv_target, "current": priv_current},
                )
            )

        # 7. Systematic alpha sleeve ------------------------------------------
        alpha = self.alpha()
        w = alpha["sizing"]["recommended_weight"]
        if w > 0.01:
            annual = w * total * (
                alpha["live_expectation"]["expected_return"] - self.cma[list(self.cma)[-1]].expected_return
            )
            recs.append(
                Recommendation(
                    title=f"Allocate {w:.0%} to the systematic market-neutral sleeve",
                    category="alpha",
                    annual_value=max(annual, 0.0),
                    pv_value=_pv(max(annual, 0.0), Y),
                    confidence="low",
                    horizon="12-24 months",
                    rationale=(
                        f"The sleeve's backtest Sharpe of {alpha['backtest']['sharpe']} deflates to "
                        f"{alpha['live_expectation']['sharpe']} after taking the lower bound of the bootstrap "
                        "confidence interval, an overfitting factor, capacity and fees. Even at that level its "
                        f"near-zero equity correlation improves portfolio Sharpe by "
                        f"{alpha['sizing']['sharpe_improvement']}. Value it as a diversifier, not a return engine."
                    ),
                    actions=[
                        "Hold exclusively in a tax-deferred wrapper — 120% monthly turnover is taxed at short-term rates.",
                        "Require 24 months of live, audited track record before sizing above 5%.",
                        "Cap at 10-15%: this is single-model, single-researcher risk.",
                    ],
                    risks=alpha["risks"],
                    evidence=alpha,
                )
            )

        # 8. Cost --------------------------------------------------------------
        assumed_current_fee = 0.0110  # typical all-in for this segment
        target_fee = 0.0055
        fee_saving = (assumed_current_fee - target_fee) * total
        recs.append(
            Recommendation(
                title="Compress the all-in fee load",
                category="cost",
                annual_value=fee_saving,
                pv_value=_pv(fee_saving, Y),
                confidence="high",
                horizon="0-6 months",
                rationale=(
                    f"A typical $10-50M household pays roughly {assumed_current_fee:.2%} all-in once advisory, "
                    f"fund, platform and trading costs are stacked. Getting to {target_fee:.2%} is achievable "
                    f"without changing the asset mix and is worth ${fee_saving:,.0f}/yr. Fees are the only "
                    "portfolio input known with certainty in advance."
                ),
                actions=[
                    "Negotiate advisory to a flat retainer or a fee that breakpoints down above $25M.",
                    "Replace active public equity funds with direct indexing or low-cost ETFs.",
                    "Eliminate fund-of-fund layers in alternatives; go direct or via co-invest.",
                    "Audit securities-lending revenue, cash sweep rates and FX spreads at the custodian.",
                ],
                risks=["Cheapest is not always best — do not trade away genuine access in alternatives to save 20bps."],
                evidence={"assumed_current_fee": assumed_current_fee, "target_fee": target_fee},
            )
        )

        # 9. Liquidity ---------------------------------------------------------
        net_spend = max(h.annual_spending - h.annual_income, 0.0)
        reserve_needed = net_spend * h.liquidity_reserve_years + h.unfunded_commitments
        if h.liquid_assets < reserve_needed:
            recs.append(
                Recommendation(
                    title="Rebuild the liquidity reserve",
                    category="liquidity",
                    annual_value=0.0,
                    pv_value=0.0,
                    confidence="high",
                    horizon="0-3 months",
                    rationale=(
                        f"Liquid assets of ${h.liquid_assets:,.0f} fall short of the "
                        f"${reserve_needed:,.0f} needed to cover {h.liquidity_reserve_years:.0f} years of "
                        "spending plus unfunded commitments. Forced selling in a drawdown is the most "
                        "expensive mistake available to this balance sheet."
                    ),
                    actions=[
                        "Hold 2 years of spending in T-bills laddered to spending dates.",
                        "Establish a securities-based line of credit BEFORE it is needed — lenders withdraw in stress.",
                        "Model capital calls at 1.5x the manager's stated pace.",
                    ],
                    risks=["Cash drag if the reserve is oversized; ladder rather than sit in a sweep account."],
                    evidence={"liquid": h.liquid_assets, "required": reserve_needed},
                )
            )

        # Rank by CONFIDENCE-WEIGHTED value. Ranking on raw present value lets
        # a backtested alpha sleeve outrank a tax action that is pure
        # arithmetic, which is exactly how families get sold the wrong thing.
        return sorted(recs, key=lambda r: r.weighted_pv, reverse=True)

    # -- full report --------------------------------------------------------

    def full_report(self) -> dict:
        recs = self.recommendations()
        proj = self.projection()
        total_pv = sum(max(r.pv_value, 0) for r in recs)
        return {
            "household": {
                "name": self.h.name,
                "total_assets": round(self.h.total_assets, 0),
                "horizon_years": self.h.horizon_years,
                "spending_rate": round(self.h.spending_rate(), 4),
                "max_drawdown_tolerance": self.h.max_drawdown_tolerance,
            },
            "diagnostics": self.diagnostics(),
            "policy": self.policy.summary(),
            "gap_analysis": gap_analysis(self.h, self.policy.weights).reset_index().to_dict(orient="records"),
            "projection": proj,
            "recommendations": [r.as_dict() for r in recs],
            "total_pv_of_recommendations": round(total_pv, 0),
            "pv_as_pct_of_assets": round(total_pv / self.h.total_assets, 4)
            if self.h.total_assets
            else 0.0,
            "cma": cma_table(self.cma, self.h.tax).reset_index().to_dict(orient="records"),
            "withdrawal_plan": withdrawal_sequence(
                self.h, max(self.h.annual_spending - self.h.annual_income, 0.0), years=15
            ).to_dict(orient="records"),
            "disclosures": DISCLOSURES,
        }


DISCLOSURES = [
    "This analysis is generated by a quantitative model for informational and "
    "educational purposes. It is not personalized investment, tax, or legal advice, "
    "and no fiduciary relationship is created by its use.",
    "Expected returns are forward-looking estimates built from current market inputs. "
    "They are not guarantees; realized returns will differ, potentially materially.",
    "Tax analysis uses assumed marginal rates and simplified rules. Federal and state "
    "tax law is fact-specific and changes frequently. Confirm every tax or estate "
    "strategy with qualified counsel and a CPA before acting.",
    "Backtested and simulated performance has inherent limitations: it is derived with "
    "hindsight, does not reflect live trading, and does not guarantee future results.",
    "Distributing personalized investment recommendations for compensation generally "
    "requires registration as an investment adviser. Confirm your regulatory status "
    "with securities counsel before delivering this output to clients.",
]
