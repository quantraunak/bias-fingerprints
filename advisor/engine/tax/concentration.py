"""The concentrated position decision.

Most $10-50M families got there from one asset — founder stock, RSUs, a
building. That position is simultaneously the source of the wealth and the
largest threat to it. The instinct to "wait for a better tax year" is usually
wrong: the risk cost of holding a single volatile name typically exceeds the
tax cost of diversifying.

This module prices every realistic path on a common footing — certainty-
equivalent after-tax wealth — so the choice can be made on numbers rather than
sentiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..types import Holding, TaxProfile


@dataclass
class ConcentrationInputs:
    position_value: float
    cost_basis: float
    # The rest of the household balance sheet. Every strategy is scored on
    # TOTAL wealth, not on the position in isolation — a 30% position and a
    # 100% position are completely different decisions, and a model that
    # ignores the rest of the portfolio cannot tell them apart.
    other_assets: float = 0.0
    correlation_to_core: float = 0.60
    stock_vol: float = 0.42
    stock_beta: float = 1.15
    stock_expected_return: float | None = None  # None -> CAPM, no idio alpha
    market_return: float = 0.075
    cash_rate: float = 0.037
    diversified_return: float = 0.072
    diversified_vol: float = 0.115
    horizon_years: int = 10
    risk_aversion: float = 3.0  # CRRA gamma; 3 is typical for this cohort
    qsbs_eligible: bool = False
    qsbs_exclusion_cap: float = 15_000_000.0  # per issuer, post-OBBBA
    charitable_intent: float = 0.0  # dollars the family intends to give anyway
    dividend_yield: float = 0.005

    @property
    def embedded_gain(self) -> float:
        return max(self.position_value - self.cost_basis, 0.0)

    def capm_return(self) -> float:
        if self.stock_expected_return is not None:
            return self.stock_expected_return
        # No idiosyncratic alpha is assumed. Believing otherwise is the single
        # most expensive assumption a concentrated holder can make.
        return self.cash_rate + self.stock_beta * (self.market_return - self.cash_rate)


@dataclass
class Strategy:
    name: str
    after_tax_proceeds_now: float
    expected_terminal_wealth: float
    terminal_vol: float
    certainty_equivalent: float
    tax_paid_now: float
    residual_concentration: float  # $ still exposed to the single name
    liquidity: Literal["immediate", "staged", "locked", "none"]
    notes: str

    def as_dict(self) -> dict:
        return {
            "strategy": self.name,
            "tax_paid_now": round(self.tax_paid_now, 0),
            "expected_terminal_wealth": round(self.expected_terminal_wealth, 0),
            "terminal_vol": round(self.terminal_vol, 4),
            "certainty_equivalent": round(self.certainty_equivalent, 0),
            "residual_concentration": round(self.residual_concentration, 0),
            "liquidity": self.liquidity,
            "notes": self.notes,
        }


def _blend(
    stock_value: float,
    core_value: float,
    stock_ret: float,
    stock_vol: float,
    core_ret: float,
    core_vol: float,
    rho: float,
) -> tuple[float, float, float]:
    """Combine a concentrated position with the rest of the balance sheet.

    Returns (total wealth, portfolio expected return, portfolio volatility).
    """
    total = stock_value + core_value
    if total <= 0:
        return 0.0, 0.0, 0.0
    w = stock_value / total
    mu = w * stock_ret + (1 - w) * core_ret
    var = (
        (w * stock_vol) ** 2
        + ((1 - w) * core_vol) ** 2
        + 2 * w * (1 - w) * rho * stock_vol * core_vol
    )
    return total, mu, float(np.sqrt(max(var, 0.0)))


def _ce_wealth(wealth: float, mu: float, vol: float, years: int, gamma: float) -> float:
    """Certainty-equivalent terminal wealth for a CRRA family.

    The certainty-equivalent annual growth rate is mu - gamma*sigma^2/2, applied
    to the PORTFOLIO's return and volatility. Applying the penalty to the
    standalone position instead produces required-alpha numbers in the tens of
    percent, which is why so many concentration models are ignored by the
    families they are shown to.
    """
    if wealth <= 0:
        return 0.0
    g_ce = mu - gamma * vol**2 / 2
    return wealth * float((1 + g_ce) ** years) if g_ce > -1 else 0.0


def evaluate_strategies(inp: ConcentrationInputs, tax: TaxProfile) -> list[Strategy]:
    """Price the full menu of concentrated-position strategies."""
    V = inp.position_value
    gain = inp.embedded_gain
    ltcg = tax.ltcg_rate
    gamma = inp.risk_aversion
    Y = inp.horizon_years
    rho = inp.correlation_to_core
    core = inp.other_assets
    dr, dv = inp.diversified_return, inp.diversified_vol
    stock_ret = inp.capm_return()
    out: list[Strategy] = []

    def score(
        stock_value: float,
        core_value: float,
        s_ret: float,
        s_vol: float,
        c_ret: float = dr,
        c_vol: float = dv,
        corr: float | None = None,
    ) -> tuple[float, float, float]:
        """(expected terminal wealth, portfolio vol, certainty equivalent)."""
        w, mu, vol = _blend(
            stock_value, core_value, s_ret, s_vol, c_ret, c_vol,
            rho if corr is None else corr,
        )
        return w * (1 + mu) ** Y, vol, _ce_wealth(w, mu, vol, Y, gamma)

    # ---- 1. Hold ----------------------------------------------------------
    exp, vol, ce = score(V, core, stock_ret, inp.stock_vol)
    out.append(
        Strategy(
            "Hold (do nothing)",
            0.0, exp, vol, ce, 0.0, V, "none",
            f"Leaves {V / max(V + core, 1):.0%} of net worth in one company at "
            f"{inp.stock_vol:.0%} volatility, lifting whole-portfolio risk to "
            f"{vol:.1%}. Roughly 40% of individual large-cap stocks have "
            "underperformed cash over any given decade.",
        )
    )
    hold_ce = ce

    # ---- 2. Outright sale -------------------------------------------------
    tax_now = gain * ltcg
    exp, vol, ce = score(0.0, core + V - tax_now, 0.0, 0.0)
    out.append(
        Strategy(
            "Sell outright",
            V - tax_now, exp, vol, ce, tax_now, 0.0, "immediate",
            f"Pays {tax_now / V:.1%} of the position in tax today and removes "
            f"all single-name risk. Portfolio volatility falls to {vol:.1%}.",
        )
    )

    # ---- 3. Staged 10b5-1 sale over 3 years --------------------------------
    stage_years = 3
    # Averaged over the horizon the family holds roughly a third of the
    # position for the first three years, then none.
    avg_stock = V * (stage_years / 2) / max(Y, 1)
    exp, vol, ce = score(avg_stock, core + V - tax_now - avg_stock, stock_ret, inp.stock_vol)
    out.append(
        Strategy(
            "Staged sale (10b5-1, 3yr)",
            V - tax_now, exp, vol, ce, tax_now / stage_years, V / stage_years, "staged",
            "Spreads realization across tax years, keeps each year out of the "
            "highest brackets, and removes insider-trading timing risk. Same "
            "total tax, materially lower execution and optics risk.",
        )
    )

    # ---- 4. Exchange fund (7-year lock) ------------------------------------
    # Contribute stock in kind for a diversified partnership interest; no tax
    # event, basis carries over. Cost: ~85bps/yr fee, 7-year lock, and 20% of
    # the fund must sit in illiquid real estate to qualify under §721.
    if Y >= 7:
        ex_fee, ballast_drag = 0.0085, 0.003
        exp, vol, ce = score(0.0, core + V, 0.0, 0.0, c_ret=dr - ex_fee - ballast_drag)
        out.append(
            Strategy(
                "Exchange fund (§721)",
                0.0, exp, vol, ce, 0.0, 0.0, "locked",
                "Diversifies the full position with zero current tax. Costs "
                "~85bps/yr plus a 7-year lock and mandatory real-estate "
                "ballast. Basis carries over — the gain is deferred, not "
                "forgiven, so this pairs best with a step-up at death.",
            )
        )

    # ---- 5. Collar + securities-based loan ---------------------------------
    # Zero-cost collar (buy ~90% put, sell ~112% call), then borrow against the
    # hedged position to fund a diversified sleeve. No current tax if kept
    # clear of the §1259 constructive-sale rules.
    ltv, borrow_spread = 0.60, 0.012
    borrow_rate = inp.cash_rate + borrow_spread
    # A zero-cost collar sells the right tail to buy the left. On a 44%-vol
    # stock with roughly ±10% strikes the payoff is closer to a bond than to
    # equity, so the expected return collapses toward cash. Modelling a collar
    # as "same return, less risk" is the standard way these structures get
    # oversold — it is a financing tool, not a free lunch.
    collared_vol = inp.stock_vol * 0.35
    collared_ret = inp.cash_rate + 0.015
    loan = V * ltv
    # Net wealth is unchanged by borrowing — the loan is offset by the debt.
    # What changes is that market EXPOSURE now exceeds net wealth, which is
    # exactly where the extra return and the extra risk both come from.
    wealth = V + core
    w_stock = V / wealth
    w_div = (core + loan) / wealth
    w_debt = loan / wealth
    mu = w_stock * collared_ret + w_div * dr - w_debt * borrow_rate
    vol = float(
        np.sqrt(
            (w_stock * collared_vol) ** 2
            + (w_div * dv) ** 2
            + 2 * w_stock * w_div * rho * collared_vol * dv
        )
    )
    exp = wealth * (1 + mu) ** Y
    ce = _ce_wealth(wealth, mu, vol, Y, gamma)
    out.append(
        Strategy(
            "Collar + SBL diversification",
            loan, exp, vol, ce, 0.0, V, "immediate",
            f"Hedges the tail and monetizes {ltv:.0%} LTV at ~{borrow_rate:.1%} "
            "with no sale and no current tax. The collar also sells most of the "
            "upside, so the hedged position earns close to a cash-like return — "
            "useful as a BRIDGE to diversification, not a permanent answer. "
            "Collars run 1-3 years and must be rolled at prevailing vol. Watch "
            "§1259 constructive sale, §246(c) dividend rules, and margin-call "
            "risk if the stock gaps through the put strike.",
        )
    )

    # ---- 6. QSBS §1202 -----------------------------------------------------
    if inp.qsbs_eligible:
        excluded = min(gain, max(inp.qsbs_exclusion_cap, 10 * inp.cost_basis))
        taxable_gain = max(gain - excluded, 0.0)
        # §1202 excludes federal tax; most states conform, California does not.
        fed_saved = excluded * (tax.federal_ltcg + tax.niit)
        tax_qsbs = taxable_gain * ltcg + excluded * tax.state_ltcg
        exp, vol, ce = score(0.0, core + V - tax_qsbs, 0.0, 0.0)
        out.append(
            Strategy(
                "Sell with QSBS §1202 exclusion",
                V - tax_qsbs, exp, vol, ce, tax_qsbs, 0.0, "immediate",
                f"Excludes ${excluded:,.0f} of gain federally, saving "
                f"${fed_saved:,.0f}. Requires a 5-year hold, C-corp issuer and "
                "<$75M gross assets at issuance. Stacking across non-grantor "
                "trusts can multiply the cap — confirm with counsel first.",
            )
        )

    # ---- 7. Charitable remainder unitrust (CRUT) ---------------------------
    # Only rational to the extent the family has genuine charitable intent.
    if inp.charitable_intent > 0:
        contrib = min(inp.charitable_intent, V)
        rest = V - contrib
        payout = 0.05
        # The sale inside the CRUT is tax-free; the family receives a unitrust
        # stream plus a current deduction of roughly 25% of the contribution.
        deduction_value = contrib * 0.25 * tax.ordinary_rate
        crut_remainder = contrib * ((1 + dr - payout) ** Y)
        stream_pv = contrib * payout * Y * 0.75
        rest_tax = max(rest - inp.cost_basis * (rest / V), 0) * ltcg
        exp, vol, ce = score(0.0, core + rest - rest_tax + deduction_value, 0.0, 0.0)
        exp += stream_pv
        ce += stream_pv
        out.append(
            Strategy(
                "CRUT + sell remainder",
                rest - rest_tax, exp, vol, ce, rest_tax, 0.0, "staged",
                f"Sells inside the trust tax-free, generates a "
                f"${deduction_value:,.0f} current deduction and pays a 5% "
                f"stream. The ~${crut_remainder:,.0f} remainder goes to "
                "charity — only attractive with real philanthropic intent.",
            )
        )

    return out


def compare(inp: ConcentrationInputs, tax: TaxProfile) -> pd.DataFrame:
    """Ranked comparison table, best certainty-equivalent first."""
    strategies = evaluate_strategies(inp, tax)
    df = pd.DataFrame([s.as_dict() for s in strategies])
    df = df.sort_values("certainty_equivalent", ascending=False).reset_index(drop=True)
    best = df["certainty_equivalent"].max()
    df["ce_vs_hold"] = df["certainty_equivalent"] - df.loc[
        df["strategy"] == "Hold (do nothing)", "certainty_equivalent"
    ].iloc[0]
    df["rank"] = range(1, len(df) + 1)
    df["pct_of_best"] = (df["certainty_equivalent"] / best).round(4)
    return df


def diversification_breakeven(inp: ConcentrationInputs, tax: TaxProfile) -> dict:
    """How much alpha the concentrated stock must deliver to justify holding.

    This is the number that ends the argument. If the required alpha exceeds
    what any reasonable person forecasts for a single name, diversifying wins.
    """
    V, core = inp.position_value, inp.other_assets
    total = V + core
    if total <= 0:
        return {"required_alpha_to_justify_holding": 0.0, "verdict": "No position."}

    gain_pct = inp.embedded_gain / V if V else 0.0
    tax_cost_pct = gain_pct * tax.ltcg_rate
    Y, gamma = inp.horizon_years, inp.risk_aversion
    w = V / total  # position as a share of net worth

    # Risk penalty, measured at the PORTFOLIO level: the extra variance the
    # concentrated position adds versus holding the diversified portfolio.
    _, _, vol_hold = _blend(
        V, core, inp.capm_return(), inp.stock_vol,
        inp.diversified_return, inp.diversified_vol, inp.correlation_to_core,
    )
    var_penalty = gamma * (vol_hold**2 - inp.diversified_vol**2) / 2

    # Selling starts from less capital, so it must make that up by compounding.
    wealth_cost = w * tax_cost_pct  # tax as a share of TOTAL net worth
    tax_hurdle = -np.log(1 - wealth_cost) / Y if wealth_cost < 1 else np.inf

    # Both penalties are portfolio-level; converting to required alpha ON THE
    # STOCK means dividing by the position's weight.
    required_alpha = (var_penalty + tax_hurdle) / w if w > 0 else 0.0

    return {
        "position_weight": round(w, 4),
        "embedded_gain_pct": round(gain_pct, 4),
        "tax_cost_pct_of_position": round(tax_cost_pct, 4),
        "tax_cost_pct_of_net_worth": round(wealth_cost, 4),
        "portfolio_vol_holding": round(float(vol_hold), 4),
        "portfolio_vol_diversified": round(inp.diversified_vol, 4),
        "tax_hurdle_annualized": round(float(tax_hurdle), 4),
        "risk_penalty_annualized": round(float(var_penalty), 4),
        "required_alpha_to_justify_holding": round(float(required_alpha), 4),
        "verdict": (
            f"Diversify — the position must beat the market by "
            f"{required_alpha:.1%} per year for {Y} years to break even, which "
            "no defensible single-name forecast supports."
            if required_alpha > 0.03
            else "Close call — stage the exit over several tax years and hedge the tail."
        ),
    }


def position_report(holding: Holding, household, horizon_years: int = 10) -> dict:
    """Convenience wrapper: run the full analysis on an actual holding."""
    inp = ConcentrationInputs(
        position_value=holding.market_value,
        cost_basis=holding.cost_basis,
        other_assets=household.total_assets - holding.market_value,
        stock_vol=holding.vol or 0.42,
        stock_beta=holding.beta or 1.15,
        horizon_years=horizon_years,
        qsbs_eligible=any(l.qsbs_eligible for l in holding.lots),
    )
    table = compare(inp, household.tax)
    return {
        "ticker": holding.ticker,
        "position_value": holding.market_value,
        "pct_of_net_worth": round(holding.market_value / household.total_assets, 4),
        "embedded_gain": round(holding.unrealized_gain, 0),
        "breakeven": diversification_breakeven(inp, household.tax),
        "strategies": table.to_dict(orient="records"),
        "recommendation": table.iloc[0]["strategy"],
    }
