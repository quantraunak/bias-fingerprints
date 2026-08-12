"""Estate and wealth-transfer planning.

For a $10-50M household the estate tax is frequently the largest single
lifetime "return" decision — a 40% rate on the excess above the exemption
dwarfs any plausible alpha. Moving appreciating assets out of the estate early
compounds outside the tax base for decades.

The tension modelled here: assets gifted during life carry over basis (no
step-up at death), so gifting is not free. The right answer depends on whether
the estate is projected to exceed the exemption — which is a forecasting
problem, handled here on the wealth path rather than today's balance sheet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..types import Household, TaxProfile


@dataclass
class EstateInputs:
    current_estate: float
    growth_rate: float = 0.065
    years_to_transfer: int = 25  # life expectancy horizon
    n_spouses: int = 2
    annual_exclusion_per_donee: float = 19_000.0
    n_donees: int = 4
    already_used_exemption: float = 0.0
    state_estate_tax: float = 0.0  # e.g. 0.16 in NY/MA/WA; 0 in CA/TX/FL
    state_estate_exemption: float = 0.0


@dataclass
class TransferStrategy:
    name: str
    amount_transferred: float
    estate_tax_saved: float
    basis_step_up_lost: float
    net_benefit: float
    complexity: str
    notes: str

    def as_dict(self) -> dict:
        return {
            "strategy": self.name,
            "amount_transferred": round(self.amount_transferred, 0),
            "estate_tax_saved": round(self.estate_tax_saved, 0),
            "basis_step_up_lost": round(self.basis_step_up_lost, 0),
            "net_benefit": round(self.net_benefit, 0),
            "complexity": self.complexity,
            "notes": self.notes,
        }


def projected_estate(inp: EstateInputs) -> float:
    return inp.current_estate * (1 + inp.growth_rate) ** inp.years_to_transfer


def estate_tax_exposure(inp: EstateInputs, tax: TaxProfile) -> dict:
    """Projected federal + state estate tax with no further planning."""
    future = projected_estate(inp)
    exemption = tax.estate_exemption_per_person * inp.n_spouses - inp.already_used_exemption
    # The exemption is indexed to inflation; grow it at ~2.4%.
    exemption_future = exemption * (1.024) ** inp.years_to_transfer

    fed_taxable = max(future - exemption_future, 0.0)
    fed_tax = fed_taxable * tax.estate_tax_rate

    state_taxable = max(future - inp.state_estate_exemption, 0.0) if inp.state_estate_tax else 0.0
    state_tax = state_taxable * inp.state_estate_tax

    total = fed_tax + state_tax
    return {
        "current_estate": round(inp.current_estate, 0),
        "projected_estate": round(future, 0),
        "exemption_at_transfer": round(exemption_future, 0),
        "federal_estate_tax": round(fed_tax, 0),
        "state_estate_tax": round(state_tax, 0),
        "total_estate_tax": round(total, 0),
        "effective_rate_on_estate": round(total / future, 4) if future else 0.0,
        "heirs_receive": round(future - total, 0),
    }


def evaluate_transfers(
    inp: EstateInputs, tax: TaxProfile, embedded_gain_pct: float = 0.35
) -> list[TransferStrategy]:
    """Price the standard wealth-transfer toolkit."""
    Y = inp.years_to_transfer
    g = inp.growth_rate
    rate = tax.estate_tax_rate + inp.state_estate_tax
    out: list[TransferStrategy] = []

    exposure = estate_tax_exposure(inp, tax)
    has_exposure = exposure["total_estate_tax"] > 0
    if not has_exposure:
        out.append(
            TransferStrategy(
                "No transfer planning required",
                0.0, 0.0, 0.0, 0.0, "none",
                "Projected estate sits below the exemption. Prioritize basis "
                "step-up: hold appreciated assets until death rather than "
                "gifting them. Revisit if the exemption is reduced by statute.",
            )
        )
        return out

    def step_up_lost(amount: float) -> float:
        """Gifted assets carry over basis; heirs eventually pay LTCG on the
        gain that would otherwise have been erased at death."""
        future_val = amount * (1 + g) ** Y
        gain = future_val * embedded_gain_pct + (future_val - amount)
        return gain * tax.ltcg_rate

    # ---- Annual exclusion gifting ------------------------------------------
    annual = inp.annual_exclusion_per_donee * inp.n_donees * inp.n_spouses
    fv_annual = sum(annual * (1 + g) ** (Y - t) for t in range(Y))
    out.append(
        TransferStrategy(
            "Annual exclusion gifting",
            fv_annual,
            fv_annual * rate,
            step_up_lost(annual * Y),
            fv_annual * rate - step_up_lost(annual * Y),
            "low",
            f"${annual:,.0f}/yr moves out of the estate with no exemption use "
            "and no gift tax return required. The simplest available lever.",
        )
    )

    # ---- Lifetime exemption gift to a SLAT ---------------------------------
    exemption_available = max(
        tax.estate_exemption_per_person * inp.n_spouses - inp.already_used_exemption, 0.0
    )
    gift = min(exemption_available, inp.current_estate * 0.35)
    fv_gift = gift * (1 + g) ** Y
    appreciation_removed = fv_gift - gift
    out.append(
        TransferStrategy(
            "Lifetime gift to SLAT (grantor trust)",
            fv_gift,
            appreciation_removed * rate,
            step_up_lost(gift),
            appreciation_removed * rate - step_up_lost(gift),
            "medium",
            f"Gift ${gift:,.0f} now; all future appreciation "
            f"(${appreciation_removed:,.0f}) grows outside the estate. As a "
            "grantor trust the donor pays its income tax — an additional "
            "tax-free transfer every year. Spousal access preserves optionality; "
            "beware the reciprocal-trust doctrine if both spouses create one.",
        )
    )

    # ---- GRAT ladder --------------------------------------------------------
    # A zeroed-out GRAT transfers only the excess over the §7520 hurdle. Rolling
    # short-term GRATs is close to a free option: no exemption used, downside
    # limited to setup cost.
    hurdle = 0.052
    grat_funding = inp.current_estate * 0.20
    excess_growth = max(g - hurdle, 0.0)
    fv_grat = grat_funding * ((1 + g) ** 2 - (1 + hurdle) ** 2) * (Y / 2) * 0.5
    fv_grat = max(fv_grat, 0.0)
    out.append(
        TransferStrategy(
            "Rolling 2-year GRAT ladder",
            fv_grat,
            fv_grat * rate,
            step_up_lost(fv_grat * 0.5),
            fv_grat * rate - step_up_lost(fv_grat * 0.5),
            "medium",
            f"Transfers only growth above the §7520 rate ({hurdle:.1%}). Uses no "
            "lifetime exemption and the downside is limited to setup cost — "
            f"the strategy simply fails if returns undershoot. Excess growth "
            f"assumption: {excess_growth:.1%}/yr.",
        )
    )

    # ---- Sale to an IDGT ----------------------------------------------------
    idgt_seed = inp.current_estate * 0.10
    idgt_note = idgt_seed * 9  # 10% seed, 9x note
    afr = 0.043
    idgt_excess = (idgt_note * ((1 + g) ** Y - (1 + afr) ** Y))
    idgt_excess = max(idgt_excess, 0.0)
    out.append(
        TransferStrategy(
            "Installment sale to IDGT",
            idgt_excess,
            idgt_excess * rate,
            step_up_lost(idgt_seed),
            idgt_excess * rate - step_up_lost(idgt_seed),
            "high",
            f"Sell assets to a grantor trust for a note at the AFR ({afr:.1%}); "
            "no capital gain on the sale (the trust is the grantor for income "
            "tax purposes) and all growth above the AFR passes free of transfer "
            "tax. Best paired with valuation discounts on non-marketable "
            "interests. Requires appraisal and disciplined administration.",
        )
    )

    # ---- Charitable lead annuity trust --------------------------------------
    if inp.current_estate > 20_000_000:
        clat_funding = inp.current_estate * 0.10
        clat_excess = max(clat_funding * ((1 + g) ** Y - (1 + hurdle) ** Y), 0.0) * 0.5
        out.append(
            TransferStrategy(
                "Charitable lead annuity trust (CLAT)",
                clat_excess,
                clat_excess * rate,
                0.0,
                clat_excess * rate,
                "high",
                "Pays an annuity to charity for a term; the remainder passes to "
                "heirs free of transfer tax. A zeroed-out CLAT works best in a "
                "low §7520-rate environment and pairs with a large income year.",
            )
        )

    return sorted(out, key=lambda s: s.net_benefit, reverse=True)


def estate_report(household: Household, growth_rate: float = 0.065) -> dict:
    """Full estate analysis anchored to the household balance sheet."""
    embedded = (
        household.unrealized_gain / household.total_assets if household.total_assets else 0.0
    )
    inp = EstateInputs(
        current_estate=household.total_assets,
        growth_rate=growth_rate,
        years_to_transfer=household.horizon_years,
        n_donees=max(household.n_beneficiaries * 2, 2),
    )
    exposure = estate_tax_exposure(inp, household.tax)
    strategies = evaluate_transfers(inp, household.tax, embedded_gain_pct=max(embedded, 0.0))

    # Transfer strategies OVERLAP — a SLAT gift, a GRAT ladder and an IDGT sale
    # all move the same dollars out of the same estate. Summing them double,
    # triple and quadruple counts the benefit. A family realistically runs
    # annual exclusion gifting (which stacks with anything) plus ONE primary
    # structure, so that is what we credit.
    stackable = sum(
        max(s.net_benefit, 0)
        for s in strategies
        if s.name == "Annual exclusion gifting"
    )
    primary = max(
        (
            max(s.net_benefit, 0)
            for s in strategies
            if s.name != "Annual exclusion gifting"
        ),
        default=0.0,
    )
    total_saved = stackable + primary
    return {
        "exposure": exposure,
        "strategies": [s.as_dict() for s in strategies],
        "total_potential_savings": round(total_saved, 0),
        "savings_as_pct_of_estate": round(
            total_saved / exposure["projected_estate"], 4
        ) if exposure["projected_estate"] else 0.0,
        "priority": (
            "Estate planning is the highest-value action on this balance sheet — "
            "the projected transfer tax exceeds a decade of plausible alpha."
            if exposure["total_estate_tax"] > household.total_assets * 0.10
            else "Moderate exposure. Execute annual gifting and revisit if the "
            "exemption is legislated downward."
        ),
    }


def strategies_table(household: Household) -> pd.DataFrame:
    rep = estate_report(household)
    return pd.DataFrame(rep["strategies"])
