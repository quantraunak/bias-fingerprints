"""Direct-indexing tax-loss harvesting.

Loss harvesting is the only "alpha" in this system with a near-certain sign:
it does not depend on forecasting anything. You hold the index as individual
lots, sell losers into a correlated replacement, bank the loss against gains,
and defer the tax. The value is real but decays — as the portfolio appreciates,
fewer lots sit below basis.

The simulation is lot-level and Monte Carlo, because the two things that drive
the answer (dispersion across names and the survival of loss lots) cannot be
captured by a single-path or index-level model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..types import TaxProfile


@dataclass
class HarvestConfig:
    n_stocks: int = 250  # names held directly in the SMA
    market_vol: float = 0.16
    idio_vol: float = 0.26  # single-name vol beyond the market factor
    market_drift: float = 0.075
    dividend_yield: float = 0.013
    harvest_threshold: float = 0.05  # harvest a lot once >5% below basis
    wash_sale_days: int = 31
    rebalance_days: int = 21  # scan for harvests monthly
    years: int = 10
    contributions_per_year: float = 0.0  # new money creates fresh high basis
    n_paths: int = 200
    seed: int = 7
    management_fee: float = 0.0025  # direct-index SMA fee vs ~0.03% for an ETF
    etf_fee: float = 0.0003
    # Harvested losses are only worth cash if the family has gains to absorb
    # them. Excess losses carry forward indefinitely but their value is
    # discounted by the delay. Expressed as a fraction of portfolio value per
    # year; the default assumes an active family office realizing gains from
    # rebalancing, private-markets distributions and a concentration unwind.
    annual_gain_capacity_pct: float = 0.06
    carryforward_discount: float = 0.06  # rate at which deferred usage is PV'd


@dataclass
class HarvestResult:
    annual_harvest_by_year: pd.Series  # losses harvested / portfolio value
    utilized_by_year: pd.Series  # losses actually usable against gains
    annual_tax_alpha_by_year: pd.Series  # after fee differential
    cumulative_tax_savings: float
    avg_annual_tax_alpha: float
    first_year_tax_alpha: float
    terminal_embedded_gain_pct: float
    deferral_liability: float
    net_of_fee_alpha: float

    def summary(self) -> dict:
        return {
            "first_year_tax_alpha": round(self.first_year_tax_alpha, 4),
            "avg_annual_tax_alpha": round(self.avg_annual_tax_alpha, 4),
            "net_of_fee_alpha": round(self.net_of_fee_alpha, 4),
            "cumulative_tax_savings": round(self.cumulative_tax_savings, 0),
            "terminal_embedded_gain_pct": round(self.terminal_embedded_gain_pct, 4),
            "deferral_liability": round(self.deferral_liability, 0),
            "by_year": {int(k): round(v, 4) for k, v in self.annual_tax_alpha_by_year.items()},
            "gross_harvest_by_year": {
                int(k): round(v, 4) for k, v in self.annual_harvest_by_year.items()
            },
            "utilized_by_year": {int(k): round(v, 4) for k, v in self.utilized_by_year.items()},
        }


def simulate_harvesting(
    portfolio_value: float,
    tax: TaxProfile,
    config: HarvestConfig | None = None,
) -> HarvestResult:
    """Monte Carlo lot-level harvesting simulation.

    Model: each stock follows a one-factor GBM (common market factor plus
    idiosyncratic noise). Lots are tracked individually; a lot more than
    `harvest_threshold` below basis is sold and replaced at the current price,
    which resets basis and blocks re-harvesting until it falls again.
    """
    cfg = config or HarvestConfig()
    rng = np.random.default_rng(cfg.seed)

    steps_per_year = int(252 / cfg.rebalance_days)
    n_steps = cfg.years * steps_per_year
    dt = cfg.rebalance_days / 252

    credit_rate = tax.harvest_credit_rate
    harvest_by_year = np.zeros((cfg.n_paths, cfg.years))
    terminal_gain_pct = np.zeros(cfg.n_paths)

    for p in range(cfg.n_paths):
        n = cfg.n_stocks
        # Each name starts as one lot at price 1.0, equally weighted.
        # Lots are (shares, basis_price) arrays that grow as we harvest.
        prices = np.ones(n)
        shares = np.full(n, portfolio_value / n)
        basis = np.ones(n)  # basis per share, per name
        # Track a second-generation lot per name so harvesting doesn't collapse
        # to a single average basis (the main source of overstatement).
        shares2 = np.zeros(n)
        basis2 = np.zeros(n)
        blocked_until = np.full(n, -1)

        pv = portfolio_value

        for step in range(n_steps):
            year = step // steps_per_year

            mkt = (cfg.market_drift - cfg.dividend_yield - 0.5 * cfg.market_vol**2) * dt + \
                cfg.market_vol * np.sqrt(dt) * rng.standard_normal()
            idio = -0.5 * cfg.idio_vol**2 * dt + \
                cfg.idio_vol * np.sqrt(dt) * rng.standard_normal(n)
            prices = prices * np.exp(mkt + idio)

            harvestable = np.zeros(n)
            open_lot = step >= blocked_until

            # Lot 1
            loss1 = (basis - prices) * shares
            do1 = open_lot & (prices < basis * (1 - cfg.harvest_threshold)) & (shares > 0)
            harvestable += np.where(do1, loss1, 0.0)
            # Lot 2
            loss2 = (basis2 - prices) * shares2
            do2 = open_lot & (shares2 > 0) & (prices < basis2 * (1 - cfg.harvest_threshold))
            harvestable += np.where(do2, loss2, 0.0)

            harvested = harvestable.sum()
            if harvested > 0:
                # Reset basis on harvested lots to the current price; the
                # replacement security is bought immediately (correlated proxy),
                # so market exposure is continuous. Block re-harvest for 31 days.
                basis = np.where(do1, prices, basis)
                basis2 = np.where(do2, prices, basis2)
                blocked_until = np.where(
                    do1 | do2, step + int(np.ceil(cfg.wash_sale_days / cfg.rebalance_days)), blocked_until
                )
                harvest_by_year[p, year] += harvested

            pv = float((shares * prices).sum() + (shares2 * prices).sum())

            # New contributions arrive as fresh lots at current prices — this is
            # what keeps harvest capacity alive in later years.
            if cfg.contributions_per_year > 0 and (step + 1) % steps_per_year == 0:
                add = cfg.contributions_per_year / n
                new_shares = add / prices
                # Fold into lot 2 at a share-weighted basis.
                tot = shares2 + new_shares
                basis2 = np.where(tot > 0, (basis2 * shares2 + prices * new_shares) / np.maximum(tot, 1e-12), prices)
                shares2 = tot

        mv = float((shares * prices).sum() + (shares2 * prices).sum())
        cost = float((shares * basis).sum() + (shares2 * basis2).sum())
        terminal_gain_pct[p] = (mv - cost) / mv if mv > 0 else 0.0

    # Average across paths, expressed as a fraction of starting value.
    mean_harvest = harvest_by_year.mean(axis=0) / portfolio_value
    years_index = pd.Index(range(1, cfg.years + 1), name="year")
    harvest_series = pd.Series(mean_harvest, index=years_index)

    # Apply the utilization constraint. Losses beyond the family's realized
    # gains in a year carry forward and are used later, so their value is
    # discounted rather than lost. Skipping this step is what produces the
    # implausible first-year tax-alpha figures quoted in sales material.
    capacity = cfg.annual_gain_capacity_pct
    used = np.zeros(cfg.years)
    carry = 0.0
    for i, harvested_pct in enumerate(mean_harvest):
        available = harvested_pct + carry
        applied = min(available, capacity)
        used[i] = applied
        carry = available - applied
    # Whatever is still carried forward at the end is used eventually; PV it
    # over an assumed five-year average delay.
    residual_pv = carry / (1 + cfg.carryforward_discount) ** 5
    used[-1] += residual_pv

    utilized_series = pd.Series(used, index=years_index)
    gross_alpha = utilized_series * credit_rate
    fee_drag = cfg.management_fee - cfg.etf_fee
    net_alpha = gross_alpha - fee_drag

    cumulative_savings = float(used.sum() * credit_rate * portfolio_value)
    terminal_gain = float(terminal_gain_pct.mean())

    # Harvesting defers, it does not erase. Track the liability honestly.
    final_value = portfolio_value * (1 + cfg.market_drift) ** cfg.years
    deferral_liability = final_value * terminal_gain * tax.ltcg_rate

    return HarvestResult(
        annual_harvest_by_year=harvest_series,
        utilized_by_year=utilized_series,
        annual_tax_alpha_by_year=net_alpha,
        cumulative_tax_savings=cumulative_savings,
        avg_annual_tax_alpha=float(gross_alpha.mean()),
        first_year_tax_alpha=float(gross_alpha.iloc[0]),
        terminal_embedded_gain_pct=terminal_gain,
        deferral_liability=deferral_liability,
        net_of_fee_alpha=float(net_alpha.mean()),
    )


def harvest_capacity_now(household, prices: dict[str, float] | None = None) -> pd.DataFrame:
    """Losses available for harvest in the CURRENT portfolio, lot by lot.

    This is the immediately actionable version: what can be sold today.
    """
    rows = []
    asof = pd.Timestamp.today()
    for acct in household.accounts:
        if not acct.is_taxable:
            continue
        for h in acct.holdings:
            px = (prices or {}).get(h.ticker, h.price)
            for lot in h.lots:
                unreal = lot.unrealized(px)
                if unreal >= 0:
                    continue
                long_term = lot.is_long_term(asof)
                rate = household.tax.ltcg_rate if long_term else household.tax.stcg_rate
                rows.append(
                    {
                        "account": acct.account_id,
                        "ticker": h.ticker,
                        "shares": lot.shares,
                        "acquired": lot.acquired.date().isoformat(),
                        "market_value": round(lot.market_value(px), 0),
                        "unrealized_loss": round(unreal, 0),
                        "term": "long" if long_term else "short",
                        "tax_benefit": round(abs(unreal) * rate, 0),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("tax_benefit", ascending=False).reset_index(drop=True)
