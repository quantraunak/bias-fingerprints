"""Systematic alpha sleeve — sizing an uncorrelated return stream.

This reads live results from the r1000-ls-strategy backtest when available and
answers the only question that matters at the household level: how much of a
market-neutral sleeve should the family hold, and what does it actually do to
portfolio outcomes after fees, taxes, and funding costs?

The honest framing: a market-neutral sleeve is NOT a return enhancer at this
size. Its value is that it lets the family carry the SAME total risk with less
equity beta, or more expected return at the same risk. It also has the worst
tax profile of anything in the portfolio, so it belongs in deferred accounts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Where the backtest writes its output. Falls back to documented results.
R1000_REPORTS = Path.home() / "Projects" / "r1000-ls-strategy" / "project" / "reports" / "latest"


@dataclass
class SleeveStats:
    """Backtest statistics, always haircut before use in allocation."""

    source: str
    gross_sharpe: float
    gross_return: float
    volatility: float
    max_drawdown: float
    beta_to_equity: float
    turnover: float
    sharpe_ci_low: float | None = None
    sharpe_ci_high: float | None = None

    def haircut(
        self,
        overfitting_factor: float = 0.55,
        fee: float = 0.015,
        capacity_haircut: float = 0.10,
    ) -> "SleeveStats":
        """Deflate backtest results to a live expectation.

        Three deductions, all of which are routinely omitted and all of which
        are real:
          - Overfitting/selection: published backtest Sharpe historically
            realizes at roughly half live. Use the LOWER bound of the
            bootstrap CI when one is available.
          - Fees and financing on the short book.
          - Capacity: the strategy trades ~120%/month; slippage rises with AUM.
        """
        base = self.sharpe_ci_low if self.sharpe_ci_low is not None else self.gross_sharpe
        live_sharpe = base * overfitting_factor * (1 - capacity_haircut)
        live_return = live_sharpe * self.volatility - fee
        return SleeveStats(
            source=f"{self.source} (haircut)",
            gross_sharpe=max(live_return / self.volatility, 0.0) if self.volatility else 0.0,
            gross_return=live_return,
            volatility=self.volatility,
            max_drawdown=self.max_drawdown * 1.4,  # live drawdowns exceed backtest
            beta_to_equity=self.beta_to_equity,
            turnover=self.turnover,
        )


def load_r1000_stats(reports_dir: Path | None = None) -> SleeveStats:
    """Read the most recent backtest summary; fall back to README figures."""
    d = reports_dir or R1000_REPORTS
    if d.exists():
        runs = sorted([p for p in d.iterdir() if p.is_dir()], reverse=True)
        for run in runs:
            f = run / "performance_summary.json"
            if not f.exists():
                continue
            try:
                data = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue

            def g(*names, default=None):
                for nm in names:
                    if nm in data and data[nm] is not None:
                        return float(data[nm])
                return default

            ci = data.get("sharpe_ci") or data.get("sharpe_95_ci")
            ci_low = float(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else g("sharpe_ci_low")
            ci_high = float(ci[1]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else g("sharpe_ci_high")

            return SleeveStats(
                source=f"r1000-ls-strategy {run.name}",
                gross_sharpe=g("sharpe", "Sharpe", default=1.46),
                gross_return=g("cagr", "CAGR", default=0.353),
                volatility=g("volatility", "annual_volatility", default=0.224),
                max_drawdown=abs(g("max_drawdown", "max_dd", default=0.247)),
                beta_to_equity=abs(g("beta", "beta_spy", default=0.14)),
                turnover=g("avg_turnover", "turnover", default=1.207),
                sharpe_ci_low=ci_low if ci_low is not None else 0.90,
                sharpe_ci_high=ci_high if ci_high is not None else 2.07,
            )

    # Documented results from the repository README.
    return SleeveStats(
        source="r1000-ls-strategy (README, 2010-2024)",
        gross_sharpe=1.46,
        gross_return=0.353,
        volatility=0.224,
        max_drawdown=0.247,
        beta_to_equity=0.14,
        turnover=1.207,
        sharpe_ci_low=0.90,
        sharpe_ci_high=2.07,
    )


def optimal_sleeve_weight(
    sleeve: SleeveStats,
    core_return: float,
    core_vol: float,
    correlation: float = 0.05,
    risk_aversion: float = 3.0,
    max_weight: float = 0.15,
) -> dict:
    """Two-asset mean-variance solution for the sleeve's share of the portfolio.

    Capped at `max_weight` because a single-manager, single-model sleeve carries
    model risk that mean-variance does not see.
    """
    mu = np.array([core_return, sleeve.gross_return])
    s = np.array([core_vol, sleeve.volatility])
    C = np.array([[1.0, correlation], [correlation, 1.0]])
    Sigma = C * np.outer(s, s)

    raw = np.linalg.solve(risk_aversion * Sigma, mu)
    raw = raw / raw.sum() if raw.sum() != 0 else np.array([1.0, 0.0])
    w_sleeve = float(np.clip(raw[1], 0.0, max_weight))
    w_core = 1.0 - w_sleeve

    w = np.array([w_core, w_sleeve])
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ Sigma @ w))
    base_sharpe = core_return / core_vol if core_vol else 0.0

    return {
        "unconstrained_weight": round(float(raw[1]), 4),
        "recommended_weight": round(w_sleeve, 4),
        "portfolio_return": round(port_ret, 4),
        "portfolio_vol": round(port_vol, 4),
        "portfolio_sharpe": round(port_ret / port_vol if port_vol else 0.0, 3),
        "baseline_sharpe": round(base_sharpe, 3),
        "sharpe_improvement": round((port_ret / port_vol if port_vol else 0) - base_sharpe, 3),
        "return_at_equal_risk": round(
            (port_ret / port_vol * core_vol) if port_vol else core_return, 4
        ),
        "equity_beta_reduction": round(w_sleeve * (1 - sleeve.beta_to_equity), 4),
    }


def sleeve_after_tax(sleeve: SleeveStats, tax, in_deferred_account: bool) -> dict:
    """The sleeve's tax problem, quantified.

    At ~120% monthly turnover essentially all gains are SHORT TERM. In a taxable
    account that converts a respectable pre-tax return into a poor one — this is
    the single most important implementation detail for the sleeve.
    """
    if in_deferred_account:
        net = sleeve.gross_return
        rate = 0.0
    else:
        rate = tax.stcg_rate
        net = sleeve.gross_return * (1 - rate)
    return {
        "pre_tax_return": round(sleeve.gross_return, 4),
        "effective_tax_rate": round(rate, 4),
        "after_tax_return": round(net, 4),
        "after_tax_sharpe": round(net / sleeve.volatility if sleeve.volatility else 0.0, 3),
        "guidance": (
            "Hold in a tax-deferred or tax-exempt wrapper. In a taxable account "
            "the short-term rate consumes roughly half the return and the sleeve "
            "stops being worth its fee."
        ),
    }


def return_stacking(
    core_return: float,
    core_vol: float,
    sleeve: SleeveStats,
    cash_rate: float,
    borrow_spread: float = 0.006,
    stack_ratio: float = 0.20,
    correlation: float = 0.05,
) -> dict:
    """Overlay the sleeve on top of a fully-invested core using financing.

    Rather than selling equities to fund the sleeve (which triggers tax and
    gives up beta), the sleeve is funded with margin or capital-efficient
    futures. This adds return only if the sleeve clears its financing cost —
    which is exactly the test applied here.
    """
    financing = cash_rate + borrow_spread
    excess = sleeve.gross_return - financing
    total_return = core_return + stack_ratio * excess
    total_vol = float(
        np.sqrt(
            core_vol**2
            + (stack_ratio * sleeve.volatility) ** 2
            + 2 * correlation * core_vol * stack_ratio * sleeve.volatility
        )
    )
    return {
        "stack_ratio": stack_ratio,
        "financing_rate": round(financing, 4),
        "sleeve_excess_over_financing": round(excess, 4),
        "stacked_return": round(total_return, 4),
        "stacked_vol": round(total_vol, 4),
        "stacked_sharpe": round((total_return - cash_rate) / total_vol if total_vol else 0.0, 3),
        "base_sharpe": round((core_return - cash_rate) / core_vol if core_vol else 0.0, 3),
        "verdict": (
            "Accretive — sleeve clears financing with room to spare"
            if excess > 0.02
            else "Marginal — do not lever this; the edge does not cover the borrow"
        ),
    }


def sleeve_report(core_return: float, core_vol: float, tax, cash_rate: float = 0.037) -> dict:
    """End-to-end sleeve analysis for the investment committee."""
    raw = load_r1000_stats()
    live = raw.haircut()
    sizing = optimal_sleeve_weight(live, core_return, core_vol)
    return {
        "backtest": {
            "source": raw.source,
            "sharpe": round(raw.gross_sharpe, 2),
            "cagr": round(raw.gross_return, 4),
            "volatility": round(raw.volatility, 4),
            "max_drawdown": round(-raw.max_drawdown, 4),
            "beta": round(raw.beta_to_equity, 3),
            "sharpe_ci": [raw.sharpe_ci_low, raw.sharpe_ci_high],
        },
        "live_expectation": {
            "sharpe": round(live.gross_sharpe, 2),
            "expected_return": round(live.gross_return, 4),
            "volatility": round(live.volatility, 4),
            "expected_max_drawdown": round(-live.max_drawdown, 4),
            "basis": (
                "Lower bound of the bootstrap Sharpe CI, multiplied by a 0.55 "
                "overfitting factor and a 10% capacity haircut, less 150bps of fees."
            ),
        },
        "sizing": sizing,
        "tax_treatment": sleeve_after_tax(live, tax, in_deferred_account=True),
        "return_stacking": return_stacking(core_return, core_vol, live, cash_rate),
        "risks": [
            "Single-model concentration: one signal set, one researcher, one codebase.",
            "Survivorship-biased universe in the backtest overstates the achievable edge.",
            "Short book carries borrow cost and recall risk not modelled in the backtest.",
            "120% monthly turnover makes the strategy capacity- and cost-sensitive.",
            "Requires 24+ months of live track record before sizing above 5%.",
        ],
    }
