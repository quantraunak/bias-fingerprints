"""Growth-maximizing configuration and target-return feasibility.

Answers the question every family actually asks: "can you beat the number I
already earn?" — with a probability rather than a promise.

The return build-up below is additive and auditable. Each lever is listed with
the confidence tier it deserves, because a plan that reaches its target only by
stacking low-confidence levers is not a plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Lever:
    name: str
    contribution: float  # annual return added at the PORTFOLIO level
    confidence: str  # high | medium | low
    basis: str


@dataclass
class GrowthConfig:
    """A deliberately aggressive but investable configuration."""

    equity_beta: float = 1.05
    base_equity_return: float = 0.070
    unlevered_vol: float = 0.140
    leverage: float = 1.25
    borrow_spread: float = 0.012
    cash_rate: float = 0.037
    horizon_years: int = 20

    @property
    def borrow_rate(self) -> float:
        return self.cash_rate + self.borrow_spread


def growth_levers(cfg: GrowthConfig | None = None) -> list[Lever]:
    cfg = cfg or GrowthConfig()
    return [
        Lever(
            "Global equity beta",
            cfg.base_equity_return,
            "high",
            "Grinold-Kroner build-up at current dividend, buyback and growth "
            "rates, with partial valuation reversion. This is the engine; "
            "everything else is a modifier.",
        ),
        Lever(
            "Factor tilts (value, momentum, quality)",
            0.010,
            "medium",
            "Well-documented premia, but heavily arbitraged and capable of "
            "decade-long drawdowns. Sized as a tilt, not a bet.",
        ),
        Lever(
            "Tax-aware long/short extension",
            0.020,
            "high",
            "A 130/30 or 150/50 extension harvests losses on BOTH legs, so it "
            "keeps generating deductions long after a long-only direct-index "
            "portfolio has run dry. The tax component is mechanical; the "
            "pre-tax alpha component is not.",
        ),
        Lever(
            "Asset location and withdrawal sequencing",
            0.004,
            "high",
            "Same holdings, different wrappers. Pure arithmetic.",
        ),
        Lever(
            "Private markets illiquidity premium",
            0.005,
            "low",
            "Real in aggregate but swamped by manager dispersion — the gap "
            "between top and bottom quartile PE exceeds 15%/yr. Only counts "
            "if access is genuinely top-half.",
        ),
        Lever(
            "Cost compression",
            0.005,
            "high",
            "Moving all-in costs from ~110bps to ~55bps. The only input known "
            "with certainty in advance.",
        ),
    ]


def unlevered_return(cfg: GrowthConfig | None = None) -> float:
    return sum(l.contribution for l in growth_levers(cfg))


def levered_profile(cfg: GrowthConfig | None = None) -> dict:
    """Apply leverage and report the resulting risk/return profile."""
    cfg = cfg or GrowthConfig()
    r_u = unlevered_return(cfg)
    L = cfg.leverage
    r_l = L * r_u - (L - 1) * cfg.borrow_rate
    vol_l = L * cfg.unlevered_vol
    # Geometric (compound) return is what actually accrues to the family.
    geo = r_l - vol_l**2 / 2
    return {
        "unlevered_arithmetic": round(r_u, 4),
        "unlevered_vol": round(cfg.unlevered_vol, 4),
        "leverage": L,
        "borrow_rate": round(cfg.borrow_rate, 4),
        "levered_arithmetic": round(r_l, 4),
        "levered_vol": round(vol_l, 4),
        "levered_geometric": round(geo, 4),
        "expected_max_drawdown": round(-vol_l * 2.4, 4),
        "sharpe": round((r_l - cfg.cash_rate) / vol_l, 3),
    }


def probability_of_beating(
    target_cagr: float,
    arithmetic_return: float,
    volatility: float,
    years: int = 20,
    n_paths: int = 200_000,
    seed: int = 11,
    fat_tails: bool = True,
) -> float:
    """P(realized CAGR over `years` exceeds `target_cagr`).

    Uses Student-t innovations so the answer reflects the fat left tail that
    lognormal models miss.
    """
    rng = np.random.default_rng(seed)
    sigma_log = volatility / (1 + arithmetic_return)
    mu_log = np.log(1 + arithmetic_return) - 0.5 * sigma_log**2

    if fat_tails:
        df = 5
        z = rng.standard_t(df, size=(n_paths, years)) / np.sqrt(df / (df - 2))
    else:
        z = rng.standard_normal((n_paths, years))

    log_returns = mu_log + sigma_log * z
    total_log = log_returns.sum(axis=1)
    realized_cagr = np.exp(total_log / years) - 1
    return float((realized_cagr >= target_cagr).mean())


def target_feasibility(
    targets: list[float] | None = None,
    cfg: GrowthConfig | None = None,
    years: int = 20,
) -> pd.DataFrame:
    """Probability table across candidate return targets."""
    cfg = cfg or GrowthConfig()
    targets = targets or [0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    prof = levered_profile(cfg)

    rows = []
    for t in targets:
        p = probability_of_beating(
            t, prof["levered_arithmetic"], prof["levered_vol"], years
        )
        rows.append(
            {
                "target_cagr": t,
                "probability": round(p, 4),
                "odds": f"1 in {round(1 / p):,}" if p > 0 else "worse than 1 in 100,000",
            }
        )
    return pd.DataFrame(rows)


def concentration_required_for(
    target_cagr: float,
    cfg: GrowthConfig | None = None,
    single_name_vol: float = 0.42,
) -> dict:
    """How much single-name concentration a target implicitly requires.

    If a diversified portfolio cannot reach the target with acceptable
    probability, the only remaining route is concentration. This quantifies
    the trade rather than leaving it unsaid.
    """
    cfg = cfg or GrowthConfig()
    prof = levered_profile(cfg)
    base = prof["levered_arithmetic"]
    gap = target_cagr + prof["levered_vol"] ** 2 / 2 - base

    if gap <= 0:
        return {"required": False, "note": "Diversified portfolio reaches the target in expectation."}

    # Assume a concentrated name is expected to beat the market by `edge`.
    # Even a generous 5% edge requires a large weight to move the total.
    for edge in (0.03, 0.05, 0.08):
        w = gap / edge
        if w <= 1.0:
            blended_vol = float(
                np.sqrt(
                    (w * single_name_vol) ** 2
                    + ((1 - w) * prof["levered_vol"]) ** 2
                    + 2 * w * (1 - w) * 0.6 * single_name_vol * prof["levered_vol"]
                )
            )
            return {
                "required": True,
                "assumed_single_name_edge": edge,
                "required_weight_in_one_name": round(w, 3),
                "resulting_portfolio_vol": round(blended_vol, 4),
                "resulting_expected_drawdown": round(-blended_vol * 2.4, 4),
                "note": (
                    f"Reaching {target_cagr:.0%} requires roughly {w:.0%} of net worth in a "
                    f"single name expected to beat the market by {edge:.0%}/yr — lifting "
                    f"portfolio volatility to {blended_vol:.0%} and expected worst drawdown "
                    f"to {-blended_vol * 2.4:.0%}. That is a concentration decision, not a "
                    "diversification strategy, and it is how most such track records were built."
                ),
            }
    return {
        "required": True,
        "note": "Target is not reachable even with extreme concentration under these assumptions.",
    }


def growth_report(target_cagr: float = 0.18, years: int = 20) -> dict:
    cfg = GrowthConfig(horizon_years=years)
    levers = growth_levers(cfg)
    prof = levered_profile(cfg)
    by_conf: dict[str, float] = {}
    for l in levers:
        by_conf[l.confidence] = by_conf.get(l.confidence, 0.0) + l.contribution

    return {
        "levers": [
            {
                "name": l.name,
                "contribution": round(l.contribution, 4),
                "confidence": l.confidence,
                "basis": l.basis,
            }
            for l in levers
        ],
        "contribution_by_confidence": {k: round(v, 4) for k, v in by_conf.items()},
        "profile": prof,
        "feasibility": target_feasibility(cfg=cfg, years=years).to_dict(orient="records"),
        "target": target_cagr,
        "probability_of_target": round(
            probability_of_beating(target_cagr, prof["levered_arithmetic"], prof["levered_vol"], years), 4
        ),
        "concentration_required": concentration_required_for(target_cagr, cfg),
        "passive_baseline": {
            "sixty_forty_return": 0.058,
            "engineered_uplift": round(prof["levered_geometric"] - 0.058, 4),
        },
    }
