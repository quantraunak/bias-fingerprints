"""The backtest loop.

Two structural differences from the engine this replaces.

**Every trading day gets a return.** Previously, a rebalance that could not be
computed caused the loop to `continue` *before* accruing the period's P&L, so
that stretch vanished from the equity curve entirely -- 29% of the sample. Here
the position book is carried forward when a rebalance cannot be produced and the
reason is recorded, so a failure shows up as a flat period rather than as a hole.

**Positions drift.** Holding fixed weights between rebalances quietly assumes a
free daily rebalance back to target. Here the book is marked to market each day
and weights move with prices, which is what actually happens between trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config
from src.data.panel import Panel
from src.model.walkforward import month_end_trading_days
from src.portfolio import costs as cost_module
from src.portfolio import optimizer, risk


@dataclass
class BacktestResult:
    net_returns: pd.Series
    gross_returns: pd.Series
    holdings: pd.DataFrame
    rebalances: pd.DataFrame
    skipped: list[dict] = field(default_factory=list)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net_returns).cumprod()


def run(panel: Panel, scores: pd.Series, config: Config) -> BacktestResult:
    returns = panel.returns
    trading_days = returns.index
    rebalance_dates = month_end_trading_days(trading_days)
    rebalance_dates = rebalance_dates[rebalance_dates >= pd.Timestamp(config.oos_start)]
    if len(rebalance_dates) == 0:
        raise ValueError("No rebalance dates in the out-of-sample window.")

    schedule = set(rebalance_dates)
    days = trading_days[(trading_days >= rebalance_dates[0]) & (trading_days <= trading_days[-1])]

    positions = pd.Series(dtype=float)
    net, gross, holdings, records, skipped = {}, {}, {}, [], []
    equity_so_far = pd.Series(dtype=float)

    for day in days:
        daily = returns.loc[day]
        pnl = float((positions * daily.reindex(positions.index)).fillna(0.0).sum()) if len(positions) else 0.0

        # Mark the book to market: weights move with prices between trades.
        if len(positions):
            growth = (1.0 + daily.reindex(positions.index).fillna(0.0))
            positions = positions * growth / (1.0 + pnl)

        charge = 0.0
        if day in schedule:
            target = _target_weights(panel, scores, config, day, positions, equity_so_far, skipped)
            if target is not None:
                charge = cost_module.cost(
                    positions, target, config.costs.commission_bps, config.costs.slippage_bps
                )
                records.append(
                    {
                        "date": day,
                        "turnover": cost_module.turnover(positions, target),
                        "cost": charge,
                        "n_names": int((target.abs() > 1e-8).sum()),
                        "gross": float(target.abs().sum()),
                        "net": float(target.sum()),
                        "transfer_coefficient": target.attrs.get("transfer_coefficient"),
                        "n_candidates": target.attrs.get("n_candidates"),
                    }
                )
                holdings[day] = target
                positions = target

        gross[day] = pnl
        net[day] = pnl - charge
        equity_so_far = pd.Series(net).sort_index()

    return BacktestResult(
        net_returns=pd.Series(net).sort_index(),
        gross_returns=pd.Series(gross).sort_index(),
        holdings=pd.DataFrame(holdings).T.sort_index(),
        rebalances=pd.DataFrame(records).set_index("date") if records else pd.DataFrame(),
        skipped=skipped,
    )


def _target_weights(
    panel: Panel,
    scores: pd.Series,
    config: Config,
    day: pd.Timestamp,
    positions: pd.Series,
    equity_so_far: pd.Series,
    skipped: list[dict],
) -> pd.Series | None:
    """Deciles of the score, optimised under the risk constraints. None means hold."""
    if day not in scores.index.get_level_values("date"):
        skipped.append({"date": day, "reason": "no_scores"})
        return None

    today = scores.xs(day, level="date").dropna()
    tradable = panel.tradable.loc[day]
    today = today[today.index.isin(tradable[tradable].index)]

    if config.portfolio.selection == "full":
        # Hand the optimiser the whole cross-section. Truncating to deciles
        # first discards the ordering information across the middle of the
        # distribution -- most of the IC -- before the optimiser ever sees it,
        # and no risk term can recover what was thrown away upstream.
        if len(today) > config.portfolio.max_cross_section:
            half = config.portfolio.max_cross_section // 2
            selected = pd.concat([today.nlargest(half), today.nsmallest(half)])
        else:
            selected = today
        if len(selected) < 20:
            skipped.append({"date": day, "reason": "cross_section_too_small", "n": len(today)})
            return None
    else:
        n_side = int(len(today) * config.portfolio.quantile)
        if n_side < 5:
            skipped.append({"date": day, "reason": "cross_section_too_small", "n": len(today)})
            return None
        selected = pd.concat([today.nlargest(n_side), today.nsmallest(n_side)])

    selected = selected[~selected.index.duplicated(keep="first")]
    names = selected.index.tolist()

    history = panel.returns[names].loc[:day].iloc[-config.portfolio.cov_window :]
    covariance = risk.shrunk_covariance(history)
    names = [n for n in names if n in covariance.columns]
    selected = selected.reindex(names)

    betas = risk.betas(
        panel.returns[names].loc[:day].iloc[-config.portfolio.beta_window :],
        panel.market.loc[:day].iloc[-config.portfolio.beta_window :],
    )

    scale = risk.drawdown_scale(
        equity_so_far, config.portfolio.dd_threshold, config.portfolio.dd_min_scale
    )

    target = optimizer.optimize(
        selected,
        covariance,
        betas,
        positions,
        optimizer.Constraints(
            gross_leverage=config.portfolio.gross_leverage,
            max_weight=config.portfolio.max_weight,
            beta_tolerance=config.portfolio.beta_tolerance,
            vol_target=config.portfolio.vol_target * scale,
            industry_tolerance=config.portfolio.industry_tolerance,
        ),
        optimizer.Objective(
            turnover_penalty=config.portfolio.turnover_penalty,
            solver=config.portfolio.solver,
            information_coefficient=config.portfolio.information_coefficient,
            horizon_days=config.label.horizon_days,
            risk_in_objective=config.portfolio.risk_in_objective,
        ),
        industries=panel.industries if config.portfolio.industry_tolerance >= 0 else None,
    )

    # Transfer coefficient: how much of the forecast survived implementation.
    alpha = optimizer.expected_returns(
        selected, covariance.loc[names, names],
        config.portfolio.information_coefficient, config.label.horizon_days,
    )
    target.attrs["transfer_coefficient"] = optimizer.transfer_coefficient(
        alpha, target.reindex(names).fillna(0.0).to_numpy(),
        covariance.loc[names, names].fillna(0.0).to_numpy() + np.eye(len(names)) * 1e-6,
    )
    target.attrs["n_candidates"] = len(names)
    return target
