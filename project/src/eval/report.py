"""Write one self-describing folder per run.

Every artefact needed to argue with the result goes in the same place: the
config that produced it, the data coverage it ran on, the signal diagnostics,
the performance statistics, and the series themselves. A run that cannot be
reproduced from its own folder is not a result.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import REPORTS, Config


def new_run_directory(root: Path = REPORTS) -> Path:
    path = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write(
    directory: Path,
    config: Config,
    panel_coverage: pd.DataFrame,
    factor_table: pd.DataFrame,
    signal: dict,
    quantiles: pd.DataFrame,
    performance: dict,
    result,
    importances: pd.DataFrame,
) -> None:
    _json(directory / "config.json", asdict(config))
    _json(directory / "performance.json", performance)
    _json(directory / "signal.json", signal)

    panel_coverage.to_csv(directory / "universe_coverage.csv")
    factor_table.to_csv(directory / "factor_ic.csv")
    quantiles.to_csv(directory / "quantile_returns.csv")
    result.net_returns.rename("net").to_csv(directory / "returns_net.csv")
    result.gross_returns.rename("gross").to_csv(directory / "returns_gross.csv")
    result.equity.rename("equity").to_csv(directory / "equity_curve.csv")
    result.holdings.to_csv(directory / "holdings.csv")
    result.rebalances.to_csv(directory / "rebalances.csv")
    if not importances.empty:
        importances.to_csv(directory / "feature_importances.csv")
    _json(directory / "skipped_rebalances.json", result.skipped)


def summary_text(performance: dict, signal: dict, factor_table: pd.DataFrame, top: int = 8) -> str:
    lines = [
        "SIGNAL",
        f"  out-of-sample IC        {signal.get('mean_ic')}  (ICIR {signal.get('icir')}, "
        f"annualised {signal.get('icir_annual')})",
        f"  t-statistic             {signal.get('t_stat')}  on {signal.get('n_independent')} "
        f"independent observations",
        f"  dates positive          {signal.get('pct_positive')}",
        "",
        "STRATEGY",
        f"  period                  {performance.get('start')} -> {performance.get('end')}  "
        f"({performance.get('years')}y)",
        f"  day coverage            {performance.get('coverage')}  "
        f"({performance.get('n_days')} of {performance.get('business_days_in_span')} business days)",
        f"  CAGR                    {performance.get('cagr')}",
        f"  Sharpe                  {performance.get('sharpe')}",
        f"  Sortino                 {performance.get('sortino')}",
        f"  max drawdown            {performance.get('max_drawdown')}",
        f"  annualised vol          {performance.get('annual_vol')}",
        f"  beta vs market          {performance.get('beta_vs_market')}",
        f"  annualised alpha        {performance.get('alpha_annual')}",
        "",
        f"TOP FACTORS BY ICIR (of {len(factor_table)})",
    ]
    for name, row in factor_table.head(top).iterrows():
        lines.append(f"  {name:<22}{row['mean_ic']:>9.5f} IC   {row['icir']:>7.4f} ICIR   {row['t_stat']:>6.2f} t")
    return "\n".join(lines)


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))
