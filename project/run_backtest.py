from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import summarize
from src.data.fetch_prices import FetchConfig, get_price_data
from src.data.universe import load_universe
from src.features.factors import compute_factors
from src.reporting.factor_ic import compute_factor_ic, save_factor_ic
from src.reporting.tear_sheet import generate_tear_sheet


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    np.random.seed(7)

    config = load_config(Path(config_path))

    data_raw = PROJECT_ROOT / "data" / "raw"
    reports_root = PROJECT_ROOT / "reports" / "latest"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = reports_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    universe_path = PROJECT_ROOT / "data" / "raw" / "universe.csv"
    tickers = load_universe(universe_path, fallback=config["universe"]["fallback"])
    if "SPY" not in tickers:
        tickers.append("SPY")

    data_cfg = config["data"]
    fetch_cfg = FetchConfig(
        source=data_cfg.get("source", "auto"),
        start=data_cfg["start"],
        end=data_cfg["end"],
        force_refresh=data_cfg.get("force_refresh", False),
        min_history_days=data_cfg.get("min_history_days", 252),
        min_tickers=data_cfg.get("min_tickers", 20),
        retries=data_cfg.get("retries", 3),
    )

    price_data = get_price_data(
        tickers=tickers,
        cfg=fetch_cfg,
        cache_dir=data_raw,
    )

    risk_cfg = config.get("risk", {})

    bt_cfg = BacktestConfig(
        start=config["data"]["start"],
        end=config["data"]["end"],
        rebalance_freq=config["backtest"]["rebalance_freq"],
        train_lookback_months=config["backtest"]["train_lookback_months"],
        horizon_days=config["labels"]["horizon_days"],
        long_short_quantile=config["portfolio"]["long_short_quantile"],
        beta_window=config["portfolio"]["beta_window"],
        cov_window=config["portfolio"]["cov_window"],
        min_train_samples=config["model"]["min_train_samples"],
        risk_aversion=config["portfolio"]["risk_aversion"],
        turnover_penalty=config["portfolio"]["turnover_penalty"],
        commission_bps=config["costs"]["commission_bps"],
        slippage_bps=config["costs"]["slippage_bps"],
        max_weight=config["portfolio"]["max_weight"],
        gross_leverage=config["portfolio"]["gross_leverage"],
        beta_tolerance=config["portfolio"]["beta_tolerance"],
        solver=config["portfolio"].get("solver"),
        normalize=config["features"]["normalize"],
        winsorize_limits=tuple(config["features"]["winsorize_limits"]),
        model_config=config["model"],
        cov_method=risk_cfg.get("cov_method", "ledoit_wolf"),
        vol_target=risk_cfg.get("vol_target"),
        vol_lookback=risk_cfg.get("vol_lookback", 63),
        vol_ewma_span=risk_cfg.get("vol_ewma_span"),
        min_leverage=risk_cfg.get("min_leverage", 0.5),
        max_leverage=risk_cfg.get("max_leverage", 2.0),
        dd_threshold=risk_cfg.get("dd_threshold"),
        dd_min_scale=risk_cfg.get("dd_min_scale", 0.35),
    )

    results = run_backtest(price_data.prices, price_data.volumes, bt_cfg)
    daily_returns = results["daily_returns"]
    holdings = results["holdings"]

    daily_returns.to_csv(run_dir / "equity_curve.csv", index=True)
    holdings.to_csv(run_dir / "holdings.csv", index=True)

    spy_returns = price_data.prices["SPY"].pct_change(fill_method=None).dropna()
    summary = summarize(daily_returns, results["turnovers"], holdings, market_returns=spy_returns)
    (run_dir / "performance_summary.json").write_text(json.dumps(summary, indent=2))

    # Feature importance (LightGBM)
    fi = results.get("feature_importances")
    if fi is not None and not fi.empty:
        fi.to_csv(run_dir / "feature_importances.csv", index=False)

    # Leverage history (vol targeting)
    lev = results.get("leverage_history")
    if lev is not None and not lev.empty:
        lev.to_csv(run_dir / "leverage_history.csv")

    generate_tear_sheet(daily_returns, run_dir, benchmark=spy_returns)

    # Factor IC analysis
    prices_slice = price_data.prices.loc[config["data"]["start"]:config["data"]["end"]]
    vol_slice = price_data.volumes.loc[prices_slice.index[0]:prices_slice.index[-1]]
    factors = compute_factors(prices_slice, vol_slice)
    ic_df, ic_summary = compute_factor_ic(factors, prices_slice)
    save_factor_ic(ic_df, ic_summary, run_dir)

    # Save config snapshot for reproducibility
    (run_dir / "config_snapshot.yaml").write_text(yaml.dump(config, default_flow_style=False))

    print(f"Run complete. Outputs in: {run_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
