from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import summarize
from src.data.pipeline import load_market_data
from src.features.factors import compute_factors
from src.reporting.factor_ic import compute_factor_ic, save_factor_ic
from src.reporting.tear_sheet import generate_tear_sheet


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _write_data_manifest(run_dir: Path, market: dict) -> None:
    uni = market["universe"]
    fetch = market["fetch"]
    manifest = {
        "universe": {
            "source": uni.source,
            "path": uni.path,
            "is_point_in_time": uni.is_point_in_time,
            "n_tickers": len(uni.tickers),
            "warnings": list(uni.warnings),
        },
        "prices": {
            "source": fetch.price_source,
            "n_requested": fetch.n_requested,
            "n_loaded": fetch.n_loaded,
            "n_failed": fetch.n_failed,
            "n_dropped_short_history": fetch.n_dropped_short_history,
        },
        "panel": market["panel_stats"],
    }
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))


def main(config_path: str):
    np.random.seed(7)
    config = load_config(Path(config_path))

    reports_root = PROJECT_ROOT / "reports" / "latest"
    run_dir = reports_root / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    market = load_market_data(PROJECT_ROOT, config)
    _write_data_manifest(run_dir, market)

    for w in market["universe"].warnings:
        print(f"[universe] {w}")
    print(
        f"[data] price_source={market['fetch'].price_source} "
        f"loaded={market['fetch'].n_loaded}/{market['fetch'].n_requested} "
        f"panel={market['panel_stats']['n_tickers']} tickers "
        f"{market['panel_stats']['date_start']}..{market['panel_stats']['date_end']}"
    )

    risk_cfg = config.get("risk", {})
    costs_cfg = config.get("costs", {})
    research_cfg = config.get("research", {})
    bench = config["universe"].get("benchmark", "SPY")

    bt_cfg = BacktestConfig(
        start=config["data"]["start"],
        end=config["data"]["end"],
        oos_start=research_cfg.get("oos_start"),
        rebalance_freq=config["backtest"]["rebalance_freq"],
        train_lookback_months=config["backtest"]["train_lookback_months"],
        horizon_days=config["labels"]["horizon_days"],
        long_short_quantile=config["portfolio"]["long_short_quantile"],
        beta_window=config["portfolio"]["beta_window"],
        cov_window=config["portfolio"]["cov_window"],
        min_train_samples=config["model"]["min_train_samples"],
        risk_aversion=config["portfolio"]["risk_aversion"],
        turnover_penalty=config["portfolio"]["turnover_penalty"],
        commission_bps=costs_cfg["commission_bps"],
        slippage_bps=costs_cfg["slippage_bps"],
        max_weight=config["portfolio"]["max_weight"],
        gross_leverage=config["portfolio"]["gross_leverage"],
        beta_tolerance=config["portfolio"]["beta_tolerance"],
        solver=config["portfolio"].get("solver"),
        normalize=config["features"]["normalize"],
        winsorize_limits=tuple(config["features"]["winsorize_limits"]),
        sector_neutral=config["features"].get("sector_neutral", True),
        benchmark=bench,
        model_config=config["model"],
        cov_method=risk_cfg.get("cov_method", "ledoit_wolf"),
        vol_target=risk_cfg.get("vol_target"),
        vol_lookback=risk_cfg.get("vol_lookback", 126),
        vol_ewma_span=risk_cfg.get("vol_ewma_span"),
        min_leverage=risk_cfg.get("min_leverage", 0.4),
        max_leverage=risk_cfg.get("max_leverage", 2.0),
        dd_threshold=risk_cfg.get("dd_threshold"),
        dd_min_scale=risk_cfg.get("dd_min_scale", 0.35),
    )

    results = run_backtest(
        market["prices"],
        market["volumes"],
        bt_cfg,
        sector_map=market["sectors"],
        membership=market["universe"].membership,
    )

    daily_returns = results["daily_returns"]
    daily_returns.to_csv(run_dir / "equity_curve.csv", index=True)
    results["holdings"].to_csv(run_dir / "holdings.csv", index=True)

    spy_returns = market["prices"][bench].pct_change(fill_method=None).dropna()
    spy_returns = spy_returns.reindex(daily_returns.index).fillna(0.0)

    summary = summarize(
        daily_returns,
        results["turnovers"],
        results["holdings"],
        market_returns=spy_returns,
        gross_returns=results.get("gross_returns"),
        rebalance_costs=results.get("rebalance_costs"),
        commission_bps=costs_cfg["commission_bps"],
        base_slippage_bps=costs_cfg["slippage_bps"],
        cost_stress_bps=costs_cfg.get("stress_slippage_bps"),
    )
    summary["oos_start"] = research_cfg.get("oos_start")
    (run_dir / "performance_summary.json").write_text(json.dumps(summary, indent=2))

    skipped = results.get("skipped_rebalances", [])
    if skipped:
        (run_dir / "skipped_rebalances.json").write_text(json.dumps(skipped, indent=2, default=str))

    fi = results.get("feature_importances")
    if fi is not None and not fi.empty:
        fi.to_csv(run_dir / "feature_importances.csv", index=False)

    lev = results.get("leverage_history")
    if lev is not None and not lev.empty:
        lev.to_csv(run_dir / "leverage_history.csv")

    generate_tear_sheet(daily_returns, run_dir, benchmark=spy_returns, benchmark_label=bench)

    prices_slice = market["prices"].loc[config["data"]["start"] : config["data"]["end"]]
    vol_slice = market["volumes"].loc[prices_slice.index[0] : prices_slice.index[-1]]
    ic_df, ic_summary = compute_factor_ic(
        compute_factors(prices_slice, vol_slice, benchmark=bench),
        prices_slice,
        horizon_days=config["labels"]["horizon_days"],
    )
    save_factor_ic(ic_df, ic_summary, run_dir)

    (run_dir / "config_snapshot.yaml").write_text(yaml.dump(config, default_flow_style=False))

    assets_dir = PROJECT_ROOT.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for chart in ("equity_curve.png", "drawdown.png"):
        src = run_dir / chart
        if src.exists():
            shutil.copy2(src, assets_dir / chart)

    print(f"Run complete. Outputs in: {run_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
