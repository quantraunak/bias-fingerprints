from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.performance_charts import save_performance_charts


def generate_tear_sheet(
    returns: pd.Series,
    output_dir: Path,
    benchmark: pd.Series | None = None,
    benchmark_label: str = "Benchmark",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    import quantstats as qs

    qs.reports.html(
        returns,
        benchmark=benchmark,
        output=str(output_dir / "tear_sheet.html"),
        title="Multi-Factor Long/Short Tear Sheet",
    )

    save_performance_charts(
        returns, output_dir, benchmark_returns=benchmark, benchmark_label=benchmark_label
    )
