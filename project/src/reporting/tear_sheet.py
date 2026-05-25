from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.performance_charts import save_performance_charts


def generate_tear_sheet(
    returns: pd.Series,
    output_dir: Path,
    benchmark: pd.Series | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import quantstats as qs

        qs.reports.html(
            returns,
            benchmark=benchmark,
            output=str(output_dir / "tear_sheet.html"),
            title="Multi-Factor Long/Short Tear Sheet",
        )
    except Exception:
        html_path = output_dir / "tear_sheet.html"
        html_path.write_text(
            "<html><body><h1>Tear Sheet</h1><p>Quantstats not available.</p></body></html>"
        )

    save_performance_charts(returns, output_dir, benchmark_returns=benchmark)
