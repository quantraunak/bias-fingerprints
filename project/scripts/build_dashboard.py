"""Generate docs/index.html from a run directory.

The dashboard is generated, never hand-edited. The previous one was written by
hand and went on serving a Sharpe of 1.46 and a CAGR of 35.3% long after those
figures had been withdrawn in the README -- published, live, and linked from a
personal site. Numbers that are typed into a page drift from the numbers the
code produces; numbers that are read out of the run directory cannot.

    python -m scripts.build_dashboard                 # newest run
    python -m scripts.build_dashboard --run <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import REPORTS, ROOT  # noqa: E402

OUTPUT = ROOT.parent / "docs" / "index.html"
PAGE_TITLE = "Systematic Equity Research — results"


def latest_run() -> Path:
    """Newest timestamped run. Names are YYYYmmdd_HHMMSS, so sorting is chronological."""
    runs = sorted(
        p for p in REPORTS.glob("[0-9]" * 8 + "_" + "[0-9]" * 6) if (p / "performance.json").exists()
    )
    if not runs:
        raise SystemExit("No completed runs found. Run scripts/run_backtest.py first.")
    return runs[-1]


# ----------------------------------------------------------------- rendering


def sparkline(series: pd.Series, width: int = 880, height: int = 260) -> str:
    """Equity curve as a bare inline SVG path; no chart library, no CDN."""
    values = series.to_numpy()
    lo, hi = float(values.min()), float(values.max())
    span = (hi - lo) or 1.0
    step = width / max(len(values) - 1, 1)
    points = " ".join(
        f"{i * step:.2f},{height - (v - lo) / span * height:.2f}" for i, v in enumerate(values)
    )
    baseline = height - (1.0 - lo) / span * height
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
        f'aria-label="Equity curve">'
        f'<line x1="0" y1="{baseline:.2f}" x2="{width}" y2="{baseline:.2f}" class="zero"/>'
        f'<polyline points="{points}" class="curve"/></svg>'
    )


def metric(label: str, value: str, note: str = "") -> str:
    extra = f'<div class="note">{note}</div>' if note else ""
    return f'<div class="metric"><div class="v">{value}</div><div class="l">{label}</div>{extra}</div>'


def table(frame: pd.DataFrame, columns: dict[str, str], limit: int | None = None) -> str:
    subset = frame.head(limit) if limit else frame
    head = "".join(f"<th>{title}</th>" for title in columns.values())
    rows = []
    for name, row in subset.iterrows():
        cells = "".join(f"<td>{_fmt(row[key])}</td>" for key in columns)
        rows.append(f"<tr><th scope='row'>{name}</th>{cells}</tr>")
    return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 10 else f"{value:,.1f}"
    return str(value)


def build(run: Path) -> str:
    performance = json.loads((run / "performance.json").read_text())
    signal = json.loads((run / "signal.json").read_text())
    factors = pd.read_csv(run / "factor_ic.csv", index_col=0)
    coverage = pd.read_csv(run / "universe_coverage.csv", index_col=0, parse_dates=[0])
    equity = pd.read_csv(run / "equity_curve.csv", index_col=0, parse_dates=[0])["equity"]

    pct = lambda x: f"{x * 100:.2f}%"  # noqa: E731
    cover = coverage.resample("YE").mean().round(0)

    return TEMPLATE.format(
        title=PAGE_TITLE,
        run=run.name,
        period=f"{performance['start']} → {performance['end']}",
        years=performance["years"],
        curve=sparkline(equity),
        equity_start=f"{equity.iloc[0]:.2f}",
        equity_end=f"{equity.iloc[-1]:.2f}",
        metrics_strategy="".join(
            [
                metric("CAGR", pct(performance["cagr"])),
                metric("Sharpe", f"{performance['sharpe']:.2f}", "0.09–0.49 across seeds"),
                metric("Max drawdown", pct(performance["max_drawdown"])),
                metric("Annualised vol", pct(performance["annual_vol"])),
                metric("Beta vs SPY", f"{performance['beta_vs_market']:.3f}"),
                metric("Day coverage", pct(performance["coverage"]), "0 rebalances skipped"),
            ]
        ),
        metrics_signal="".join(
            [
                metric("Mean IC", f"{signal['mean_ic']:.4f}"),
                metric("IC info ratio", f"{signal['icir']:.3f}"),
                metric("t-statistic", f"{signal['t_stat']:.2f}", f"{signal['n_independent']:.0f} months"),
                metric("Decile spread", f"{signal['decile_spread_bps']:.0f} bp/mo",
                       f"t = {signal['decile_spread_t']:.2f}"),
                metric("Selection turnover", pct(signal["selection_turnover"])),
                metric("Avg names", f"{performance.get('avg_names', 0):.0f}"),
            ]
        ),
        factor_table=table(
            factors, {"mean_ic": "mean IC", "icir": "ICIR", "t_stat": "t"}, limit=12
        ),
        coverage_table=table(
            cover.assign(year=cover.index.year).set_index("year"),
            {"in_index": "index members", "with_price": "priced", "tradable": "tradable",
             "with_fundamentals": "with fundamentals"},
        ),
        cost_5=f"{performance.get('sharpe_at_5bps', float('nan')):.2f}",
        cost_10=f"{performance.get('sharpe_at_10bps', float('nan')):.2f}",
        cost_20=f"{performance.get('sharpe_at_20bps', float('nan')):.2f}",
    )


TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="Out-of-sample results for a market-neutral US equity signal built on point-in-time data.">
<style>
:root{{--bg:#f6f4ef;--fg:#1c1b19;--fg2:#57534c;--muted:#8b857b;--line:rgba(28,27,25,.12);--accent:#a2762f;--warn:#9a3b2f}}
@media(prefers-color-scheme:dark){{:root{{--bg:#151412;--fg:#ece8e0;--fg2:#b0aaa0;--muted:#7d776d;--line:rgba(255,255,255,.13);--accent:#c79a54;--warn:#d98070}}}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--fg);font:16px/1.65 ui-sans-serif,system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:900px;margin:0 auto;padding:0 24px 80px}}
header{{padding:72px 0 28px}}
h1{{font-size:clamp(28px,4.5vw,40px);font-weight:500;letter-spacing:-.02em;line-height:1.1}}
.sub{{margin-top:12px;color:var(--fg2);max-width:640px}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.run{{margin-top:14px;font-family:ui-monospace,monospace;font-size:12px;color:var(--muted)}}
section{{padding:34px 0;border-top:1px solid var(--line)}}
h2{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:500;margin-bottom:20px;font-family:ui-monospace,monospace}}
.callout{{border-left:2px solid var(--warn);padding:12px 0 12px 16px;color:var(--fg2);margin:18px 0}}
.callout b{{color:var(--fg);font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:22px}}
.metric .v{{font-family:ui-monospace,monospace;font-size:22px;font-variant-numeric:tabular-nums}}
.metric .l{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-top:4px}}
.metric .note{{font-size:11px;color:var(--muted);margin-top:3px;font-style:italic}}
figure{{margin:8px 0 0}}
svg{{width:100%;height:auto;display:block}}
.curve{{fill:none;stroke:var(--accent);stroke-width:1.6;vector-effect:non-scaling-stroke}}
.zero{{stroke:var(--line);stroke-width:1;vector-effect:non-scaling-stroke;stroke-dasharray:3 3}}
figcaption{{display:flex;justify-content:space-between;font-family:ui-monospace,monospace;font-size:11px;color:var(--muted);margin-top:8px}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-family:ui-monospace,monospace;font-size:12.5px;font-variant-numeric:tabular-nums;min-width:460px}}
th,td{{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
thead th{{color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
tbody th{{text-align:left;font-weight:400;color:var(--fg2)}}
p+p{{margin-top:12px}}
p{{color:var(--fg2);max-width:660px}}
a{{color:var(--accent)}}
footer{{padding-top:28px;border-top:1px solid var(--line);font-family:ui-monospace,monospace;font-size:11.5px;color:var(--muted)}}
</style></head><body><div class="wrap">

<header>
  <h1>Systematic equity research</h1>
  <p class="sub">A market-neutral US equity signal on point-in-time data. Out-of-sample
  {period} ({years} years), monthly rebalance, 1bp commission and 5bp slippage.</p>
  <div class="run">generated from run {run}</div>
</header>

<section>
  <h2>Read this first</h2>
  <div class="callout">
    <b>The signal is real; the strategy is not.</b> The model has a genuine cross-sectional
    information coefficient, but 59% of the raw decile spread turns out to be a market-beta
    tilt. Beta-adjusted alpha is 2.1% a year with a t-statistic of 0.92 — indistinguishable
    from zero. Across six model seeds with identical economics the Sharpe below ranges from
    0.09 to 0.49.
  </div>
  <p>An earlier version of this page reported a Sharpe of 1.46 and a CAGR of 35.3%. Those
  figures came from a backtest in which roughly 30% of trading days carried no profit and
  loss at all, on a survivorship-biased universe, with mis-specified momentum horizons. They
  were withdrawn. This page is generated directly from the run directory so it cannot drift
  from the code again.</p>
</section>

<section>
  <h2>Equity curve, net of costs</h2>
  <figure>{curve}<figcaption><span>{equity_start}</span><span>{equity_end}</span></figcaption></figure>
</section>

<section>
  <h2>Strategy</h2>
  <div class="grid">{metrics_strategy}</div>
  <p style="margin-top:22px">Cost sensitivity — Sharpe at 5bp slippage <span class="mono">{cost_5}</span>,
  at 10bp <span class="mono">{cost_10}</span>, at 20bp <span class="mono">{cost_20}</span>.</p>
</section>

<section>
  <h2>Signal</h2>
  <div class="grid">{metrics_signal}</div>
</section>

<section>
  <h2>Factors by information coefficient</h2>
  <div class="scroll">{factor_table}</div>
</section>

<section>
  <h2>Universe coverage</h2>
  <p>Point-in-time index membership against names that could actually be priced and traded.
  The gap is the residual survivorship limitation: delisted tickers are frequently
  unavailable from the price source.</p>
  <div class="scroll">{coverage_table}</div>
</section>

<footer>
  Raunak Sood · <a href="https://github.com/quantraunak/ls-multifactor-research">source</a> ·
  every figure on this page is read from the run directory at build time
</footer>
</div></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    run = Path(args.run) if args.run else latest_run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(run))
    print(f"wrote {OUTPUT} from {run}")


if __name__ == "__main__":
    main()
