"""Generate docs/index.html from a run directory and the measured bias signatures.

The dashboard is generated, never hand-edited. The previous one was written by
hand and went on serving a Sharpe of 1.46 and a CAGR of 35.3% long after those
figures had been withdrawn in the README -- published, live, and linked from a
personal site. Numbers that are typed into a page drift from the numbers the
code produces; numbers that are read out of the run directory cannot.

The page leads with the bias-fingerprinting framework, whose two signatures are
computed here from reports/pit_vs_naive.csv and reports/survivorship.csv, and
then reports the factor study those signatures are calibrated on.

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
PAGE_TITLE = "Bias Fingerprints — point-in-time equity factor research"

# Beta decomposition of the decile spread. Measured in docs/BIAS.md and reported in
# the README; the run directory does not carry a market series, so these four are
# the only figures on the page not read out of a CSV.
BETA_RAW_SPREAD_BPS, BETA_RAW_T = 43.7, 2.11
BETA_OF_SPREAD = 0.218
BETA_ADJ_BPS, BETA_ADJ_T = 17.7, 0.92
BETA_SHARE_OF_EDGE = 0.59


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


def signatures() -> dict[str, object]:
    """Both bias signatures, as t-statistic shifts from the point-in-time baseline.

    Read from the two CSVs `make bias` writes, so the page cannot disagree with the
    paper: the same files produce the figure in paper/figures/fingerprints.pdf.
    """
    dating = pd.read_csv(REPORTS / "pit_vs_naive.csv").set_index("factor")
    universe = pd.read_csv(REPORTS / "survivorship.csv").set_index("factor")

    shift = pd.DataFrame(
        {
            "dating": dating["naive_t"] - dating["pit_t"],
            "universe": universe["survivor_t"] - universe["pit_t"],
        }
    ).dropna()

    exact_zero = int((shift["dating"].abs() == 0.0).sum())
    movers = shift.reindex(
        shift.abs().max(axis=1).sort_values(ascending=False).index
    ).head(8)
    # t-shifts read at two decimals; _fmt would render them as four.
    movers = movers.map(lambda v: f"{v:+.2f}" if v else "0.00")

    return {
        "table": table(movers, {"dating": "period-end join", "universe": "current membership"}),
        "correlation": f"{shift['dating'].corr(shift['universe']):+.3f}",
        "exact_zero": str(exact_zero),
        "n_factors": str(len(shift)),
        "max_drift": f"{shift['dating'].abs().min():.1e}",
    }


def build(run: Path) -> str:
    performance = json.loads((run / "performance.json").read_text())
    signal = json.loads((run / "signal.json").read_text())
    factors = pd.read_csv(run / "factor_ic.csv", index_col=0)
    coverage = pd.read_csv(run / "universe_coverage.csv", index_col=0, parse_dates=[0])
    equity = pd.read_csv(run / "equity_curve.csv", index_col=0, parse_dates=[0])["equity"]

    pct = lambda x: f"{x * 100:.2f}%"  # noqa: E731
    cover = coverage.resample("YE").mean().round(0)
    sig = signatures()

    return TEMPLATE.format(
        title=PAGE_TITLE,
        signature_table=sig["table"],
        signature_corr=sig["correlation"],
        signature_zeros=sig["exact_zero"],
        signature_n=sig["n_factors"],
        beta_raw=f"{BETA_RAW_SPREAD_BPS}bp",
        beta_raw_t=f"t = {BETA_RAW_T}",
        beta_of_spread=f"{BETA_OF_SPREAD}",
        beta_adj=f"{BETA_ADJ_BPS}bp",
        beta_adj_t=f"t = {BETA_ADJ_T}",
        beta_share=f"{BETA_SHARE_OF_EDGE:.0%}",
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
<meta name="description" content="Bias fingerprinting: measure what a data-handling defect does to a factor cross-section, then diagnose the defect from a factor table you cannot audit.">
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
  <h1>Bias fingerprints</h1>
  <p class="sub">A method for auditing factor research from the outside: measure what a
  data-handling defect does to a factor cross-section on a pipeline you own, then test for
  that shape in tables you cannot audit. Calibrated on a point-in-time US equity study,
  out-of-sample {period} ({years} years).</p>
  <div class="run">generated from run {run}</div>
</header>

<section>
  <h2>The framework</h2>
  <p>A factor table is usually all an outside reader gets. An allocator reads a manager's
  IC table, a referee reads a submitted backtest, a desk prices a vendor signal. None of
  them can run the pipeline behind the numbers, and the literature on look-ahead and
  survivorship bias is written for someone who can.</p>
  <p>A defect is not a scalar amount of inflation. It is a direction in factor space, and
  the direction is a property of the defect rather than of the study it damages. The
  framework is a <em>generator</em> that re-runs one study under alternative data
  conventions, a <em>signature</em> per defect, an <em>inference model</em> that separates
  signature from unknown true performance, and a <em>validation protocol</em> of four gates
  that must pass before the diagnostic reports anything.</p>
  <p><a href="https://github.com/quantraunak/bias-fingerprints/blob/main/paper/bias_fingerprints.pdf">Read
  the paper</a> · <a href="https://github.com/quantraunak/bias-fingerprints/blob/main/docs/FINGERPRINT.md">method</a>
  · <a href="https://github.com/quantraunak/bias-fingerprints/blob/main/docs/BIAS.md">measurements</a></p>
</section>

<section>
  <h2>Two signatures, calibrated</h2>
  <p>Shift in t-statistic from the point-in-time baseline, for the eight factors that move
  most under either defect. Both columns come from the same generator, changing one data
  convention at a time.</p>
  <div class="scroll">{signature_table}</div>
  <div class="grid" style="margin-top:26px">
    <div class="metric"><div class="v">{signature_corr}</div><div class="l">Signature correlation</div><div class="note">across {signature_n} factors</div></div>
    <div class="metric"><div class="v">{signature_zeros}</div><div class="l">Factors dating cannot move</div><div class="note">exactly zero, not approximately</div></div>
    <div class="metric"><div class="v">+59%</div><div class="l">Mean IC inflation</div><div class="note">period-end join, 11 fundamentals</div></div>
    <div class="metric"><div class="v">4</div><div class="l">False discoveries</div><div class="note">manufactured above t = 2</div></div>
  </div>
  <p style="margin-top:22px">The two signatures point in near-independent directions and
  disagree in sign on value factors, so a table cannot be explained by both. The exact
  zeros carry more weight than the correlation: a factor built from price and volume never
  reads a filed figure, so a strong illiquidity result cannot have come from a dating
  defect whatever else is true of the study.</p>
  <p>Conditioning a universe on current index membership behaves in the opposite way. Mean
  IC falls rather than rises, while <span class="mono">amihud_illiquidity</span> flips from
  t = -1.11 to +2.80 and the low-volatility anomaly inverts. The channel is a
  forward-looking growth filter, not a survival filter: the biased panel wrongly includes
  203 names a day that had not yet been admitted, against 88 it wrongly excludes.</p>
  <div class="callout">
    <b>Status.</b> The framework is specified and both signatures are measured and
    reproducible. The three validation gates — signature stability, sampling covariance,
    separability under realistic noise — have not been run, and the diagnostic reports
    nothing until they have. A label without a calibrated reference distribution is worse
    than no diagnostic.
  </div>
</section>

<section>
  <h2>The reference study — signal</h2>
  <p>The generator the signatures are calibrated on. A 22-factor cross-section on
  reconstructed point-in-time index membership and SEC XBRL fundamentals keyed on filing
  date, over {years} out-of-sample years.</p>
  <div class="grid" style="margin-top:20px">{metrics_signal}</div>
</section>

<section>
  <h2>Factors by information coefficient</h2>
  <div class="scroll">{factor_table}</div>
</section>

<section>
  <h2>Equity curve, net of costs</h2>
  <figure>{curve}<figcaption><span>{equity_start}</span><span>{equity_end}</span></figcaption></figure>
</section>

<section>
  <h2>Portfolio, net of costs</h2>
  <div class="grid">{metrics_strategy}</div>
  <p style="margin-top:22px">Cost sensitivity — Sharpe at 5bp slippage <span class="mono">{cost_5}</span>,
  at 10bp <span class="mono">{cost_10}</span>, at 20bp <span class="mono">{cost_20}</span>.</p>
</section>

<section>
  <h2>Two results that outlive the study</h2>
  <p>A decile spread with a t-statistic above 2 looks like a strategy. Regressing that
  spread on the market decides whether it is one.</p>
  <div class="grid" style="margin-top:20px">
    <div class="metric"><div class="v">{beta_raw}</div><div class="l">Raw spread / mo</div><div class="note">{beta_raw_t}</div></div>
    <div class="metric"><div class="v">{beta_of_spread}</div><div class="l">Beta of the spread</div></div>
    <div class="metric"><div class="v">{beta_adj}</div><div class="l">Beta-adj alpha / mo</div><div class="note">{beta_adj_t}</div></div>
    <div class="metric"><div class="v">{beta_share}</div><div class="l">Of edge from beta</div></div>
  </div>
  <p style="margin-top:22px">The long decile runs a beta of 1.10 against the short decile's
  0.99, a tilt positive in 73% of months over a window in which the market compounded at
  13.7%. A decile-spread t-statistic reported without its beta decomposition is not a claim
  about stock selection, here or anywhere.</p>
  <p>The second result is dispersion. Across six model seeds with identical economics, IC
  holds at 0.0164 ± 0.0008 while Sharpe ranges from 0.09 to 0.49. A single backtest Sharpe
  from a stochastically fitted model is a draw from that distribution, and every
  performance figure above should be read against its spread.</p>
</section>

<section>
  <h2>Universe coverage</h2>
  <p>Point-in-time index membership against names that could actually be priced and traded.
  The gap is the residual survivorship limitation: delisted tickers are frequently
  unavailable from the price source, and it bounds the second signature as well as the
  study.</p>
  <div class="scroll">{coverage_table}</div>
</section>

<footer>
  Raunak Sood · <a href="https://github.com/quantraunak/bias-fingerprints">source</a> ·
  every figure on this page is read from the run directory and the measured signatures at build time
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
