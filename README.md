# Systematic Equity Research

A market-neutral US equity signal built on point-in-time data, with the evaluation
run before the portfolio and the negative results kept in.

**Headline: the signal is real and the strategy is not.** The model has a genuine
cross-sectional information coefficient of 0.0165 (t = 2.00) and a decile spread of
68bp a month (t = 3.08). Once the market-beta tilt inside that spread is removed and
realistic costs are charged, the beta-adjusted alpha is 2.1% a year with a
t-statistic of **0.92** — indistinguishable from zero. The backtest posts a Sharpe
of 0.29, but across six model seeds with identical economics that number ranges
from 0.09 to 0.49 — the signal is stable, the Sharpe is not. See
[Robustness](#robustness).

The interesting content is not the Sharpe. It is the decomposition that explains
where a plausible-looking edge actually comes from, and the six defects in the
previous version of this repository that manufactured one.

---

## What was wrong before

The previous pipeline reported a Sharpe of 0.53 and, before that, 1.46. Both were
artefacts. Each defect below produced output that looked reasonable, which is the
dangerous kind.

**1. Roughly 30% of the backtest had no P&L.** Rebalance dates came from
`prices.resample("ME")`, which returns *calendar* month-ends. When the 31st fell on
a weekend the date was not in the price index, the cross-section came back empty,
and the loop `continue`d **before** accruing that month's returns. 55 of 180
month-ends were not trading days; 578 of 1,971 out-of-sample days were missing
entirely. The equity curve teleported across the gaps.

**2. The headline CAGR was inflated by the same bug.** Annualising with
`252 / len(returns)` treats 1,393 observations as 5.5 years when 7.8 years had
elapsed. The reported 5.9% CAGR was 4.1%.

**3. Every momentum horizon was wrong by ~1.45×.** The lag helper subtracted
*calendar* days and called them trading days, so 252 "days" was 173 trading days.
"12–1 momentum" measured 8.2 months, 6–1 measured 4.1, and the 5-day reversal
measured 3.

**4. The label was 28% market.** The model regressed raw 21-day forward returns
with an L2 loss. Measured on this panel, 28% of that variable's variance is the
common date move — identical for every name, so it cannot affect the ranking the
strategy trades, but the loss still spent a quarter of its budget on it. The file
was named `gbm_ranker.py` and contained no ranking objective.

**5. The largest feature was a size dummy with the worst IC.** `liquidity` was
`log1p(20-day mean dollar volume)` — a *level*, with monthly cross-sectional rank
autocorrelation of 0.96. It consumed the most model splits and had the most
negative IC in the book (−0.0275, IR −0.274, positive on 38% of days).

**6. There was no signal at all.** Every factor's IC information ratio fell between
−0.27 and +0.09. Three of the ten "factors" were the same volatility measure, and
`quality_proxy` was a 252-day Sharpe ratio, not quality. Sharpe 0.53 was the honest
output of a feature set with no information, measured on a calendar with holes.

Guard tests for every one of these live in [`project/tests/test_no_lookahead.py`](project/tests/test_no_lookahead.py).

---

## Data

The largest change is the data, because that is where the ceiling was.

| | before | now |
|---|---|---|
| Universe | 429 *current* S&P 500 members | 727 names, point-in-time membership |
| Survivorship | every delisted name invisible | names enter and leave on their real dates |
| Fundamentals | none | SEC EDGAR XBRL, 648 issuers, keyed on **filing date** |
| Industry | yfinance GICS, survivors only | SEC SIC → Fama-French 12, full coverage |
| Span | 2010–2024 | 2004–2026 |

**Point-in-time membership.** Spells of `(ticker, start, end)` reconstructed from
the revision history of the Wikipedia constituents page, so a company that was
acquired in 2016 is in the cross-section until 2016 and not afterwards. Several
names have multiple spells. `validate_against_live` diffs the final row against the
live table so drift is visible rather than assumed away.

**Fundamentals that respect the filing date.** Every XBRL fact carries the date it
was filed, which is what makes it usable: a quarter ending 31 March is only visible
once the 10-Q lands in May. The median filing lag here is 34 days. Joining on period
end instead is the classic look-ahead in fundamental research. Two details that are
easy to get wrong and both cost real coverage:

- *Restatements.* A period is reported repeatedly as it is revised; the **earliest**
  filing is kept, because that is the number the market saw.
- *Year-to-date reporting.* Cash-flow statements are filed cumulatively (3-, 6-,
  9-, 12-month spans) and the 10-K reports the year rather than Q4. Naively taking
  "quarterly" facts dropped 56 of 72 quarters of operating cash flow for Apple.
  Both cases are the same problem — a long period sharing its start with a shorter
  one — and are solved by differencing to a fixed point.
- *Tag migration.* `SalesRevenueNet` gave way to
  `RevenueFromContractWithCustomerExcludingAssessedTax` under ASC 606 in 2018.
  Selecting a single tag truncates history at the switch, so candidate tags are
  merged rather than chosen.

**Known limitation.** 278 of 990 historical members could not be priced — Yahoo
drops delisted tickers, and those are exactly the survivorship-relevant names. So
survivorship bias is *reduced*, not eliminated. Per-date coverage of index members,
priced names and tradable names is written to `universe_coverage.csv` on every run.

---

## Method

**Factors (22).** Momentum at true trading-day horizons, residual momentum
(Blitz–Huij–Martens), short-term reversal, low volatility, idiosyncratic volatility
(Ang et al.), betting-against-beta, MAX (Bali–Cakici–Whitelaw), Amihud illiquidity,
turnover, and a liquidity *shock* rather than a level. Fundamentals: book-to-market,
earnings/cash-flow/sales yield, gross profitability (Novy-Marx), ROE, operating
margin, asset growth (Cooper–Gulen–Schill), accruals (Sloan), net share issuance
(Pontiff–Woodgate). Each is signed so higher means predicted-higher return, so a
negative IC means the anomaly failed in-sample rather than that a sign was flipped.

**Target.** Cross-sectional rank of the 21-day forward return, per date. This
removes the market component by construction, weights every date equally, and
bounds outliers.

**Validation.** Expanding walk-forward. Training stops `horizon + embargo` trading
days before each prediction date (purge plus 21-day embargo, López de Prado).
Overlapping labels are thinned to every fifth day and weighted by average
uniqueness, because 300k nominal rows are worth roughly 15k independent ones.

**Model, chosen by measurement.** `scripts/compare_models.py` scores five
constructions on identical out-of-sample dates:

| model | mean IC | ICIR | t | decile spread | spread t |
|---|---|---|---|---|---|
| **LightGBM on rank target** | **0.0124** | **0.123** | **1.51** | 46.1bp | 1.98 |
| equal-weight factor average | 0.0131 | 0.078 | 0.96 | 9.1bp | 0.24 |
| IC-weighted ensemble | 0.0069 | 0.053 | 0.65 | 37.3bp | 1.17 |
| LightGBM `rank_xendcg` | 0.0003 | 0.002 | 0.03 | 44.5bp | 1.06 |
| ridge | −0.0002 | −0.001 | −0.02 | 17.8bp | 0.60 |

The learning-to-rank variant is the instructive failure. It produces a competitive
decile spread and almost no rank correlation, because NDCG is deliberately top-heavy
— the right loss for search results and the wrong one for a book that earns as much
from its short tail as its long.

**Signal smoothing.** A three-month trailing average of the score raises IC from
0.0124 to 0.0165 *and* halves selection turnover (0.48 → 0.24). The information is
mostly in fundamentals that only move on a filing, so averaging sheds model noise
rather than signal. The window was chosen on out-of-sample data; the effect is
monotone from 1 to 6 months, so this is not a lucky point, but it is disclosed.

**Portfolio.** Alpha is converted to return units by Grinold's rule,
`alpha = IC × volatility × score`, then maximised subject to dollar neutrality, a
gross cap, a per-name box, |beta| ≤ 0.03, and an explicit annualised volatility
budget. Trading costs enter the objective at cost, so the optimiser makes the real
trade-off rather than an invented one.

---

## Results

Out-of-sample 2014-01-31 → 2026-07-31, monthly rebalance, 1bp commission and 5bp
slippage.

### Signal

| | |
|---|---|
| Mean IC | 0.0165 |
| IC information ratio | 0.164 (0.567 annualised) |
| t-statistic | 2.00 on 150 independent months |
| Decile spread | 68.1bp / month (t = 3.08) |
| Selection turnover | 24% / month |

### Strategy

| | |
|---|---|
| CAGR | 1.85% |
| Sharpe | 0.286 |
| Sortino | 0.44 |
| Max drawdown | −18.9% |
| Annualised volatility | 7.4% |
| Beta vs SPY | 0.011 |
| Day coverage | 96.4% (3,143 of 3,261 — the remainder are market holidays) |
| Rebalances skipped | **0** |
| Average turnover | 72% / rebalance |
| Average gross leverage | 1.45× |
| Average names | 67 |

Cost sensitivity: Sharpe 0.298 at 5bp slippage, 0.239 at 10bp, 0.121 at 20bp.

### Where the edge actually goes

This is the part worth reading. Decomposing the equal-weighted decile spread:

| | |
|---|---|
| Raw spread | 43.7bp / month → 5.24% / yr, **t = 2.11** |
| Beta of the spread to the market | **0.218** |
| Beta-adjusted alpha | 17.7bp / month → 2.12% / yr, **t = 0.92** |

The long decile runs a beta of 1.10 against the short decile's 0.99 — a tilt that is
positive in 73% of months — and over a window in which the market compounded at
13.7%. **59% of the apparent edge is market exposure, not stock selection.** Once
it is removed, what remains is not statistically distinguishable from zero.

Residualising every factor against beta and size at the signal level was tried and
made things worse: IC falls from 0.0165 to 0.0062, because at this horizon the value
edge substantially *is* a size effect and orthogonalising it removes the alpha along
with the exposure. The code is kept behind a `risk_neutral` flag, off by default,
with the measurement recorded.

### Robustness

Re-ordering the factor list — no economic change, only LightGBM's column sampling
and tie-breaking — moved the backtest Sharpe from 0.07 to 0.29. That is a warning,
so `scripts/robustness.py` re-fits the entire walk-forward under six model seeds
with identical economics:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | mean ± sd |
|---|---|---|---|---|---|---|---|
| IC | 0.0178 | 0.0157 | 0.0166 | 0.0163 | 0.0164 | 0.0154 | **0.0164 ± 0.0008** |
| Sharpe | 0.308 | 0.491 | 0.091 | 0.166 | 0.277 | 0.320 | **0.275 ± 0.138** |
| CAGR | 2.03% | 2.95% | 0.38% | 0.90% | 1.80% | 1.97% | 1.67% ± 0.91% |

This is the sharpest result in the repository. **The signal is stable and the
Sharpe is not.** IC varies by ±5% across seeds; Sharpe varies by a factor of five,
from 0.09 to 0.49. Quoting any single one of those as *the* Sharpe would be
choosing a draw from a distribution and calling it a measurement — which is,
in substance, what the 1.46 in the first version of this repository was.

### Honest summary

A defensible signal sits underneath this: value and quality factors, correctly
constructed on point-in-time data, with IC t-statistics near 2. What is not
defensible is calling it a strategy. In large-cap US equities, over 2014–2026,
after beta neutralisation and realistic costs, this edge does not clear the bar —
and the sample still carries residual survivorship bias that flatters it.

---

## Layout

```
project/
  configs/default.yaml      one file describing the experiment
  src/
    data/                   universe, prices, fundamentals, sectors, panel
    factors/                price, fundamental, transforms, registry
    model/                  target, splits, estimators, walkforward
    portfolio/              optimizer, risk, costs
    eval/                   ic, backtest, metrics, report
  scripts/
    build_data.py           download and cache; idempotent
    research.py             factor IC table -- the gate
    compare_models.py       model selection by measurement
    run_backtest.py         end to end
    robustness.py           seed dispersion
  tests/test_no_lookahead.py
```

## Run it

```bash
pip install -r requirements.txt

cd project
python -m scripts.build_data --stage universe
python -m scripts.build_data --stage prices
python -m scripts.build_data --stage fundamentals   # ~20 min, SEC rate limit

python -m scripts.research          # factor ICs, before any trading
python -m scripts.compare_models    # pick the model on evidence
python -m scripts.run_backtest      # full run -> reports/<timestamp>/
python -m scripts.robustness        # error bars
pytest tests
```

Each run writes a self-contained folder: the config that produced it, universe
coverage, the factor IC table, signal diagnostics, quantile returns, both return
series, holdings, per-rebalance turnover and cost, and feature importances.

## Stack

Python · pandas · LightGBM · CVXPY/Clarabel · scikit-learn · pytest
