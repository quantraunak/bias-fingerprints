# Point-in-Time US Equity Factor Research

A cross-sectional equity signal built on reconstructed point-in-time index membership
and SEC XBRL fundamentals joined on filing date, with a purged walk-forward evaluation
that runs before any portfolio is constructed.

| | |
|---|---|
| Mean IC | **0.0165** (t = 2.00 over 150 out-of-sample months) |
| IC information ratio | 0.164 monthly · 0.567 annualised |
| Decile spread | **68.1bp / month** (t = 3.08) |
| Universe | 727 names, point-in-time membership, 2004–2026 |
| Fundamentals | SEC EDGAR XBRL, 648 issuers, keyed on filing date |
| Factors | 22, each signed to its published direction |

The signal carries information. The section on
[beta decomposition](#beta-decomposition) shows how much of it survives once the market
exposure embedded in the decile spread is charged against it — which is the question
that decides whether a factor study is a strategy.

[**Results dashboard**](https://quantraunak.github.io/ls-multifactor-research/) · [Data notes](docs/DATA.md)

---

## Data

Universe construction is where the ceiling on this kind of study is usually set, so it
is where most of the work went.

### Point-in-time membership

Index membership is stored as spells of `(ticker, start, end)`, reconstructed from the
revision history of the Wikipedia constituents page. A company acquired in 2016 is in
the cross-section until 2016 and absent afterwards; several names have multiple spells
across the window. `validate_against_live` diffs the final row against the live table
on every build, so membership drift surfaces as a failure rather than an assumption.

Applying *current* membership to a historical panel is the standard survivorship trap:
every company that was ever deleted — acquired, bankrupted, demoted — becomes invisible,
and the surviving sample is selected on exactly the outcome being predicted.

### Fundamentals that respect the filing date

Every XBRL fact carries the date it was filed, which is what makes it usable in a
backtest: a quarter ending 31 March is only visible once the 10-Q lands in May. The
median filing lag in this panel is **34 days**. Joining on period end instead is the
classic look-ahead in fundamental research.

Three details in the XBRL feed each cost real coverage if handled naively:

- **Restatements.** A period is reported repeatedly as it is revised. The *earliest*
  filing is kept, because that is the number the market actually saw.
- **Year-to-date reporting.** Cash-flow statements are filed cumulatively (3-, 6-, 9-,
  12-month spans) and the 10-K reports the full year rather than Q4. Taking "quarterly"
  facts at face value dropped 56 of 72 quarters of operating cash flow for Apple. Both
  cases are the same problem — a long period sharing its start date with a shorter one —
  and both are solved by differencing to a fixed point.
- **Tag migration.** `SalesRevenueNet` gave way to
  `RevenueFromContractWithCustomerExcludingAssessedTax` under ASC 606 in 2018. Selecting
  a single tag truncates history at the switch, so candidate tags are merged rather than
  chosen.

Industry classification comes from SEC SIC codes mapped to the Fama-French 12, which
covers the full historical panel rather than only names a vendor still lists.

### Coverage

Per-date counts of index members, priced names and tradable names are written to
`universe_coverage.csv` on every run, so the tradable cross-section is an observable
rather than an assumption.

---

## Factors

Twenty-two factors, each signed so that higher means predicted-higher return, in the
direction the published anomaly runs. A negative IC therefore means the anomaly failed
over this sample rather than that a sign was flipped somewhere in the code.

**Price and volume.** Momentum at true trading-day horizons, residual momentum
(Blitz–Huij–Martens), short-term reversal, low volatility, idiosyncratic volatility
(Ang et al.), betting-against-beta, MAX (Bali–Cakici–Whitelaw), Amihud illiquidity,
turnover, and a liquidity *shock* rather than a level.

**Fundamental.** Book-to-market, earnings yield, cash-flow yield, sales yield, gross
profitability (Novy-Marx), ROE, operating margin, asset growth
(Cooper–Gulen–Schill), accruals (Sloan), net share issuance (Pontiff–Woodgate).

Horizons are specified in trading days throughout. Subtracting calendar days and
calling them trading days compresses every lookback by roughly 1.45× — a 252-day
momentum window becomes 173 trading days, and "12–1 momentum" silently measures 8.2
months. `factors/transforms.py` takes its offsets from the price index itself.

---

## Validation

**Target.** Cross-sectional rank of the 21-day forward return, per date. Ranking
removes the common date move by construction, weights every date equally regardless of
market volatility, and bounds outliers. Regressing raw forward returns instead spends a
substantial share of the loss budget on the market component — measured at 28% of the
label's variance on this panel — which is identical for every name on a date and so
cannot affect the cross-sectional ordering the strategy actually trades.

**Splits.** Expanding walk-forward. Training stops `horizon + embargo` trading days
before each prediction date: a purge for the 21-day label span plus a 21-day embargo,
following López de Prado. Overlapping labels are thinned to every fifth day and
weighted by average uniqueness, because 300k nominal rows are worth roughly 15k
independent ones and unweighted fitting overstates the effective sample by an order of
magnitude.

**Look-ahead guards.** [`project/tests/test_no_lookahead.py`](project/tests/test_no_lookahead.py)
asserts the properties that keep the evaluation honest, including:

- every rebalance date exists in the price index — the test builds a calendar whose
  month-ends deliberately fall on weekends, since `resample("ME")` returns calendar
  month-ends and 55 of 180 in this window are not trading days
- annualisation divides by elapsed calendar span, not observation count, so a series
  with days removed cannot report a higher growth rate
- factor lookbacks resolve to trading-day offsets taken from the index
- no training fold contains a date inside the purge-plus-embargo window of its
  prediction date

**Model selection.** `scripts/compare_models.py` scores five constructions on identical
out-of-sample dates:

| model | mean IC | ICIR | t | decile spread | spread t |
|---|---|---|---|---|---|
| **LightGBM on rank target** | **0.0124** | **0.123** | **1.51** | 46.1bp | 1.98 |
| equal-weight factor average | 0.0131 | 0.078 | 0.96 | 9.1bp | 0.24 |
| IC-weighted ensemble | 0.0069 | 0.053 | 0.65 | 37.3bp | 1.17 |
| LightGBM `rank_xendcg` | 0.0003 | 0.002 | 0.03 | 44.5bp | 1.06 |
| ridge | −0.0002 | −0.001 | −0.02 | 17.8bp | 0.60 |

The learning-to-rank variant is the instructive one. It produces a competitive decile
spread and almost no rank correlation, because NDCG is deliberately top-heavy — the
right objective for search results and the wrong one for a book that earns as much from
its short tail as its long. Ridge fails differently: with missing fundamentals filled at
zero and heavy shrinkage, a linear model cannot express the interactions the trees find.

**Signal smoothing.** A three-month trailing average of the score raises IC from 0.0124
to 0.0165 *and* halves selection turnover, 0.48 → 0.24. The information sits mostly in
fundamentals that only move when a filing lands, so averaging sheds model noise rather
than signal. Improving IC and turnover together is uncommon enough to be worth stating
explicitly. The window was chosen on out-of-sample data; the effect is monotone from 1
to 6 months rather than a single lucky point, and it is disclosed on that basis.

---

## Portfolio construction

Alpha is converted to return units by Grinold's rule, `alpha = IC × volatility × score`,
then maximised subject to dollar neutrality, a gross cap, a per-name box constraint,
|beta| ≤ 0.03, and an explicit annualised volatility budget. Trading costs enter the
objective at cost, so the optimiser makes the real trade-off between expected alpha and
the friction of reaching it rather than an invented one. Solved with CVXPY/Clarabel.

Out-of-sample 2014-01-31 → 2026-07-31, monthly rebalance, 1bp commission and 5bp
slippage:

| | |
|---|---|
| CAGR | 1.85% |
| Sharpe | 0.29 |
| Sortino | 0.44 |
| Annualised volatility | 7.4% |
| Max drawdown | −18.9% |
| Beta vs SPY | 0.011 |
| Average turnover | 72% / rebalance |
| Average gross leverage | 1.45× |
| Average names | 67 |
| Day coverage | 96.4% (3,143 of 3,261 — remainder are market holidays) |
| Rebalances skipped | 0 |

Sharpe by slippage assumption: 0.30 at 5bp, 0.24 at 10bp, 0.12 at 20bp. For a book
whose gross edge is of the same order as its trading costs, that gradient is the
expected shape rather than a surprising one.

---

## Beta decomposition

A decile spread with a t-statistic above 2 looks like a strategy. Regressing that spread
on the market says otherwise, and this is the result the project is really about.

| | |
|---|---|
| Raw spread | 43.7bp / month → 5.24% / yr, t = **2.11** |
| Beta of the spread to the market | **0.218** |
| Beta-adjusted alpha | 17.7bp / month → 2.12% / yr, t = **0.92** |

The long decile runs a beta of 1.10 against the short decile's 0.99 — a tilt that is
positive in 73% of months, over a window in which the market compounded at 13.7%.
**59% of the apparent edge is market exposure rather than stock selection.** The factors
doing the work are value and quality, and value is cyclical; the book was being paid for
exposure it did not intend to take.

Residualising every factor against beta and size at the signal level was tried and made
things worse: IC falls from 0.0165 to 0.0062, because at this horizon the value edge
substantially *is* a size effect, and orthogonalising the exposure away removes the alpha
with it. The code is kept behind a `risk_neutral` flag, off by default, with the
measurement recorded rather than the attempt deleted.

---

## Robustness

Re-ordering the factor list — no economic change, only LightGBM's column sampling and
tie-breaking — moved the backtest Sharpe from 0.07 to 0.29. Rather than pick one,
`scripts/robustness.py` re-fits the entire walk-forward under six model seeds with
identical economics:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | mean ± sd |
|---|---|---|---|---|---|---|---|
| IC | 0.0178 | 0.0157 | 0.0166 | 0.0163 | 0.0164 | 0.0154 | **0.0164 ± 0.0008** |
| Sharpe | 0.308 | 0.491 | 0.091 | 0.166 | 0.277 | 0.320 | **0.275 ± 0.138** |
| CAGR | 2.03% | 2.95% | 0.38% | 0.90% | 1.80% | 1.97% | 1.67% ± 0.91% |

**The signal is stable; the Sharpe is not.** IC varies by ±5% across seeds while Sharpe
varies by a factor of five, 0.09 to 0.49. A single backtest Sharpe from a model with
stochastic fitting is a draw from that distribution, and quoting one without the spread
around it reports the draw as though it were the measurement. Every performance figure
in this repository should be read against the dispersion in this table.

---

## What this establishes, and what it does not

A defensible cross-sectional signal sits underneath this: value and quality factors,
constructed on point-in-time data with filing-date-aware fundamentals, reaching IC
t-statistics near 2 and a decile spread at t = 3.08 across 150 independent months.

At this horizon and in this universe, the portfolio built on it does not clear a
tradability bar. Most of the raw spread is beta; what remains after neutralising it is
not statistically separable from zero at these costs. Large-cap US equity is the most
heavily arbitraged cross-section available, and 22 public factors on free data is not
where an edge in it is likely to be found — a study of this design is better read as a
measurement of how much of a published anomaly stack survives correct construction.

**Residual survivorship bias.** 278 of 990 historical members could not be priced —
Yahoo drops delisted tickers, and those are precisely the survivorship-relevant names.
Survivorship bias is *reduced, not eliminated*, and the remainder flatters these
results.

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

Each run writes a self-contained folder: the config that produced it, universe coverage,
the factor IC table, signal diagnostics, quantile returns, both return series, holdings,
per-rebalance turnover and cost, and feature importances. Nothing in this README is
transcribed by hand from a notebook.

## Stack

Python · pandas · LightGBM · CVXPY/Clarabel · scikit-learn · pytest
