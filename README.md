# Bias Fingerprints

**A method for auditing factor research from the outside**, and the point-in-time
substrate it is calibrated on: prices and index membership that know what was true on
each historical date, and SEC filings keyed to the day they were filed rather than the
period they describe.

**Paper: [Bias Fingerprints](paper/bias_fingerprints.pdf)** (9pp, LaTeX source in
[`paper/`](paper/)).

---

## 1. Bias fingerprinting

**A method for auditing factor research from the outside.**

A factor table is usually all an outside reader gets. An allocator reads a manager's IC
table, a referee reads a submitted backtest, a desk prices a vendor signal. None of them
can run the pipeline that produced the numbers, and the literature on look-ahead and
survivorship bias is written for someone who can: date the fundamentals correctly, build
the universe point-in-time, shift a feature forward and check the result degrades. It
assumes the reader controls the code.

Bias fingerprinting reverses that direction. A data-handling defect is not a scalar
amount of inflation but a *direction* in factor space, and the direction is a property of
the defect rather than of the study it damages. Measure the direction once on a pipeline
you own; then test for it in tables you do not.

The framework has four parts:

**A generator.** One factor study, re-run under alternative data conventions with
prices, universe, tradability screen, forward returns and evaluation code held fixed. The
difference between two arms is attributable to the convention and to nothing else.

**A signature.** `δ_b = t_b − t_pit`, the vector of t-statistic shifts a defect `b`
induces across the factor cross-section, one component per factor.

**An inference model.** An observed table is `t = τ + δ_b + ε`, with true performance `τ`
unknown and twenty-two dimensional. Identification comes from three sources rather than a
fitted classifier: *exclusion* (dating bias cannot move a factor that never reads a filed
figure, so a study whose strongest results sit in that block is not a dating-bias study,
whatever τ is), *within-family contrast* (family means absorb τ), and *projection* onto
each signature after removing family means, which has a computable reference distribution
and so yields a p-value rather than a label.

**A validation protocol.** Four gates, in order, each of which ends the method rather
than downgrading it: signature stability across subperiods and half-universes, signature
sampling covariance, separability under realistic noise, and only then calibrated
application. Full statement in [`docs/FINGERPRINT.md`](docs/FINGERPRINT.md).

### Two signatures, calibrated

Measured on the generator described below, changing one convention at a time.

| | Period-end join | Current-membership universe |
|---|---|---|
| Mean IC | **+59%** among the 11 factors that read a filed figure | **−0.0043** across all 22 |
| Direction | Uniform inflation | Relocation, not inflation |
| Price and volume factors | **Unchanged, exactly** (max drift 0.0e+00) | Heavily distorted |
| False positives | 4 (`earnings_yield`, `cash_flow_yield`, `roe`, `accruals`) | 2, incl. one with the wrong sign |
| Largest single move | `accruals`, +1.77 t | `amihud_illiquidity`, **+3.91 t** |

The two signature vectors correlate at **−0.088** and disagree in sign on value: dating
inflates `earnings_yield` and `roe` by +0.98 and +1.07 t-units, current membership
deflates them by −1.28 and −1.24. A table cannot be explained by both, which is what makes
the diagnostic feasible.

Under current-membership conditioning, `amihud_illiquidity` flips from t = −1.11 to
**+2.80** and the low-volatility anomaly inverts. The channel is not the survival filter
usually named. The biased panel wrongly *includes* 203 names a day that had not yet been
admitted to the index, against 88 it wrongly excludes, and the included cohort is a third
the size and a third the liquidity of a genuine constituent. Conditioning on current
membership is a forward-looking growth filter, and that mechanism predicts the shape of
the signature before the signature is measured.

A fixed 45-day filing lag, by contrast, inflates mean IC by only 9% and manufactures no
false discoveries in this sample. Approximately right and wrong are different things, and
only the second leaves a signature worth testing for.

### Status

Framework specified, generator built, two signatures calibrated and reproducible with
`make bias`. Gates one through three have not been run, and the ordering is part of the
method: a diagnostic that reports a label without a calibrated reference distribution is
worse than no diagnostic. Gate three is the one that would normally be blocked on labelled
data, and it is free here — biased panels come out of the generator, so the confusion
matrix can be simulated with no annotation and no external dataset.

Measurements in [`docs/BIAS.md`](docs/BIAS.md), method in
[`docs/FINGERPRINT.md`](docs/FINGERPRINT.md), paper in [`paper/`](paper/).

---

## 2. Point-in-time economic-link extraction

**Moved to [`filing-links`](https://github.com/quantraunak/filing-links).**

A construction protocol for dated firm-relationship graphs from SEC filing text: a
benchmark with auditable negatives, a measured extraction cost/quality frontier, and a
pre-registered coverage gate. It shared a point-in-time substrate with the work above and
nothing else, so it now has its own repository, with the history that makes its
pre-registration verifiable.

The two projects still share `project/src/data` -- the universe, price and fundamentals
layer -- which is vendored into both rather than published separately.

---

## 3. The reference study

**A 22-factor long-short cross-section on the shared substrate.** It is the generator the
bias-fingerprinting framework is calibrated on, and it is complete in its own right.

| | |
|---|---|
| Mean IC | **0.0165** (t = 2.00 over 150 out-of-sample months) |
| IC information ratio | 0.164 monthly · 0.567 annualised |
| Decile spread | **68.1bp / month** (t = 3.08) |
| Universe | 727 names, point-in-time membership, prices from 2004 |
| Fundamentals | SEC EDGAR XBRL, 648 issuers, keyed on filing date, usable from 2010 |

Two results from auditing it generalise past this study, and both are reported below:
[59% of an apparently significant decile spread is market beta](#beta-decomposition), and
[a single backtest Sharpe from a stochastically fitted model is a draw from a
distribution five times as wide as the point estimate suggests](#robustness).

**Shared infrastructure.** Point-in-time index membership from reconstructed spells, SEC
XBRL keyed on filing date, an entity resolver from filing text to tickers, and a
hand-annotated benchmark for relationship extraction with
[auditable negatives](docs/gold_exclusions.md).

Fundamental coverage begins in 2010, not 2004. The SEC's XBRL mandate phased in over
2009–2011, so `companyfacts` returns nothing for earlier periods, and the first filings
carry backfilled historical statements whose filing dates are not the dates the market
saw. The price panel and the twelve price factors run from 2004; the ten fundamental
factors do not exist before 2009 and are not stable until 2011. Out-of-sample evaluation
starts in 2014, so the headline figures are unaffected — but the panel should not be read
as twenty-two years of fundamentals. This is measured in [`docs/BIAS.md`](docs/BIAS.md).

[**Results dashboard**](https://quantraunak.github.io/bias-fingerprints/) · [Data notes](docs/DATA.md)

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
and the surviving sample is selected on exactly the outcome being predicted. Section 1
measures what that does to each of the twenty-two factors.

### Fundamentals that respect the filing date

Every XBRL fact carries the date it was filed, which is what makes it usable in a
backtest: a quarter ending 31 March is only visible once the 10-Q lands in May. The
median filing lag in this panel is **34 days**. Joining on period end instead is the
classic look-ahead in fundamental research, and it is the first of the two calibrated
signatures.

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

The split is what supplies the exclusion restriction in Section 1: eleven factors read a
filed figure and eleven do not, so the second block cannot respond to the filing calendar
by construction. `turnover_1m` is the one factor that looks price-only and is not, because
it divides by shares outstanding. Dating sensitivity is determined by measurement and
checked against a declared set, so a factor that quietly acquires a filed input is
surfaced rather than hidden among the controls.

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
on the market decides whether it is one, and this is the first of the two general results
the reference study produced.

| | |
|---|---|
| Raw spread | 43.7bp / month → 5.24% / yr, t = **2.11** |
| Beta of the spread to the market | **0.218** |
| Beta-adjusted alpha | 17.7bp / month → 2.12% / yr, t = **0.92** |

The long decile runs a beta of 1.10 against the short decile's 0.99 — a tilt that is
positive in 73% of months, over a window in which the market compounded at 13.7%.
**59% of the apparent edge is market exposure rather than stock selection.** The factors
doing the work are value and quality, and value is cyclical; the book was being paid for
exposure it did not intend to take. A decile-spread t-statistic reported without its
beta decomposition is not a claim about stock selection, and that holds for any study of
this shape rather than only this one.

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
varies by a factor of five, 0.09 to 0.49. This is the second general result: a single
backtest Sharpe from a model with stochastic fitting is a draw from that distribution, and
quoting one without the spread around it reports the draw as though it were the
measurement. Every performance figure in this repository should be read against the
dispersion in this table.

---

## What the reference study establishes, and what it does not

A defensible cross-sectional signal sits underneath it: value and quality factors,
constructed on point-in-time data with filing-date-aware fundamentals, reaching IC
t-statistics near 2 and a decile spread at t = 3.08 across 150 independent months. That
is what qualifies it as a generator — the correct arm has to be worth calibrating
against.

At this horizon and in this universe, the portfolio built on it does not clear a
tradability bar. Most of the raw spread is beta; what remains after neutralising it is
not statistically separable from zero at these costs. Large-cap US equity is the most
heavily arbitraged cross-section available, and 22 public factors on free data is not
where an edge in it is likely to be found. The study is built and reported as a
measurement of how much of a published anomaly stack survives correct construction, which
is also exactly what a bias-fingerprinting generator needs to be.

**Residual survivorship limitation.** 278 of 990 historical members could not be priced —
Yahoo drops delisted tickers, and those are precisely the survivorship-relevant names.
Survivorship bias is *reduced, not eliminated*, and the remainder flatters these
results. It also bounds the second signature: `docs/BIAS.md` reports what that does to the
measurement.

---

## Layout

```
paper/
  bias_fingerprints.tex     the paper; `make` compiles, `make arxiv` tarballs
  make_figures.py           figure regenerated from the measured CSVs
docs/
  FINGERPRINT.md            the framework: generator, signature, inference, protocol
  BIAS.md                   the two calibrated signatures, in full
  HYPOTHESIS.md             link-protocol pre-registration, written before the data
  SPECIFICATIONS.md         every choice tried, including the ones that failed
  EXTRACTION.md             cost/quality frontier for local relation extraction
  RESULT_01.md              a pre-registered hypothesis, rejected
  RELATED_WORK.md           literature checked before building, not after
  DATASHEET.md / RELEASE.md what ships, in what form, under what licence
project/
  configs/default.yaml      one file describing the experiment
  src/
    data/                   universe, prices, fundamentals, sectors, panel
    factors/                price, fundamental, transforms, registry
    model/                  target, splits, estimators, walkforward
    portfolio/              optimizer, risk, costs
    eval/                   ic, backtest, metrics, report
    graph/                  filings, passages, extract, resolve, build, signal
  scripts/
    build_data.py           download and cache; idempotent
    research.py             factor IC table -- the gate
    pit_vs_naive.py         dating signature
    survivorship.py         universe signature
    benchmark_table.py      extraction frontier, from cache, no GPU
    run_backtest.py         end to end
  tests/                    126 tests + 1 xfail, offline and deterministic
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

Measure the signatures and reproduce the paper:

```bash
make bias            # writes reports/pit_vs_naive.csv and reports/survivorship.csv
cd paper && make     # regenerates the figure from those CSVs, compiles the PDF
```

Each run writes a self-contained folder: the config that produced it, universe coverage,
the factor IC table, signal diagnostics, quantile returns, both return series, holdings,
per-rebalance turnover and cost, and feature importances. Nothing in this README is
transcribed by hand from a notebook.

## Stack

Python · pandas · LightGBM · CVXPY/Clarabel · scikit-learn · pytest
