---
name: backtest-hygiene
description: Review checklist for backtest and simulation code — survivorship, look-ahead, calendar handling, cost and capacity modelling, and the arithmetic errors that inflate reported performance. Use when writing or reviewing a backtest loop, portfolio simulation, or performance report, when a backtest result looks too good, or when auditing an existing engine before trusting its numbers.
---

# Backtest hygiene

Every item below has produced a plausible-looking, wrong number in production
research code. They are ordered by how much damage they typically do.

## Universe and survivorship

- [ ] Membership is **point-in-time**: names enter on their real entry date and
      leave on their real exit date. Applying today's index membership to history
      selects on survival — the exact outcome being predicted.
- [ ] Delisted names are present, with their **delisting return** applied.
      Dropping a name that went to zero and stopping its series at the last quote
      is a large, systematic upward bias in the short book especially.
- [ ] Tickers are mapped through a **CIK/permno-style stable identifier**, not
      the ticker string. Tickers are reused: `FB` → `META`, and a ticker retired
      by one company is reissued to another.
- [ ] Coverage is written out per date and inspected. If tradable names per date
      collapses in some period, the backtest is silently trading a different
      universe there.

## Look-ahead

- [ ] Fundamentals join on **filing date**, never period end.
- [ ] Restatements resolve to the **first reported** value.
- [ ] Any `.rolling()`, `.ewm()`, or `.shift()` is checked for centering — a
      centred window sees the future. `min_periods` must not silently backfill.
- [ ] Cross-sectional transforms (z-score, rank, winsorize, neutralize) are
      computed **per date**, not over the pooled panel. Pooling leaks the
      full-sample distribution into every date.
- [ ] Missing-value imputation uses trailing statistics only. `fillna(mean())`
      over the whole column is look-ahead.
- [ ] The signal available at rebalance uses data through the **prior** close if
      you trade at that close; if you trade next open, say so and lag accordingly.
- [ ] Universe screens (price floor, ADV floor) use trailing windows, so a name
      drops out on the day that becomes visible, not for the whole sample.

## Calendar arithmetic

- [ ] Rebalance dates are **actual trading days**. `resample("ME")` returns
      calendar month-ends; when the 31st is a Saturday the date is absent from
      the price index, the cross-section comes back empty, and — depending on
      where the `continue` sits — that period's P&L can vanish entirely.
- [ ] Every trading day in the window carries a return. Assert
      `len(returns) == len(trading_days)` rather than assuming it.
- [ ] Lookback horizons are in **trading days** taken from the index, not
      calendar days. Subtracting calendar days and calling them trading days
      compresses every horizon by ~1.45×: a 252-"day" momentum window becomes 173
      trading days, so "12–1 momentum" silently measures 8.2 months.
- [ ] Annualisation divides by **elapsed calendar span**, not observation count.
      `252 / len(returns)` treats a series with holes as though it were shorter,
      inflating CAGR.
- [ ] Rebalance failures are recorded and carried forward as a flat period, not
      skipped. A skipped rebalance that also skips its P&L is invisible.

## Costs and capacity

- [ ] Commission **and** spread/slippage are charged, on turnover, at every
      rebalance — including the initial build and the final liquidation.
- [ ] Cost sensitivity is reported at several slippage assumptions. If the result
      dies at 20bp, say so; that is a capacity statement.
- [ ] **Market impact** scales with participation. A constant bps charge assumes
      infinite liquidity. At minimum, cap position size at a fraction of trailing
      ADV and report the implied capacity in dollars.
- [ ] Short positions carry **borrow cost**, and hard-to-borrow names are either
      excluded or charged realistically. A market-neutral book that shorts small
      illiquid names for free is not implementable.
- [ ] Turnover is reported per rebalance, and **book turnover is reconciled
      against selection turnover**. If the book churns far more than the signal
      changes, the optimiser is generating trades the signal did not ask for.

## Portfolio construction

- [ ] Weights are marked to market between rebalances. Holding fixed weights
      assumes a free daily rebalance back to target.
- [ ] The optimiser sees a cross-section wide enough to express the signal.
      Hard-truncating to deciles before optimisation discards the ordering
      information in the middle of the distribution, which is most of the IC.
- [ ] A linear objective over box constraints produces **corner solutions** —
      the book piles onto `max_weight` for a handful of names. If risk belongs in
      the problem, put it in the objective and solve for the risk-aversion term
      (bisect until the vol constraint binds) rather than moving it to a
      constraint and accepting the concentration.
- [ ] **Transfer coefficient** is measured: the correlation between the alpha
      vector and the implemented weights. A low TC means the constraints, not the
      signal, are determining the book.
- [ ] Constraints that neutralize exposure (beta, sector, size) are applied at
      the **portfolio** level. Orthogonalising factors at the *signal* level
      destroys information rather than shaping implementation, and the two are
      routinely confused — a negative result from signal-level residualisation
      says nothing about portfolio-level neutralisation.
- [ ] Any de-grossing or drawdown throttle is tested with it **off**. On a
      mean-reverting P&L stream, cutting risk into a drawdown systematically sells
      the recovery.

## Metrics

- [ ] Sharpe is computed on the correct frequency and annualised consistently
      (`mean/std × √252` for daily), and excess of the right rate.
- [ ] Reported beta and alpha come from a regression on the **same** return
      series that produced the Sharpe.
- [ ] Drawdown is computed on the compounded equity curve, not on cumulative
      sums of returns.
- [ ] Results are reported by sub-period. A single number over a long window
      hides regime structure.
- [ ] Every headline figure is regenerated from a run directory rather than
      transcribed by hand, so a stale number is impossible rather than merely
      unlikely.

## Tests worth writing

Guard the properties, not the outputs:

```python
def test_rebalance_dates_are_trading_days():
    # calendar built so month-ends deliberately fall on weekends
    assert set(rebalance_dates).issubset(set(price_index))

def test_annualisation_ignores_missing_days():
    # same span, a third of the days removed
    assert cagr(sparse) <= cagr(full) + tolerance

def test_no_training_row_inside_embargo():
    assert (train.index.max() + horizon + embargo) <= predict_date

def test_cross_sectional_transform_is_per_date():
    # a date whose values are shifted by a constant must produce identical ranks
    assert ranks(values).equals(ranks(values + 100))
```

## The disposition to hold

When a backtest looks good, the first hypothesis is that it is wrong. Spend the
first hour trying to break it, not extending it. A result that survives a
genuine attempt to falsify it is worth something; one that was never attacked is
worth nothing regardless of its Sharpe.
