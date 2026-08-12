# Systematic Equity Research

Two connected pieces of work on the same question — how a taxable investor should actually hold equities.

**`project/`** — a market-neutral long/short signal model on the S&P 500: gradient-boosted cross-sectional ranking, purged walk-forward validation, and convex portfolio optimization under dollar- and beta-neutral constraints.

**`advisor/`** — an after-tax portfolio engine: position sizing under a Kelly criterion adjusted for capital-gains tax, lot-level loss harvesting, asset location across tax wrappers, and estate exposure.

The two compose. The signal model estimates an edge; the advisor engine decides how much of it to hold, after tax.

---

## Results — signal model

Out-of-sample from 2017-01-01. Trained on a rolling 36-month window, monthly rebalance.

| Metric | Value |
|--------|-------|
| CAGR | 5.3% |
| Sharpe | 0.53 |
| Sortino | 0.87 |
| Max drawdown | −14.0% |
| Avg positions | 43 |
| Gross leverage | 1.78× |
| Monthly turnover | 98% |

Net of 1 bp commission and 5 bp slippage on turnover.

**Read this before the numbers.** An earlier version of this README reported a Sharpe of 1.46 and a CAGR of 35.3%. Those figures came from a configuration with four long and four short positions, on a 46-name universe, with a portfolio optimizer that was never actually binding. They were not reproducible and have been withdrawn. The table above is what the current code produces on the full universe with the optimizer running. It is a much lower number and a much more honest one — see [Known limitations](#known-limitations).

---

## Method

**Universe.** 500 S&P constituents (current membership — see limitations). Names with more than 5% missing bars over the sample are dropped, leaving ~420.

**Factors (10).** `mom_12_1`, `mom_6_1`, `mom_3_1`, `reversal`, `low_vol`, `idio_vol`, `trend`, `liquidity`, `vol_momentum`, `quality_proxy`. Winsorized at 1/99, z-scored, sector-neutralized.

**Model.** LightGBM ranker, 200 trees, depth 3. Purged walk-forward cross-validation with a 21-day embargo between train and test folds — the forward-return horizon is 21 days, so without the embargo the label of a training sample overlaps the test window.

**Selection.** Top and bottom deciles of the cross-sectional prediction.

**Portfolio.** Mean-variance optimization (CVXPY / CLARABEL) with Ledoit-Wolf shrinkage on the covariance matrix, subject to:

- `sum(w) = 0` — dollar neutral
- `sum(|w|) <= L` — gross leverage cap
- `|w_i| <= 0.05` — per-name cap
- `|beta'w| <= 0.05` — beta neutral vs SPY
- turnover penalty in the objective

Weights are rescaled to the leverage target after solving, then clipped to the box.

**Risk.** EWMA realized volatility targeted at 12% annualized, gross leverage scaled between 0.4× and 2.0×, with de-grossing when drawdown exceeds 8% from peak.

---

## Three bugs worth documenting

These were found while reproducing the original results. Each one silently produced plausible-looking output, which is the dangerous kind.

**1. The leverage floor was vacuous.** Minimum gross exposure was expressed as:

```python
t >= w,  t >= -w,  sum(t) >= 0.98 * L
```

`t` is bounded *below* by `|w|` and carries no cost in the objective, so the solver satisfies the constraint by inflating `t` while driving `w` to zero. Reported gross leverage read 2.0 while actual weights were ~1e-11. Minimum gross is a non-convex constraint and cannot be written this way; leverage is now enforced by rescaling after the solve.

**2. Signal scale vs risk penalty.** Raw model output forecasts 21-day returns (order 1e-3). The quadratic risk term is order 1e-2. On the raw scale the penalty dominates and the zero portfolio is optimal regardless of signal quality. Only the cross-sectional *ordering* carries information, so the signal is now standardized to unit dispersion before entering the objective.

**3. Per-name cap below equal weight.** `max_weight` was 0.02 while the strategy held 8 positions at 1.8× gross — an equal-weight position is 0.225, so the box was infeasible at every rebalance. It is now 0.05 against ~43 positions.

The universe was also pointed at a 51-ticker cache subset rather than the 500-name file, and both price sources had broken (Stooq's endpoints returned 404; the pinned `yfinance==0.2.40` failed against Yahoo's current API and fell through to a stale cache).

---

## Known limitations

- **Survivorship bias.** The universe is *current* S&P 500 membership, not point-in-time. Companies that were delisted, acquired, or dropped from the index never appear. This inflates results and is not fixable with free data — it needs CRSP or Norgate.
- **Turnover.** 98% monthly is high. Results are sensitive to the cost assumption; at 20 bp slippage the strategy is roughly flat.
- **No fundamental data.** Price and volume factors only.
- **Single validation window.** Hyperparameters were not selected on a held-out period, so some in-sample tuning risk remains.
- **Sharpe 0.53 is a modest result.** It is presented as-is rather than tuned upward. A market-neutral strategy at this Sharpe, before fees and financing on the short book, is not obviously investable.

---

## After-tax engine (`advisor/`)

The part I think is more defensible than the signal model.

**Tax-aware Kelly sizing.** Textbook Kelly assumes rebalancing is free. For a taxable holder sitting on an embedded gain, moving to the optimum costs a fixed fraction of every dollar traded. Setting the growth benefit of better sizing against the certain tax cost gives a no-trade band:

```
d* = 2 * gain% * taxRate / (sigma^2 * horizon)
```

Inside the band, doing nothing beats trading. Outside it, the correct action is to trade to the *edge* of the band, not to the optimum — the last stretch costs more tax than the sizing precision is worth. On a concentrated position with an 80% embedded gain this is the difference between "trim to 15%" and "trim to 25%".

**Also included:** lot-level loss-harvesting simulation with wash-sale blocking and a loss-utilization cap, asset location as an assignment problem across taxable / IRA / Roth / trust, forward-looking capital market assumptions built from yields and growth rather than historical averages, and estate transfer exposure.

**`advisor/web/`** — a browser tool over the same engine. All computation is client-side; holdings never leave the page.

---

## Run it

```bash
make install
make test
make run       # python project/run_backtest.py --config project/configs/default.yaml
```

Outputs land in `project/reports/latest/<timestamp>/`: metrics, equity curve, holdings, factor ICs, feature importances, a QuantStats tear sheet, and a frozen config snapshot.

```bash
cd advisor/web && npm install && npm run dev
```

---

## Stack

Python · LightGBM · CVXPY · scikit-learn · pandas · pytest · Next.js · TypeScript
