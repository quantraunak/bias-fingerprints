# Portfolio construction: where the signal was being lost

One signal, held fixed, run through three constructions. Every difference below
is attributable to construction rather than to the model, because the walk-forward
scores are cached and identical across variants.

All figures net of 1bp commission, 5bp slippage, and stock borrow at 40bp on
short notional.

| variant | Sharpe | CAGR | vol | max DD | beta | names | turnover | TC |
|---|---|---|---|---|---|---|---|---|
| A baseline — deciles, 1.5x gross | 0.240 | 1.47% | 7.2% | −19.5% | 0.009 | 68 | 0.71 | 0.725* |
| **D full cross-section, 5x gross** | **0.365** | 2.86% | 8.8% | −21.2% | −0.002 | 205 | 1.73 | 0.706 |
| E full, 5x gross, industry neutral | 0.229 | 1.63% | 8.8% | −21.7% | 0.002 | 208 | 1.76 | 0.691 |

\* measured inside the decile-truncated set; against the full cross-section the
baseline transfer coefficient is **0.384**.

## What was actually wrong

The book was pinned at a corner by arithmetic in the configuration, not by the
optimiser's objective.

`gross_leverage / max_weight = 1.5 / 0.03 = 50`. Reaching gross 1.5 therefore
requires at least 50 names at exactly the cap, and the gross and box constraints
together already determine the solution. Realised volatility sat at 7.0% against
an 8% budget, so the risk term never bound either — a dollar- and beta-neutral
large-cap book cannot reach 8% volatility at 1.5x gross.

The first diagnosis was that a linear objective over a box constraint corners the
book, and that moving risk into the objective would spread it. Implemented, that
changed nothing: identical weights, correlation 1.0. The objective form is
irrelevant when the feasible set has one vertex.

Measured across a 300-name cross-section at a single rebalance:

| gross | max weight | names | vol | TC |
|---|---|---|---|---|
| 1.5 | 3.0% | 50 | 7.0% | 0.384 |
| 1.5 | 0.5% | 182 | 2.1% | 0.414 |
| 3.0 | 3.0% | 122 | 8.0% | 0.487 |
| 5.0 | 3.0% | 183 | 8.0% | 0.592 |

Lowering the per-name cap diversifies but moves *away* from the volatility
budget. Raising gross does both, which is why market-neutral books run 4–6x.

## Why truncation hid it

The transfer coefficient measured inside the decile-selected names reads 0.725.
The same book measured against the full cross-section alpha reads 0.384.
Truncating to the tails before optimising conceals its own cost from the metric
that would reveal it, which is why the first pass at this missed the problem
entirely.

## Borrow is what makes the comparison honest

Trading cost scales with turnover and is neutral to leverage: doubling gross
doubles both the cost and the P&L it is charged against. Borrow is not — it
accrues daily on the short side.

Charged on short notional rather than gross. Billing full gross at a funding rate
double-counts, since in a dollar-neutral book the longs are funded by the short
proceeds. At 40bp general collateral that is 0.30% of NAV a year at 1.5x gross
against 1.00% at 5x.

Without it, the same sweep reported 0.286 → 0.497. The 0.240 → 0.365 above is
what survives paying for the book. **40bp is a floor, not a central estimate**: a
hard-to-borrow or small-cap short book would not borrow at general collateral,
and this construction would look materially worse there.

## Industry neutrality costs more than it removes

Constraining net industry exposure to ±2% drops Sharpe from 0.365 to 0.229 —
below the baseline. This is the second time neutralisation has destroyed return
in this project: residualising factors against beta and size at the signal level
cut IC from 0.0165 to 0.0062.

The two results agree, and the reading is that at this horizon the edge
substantially *is* the exposure. That is a finding about the signal rather than
about the constraint, and it argues the 59% beta share in the decile
decomposition is not separable from the alpha by construction.

## Limitations

- **Realised volatility overshoots the ex-ante budget**, 8.8% against 8.0%.
  Ledoit–Wolf shrinkage pulls toward lower correlation and under-forecasts
  portfolio volatility for a book of this size. Sharpe is volatility-normalised,
  so the comparison stands, but the risk budget is not being hit.
- **Single seed.** These are one draw from the seed distribution documented in
  the robustness table, where Sharpe ranges 0.09–0.49 across six seeds with
  identical economics. The construction comparison holds the seed fixed so the
  *difference* is meaningful, but the levels should be read against that spread.
- **Capacity is not modelled.** 5x gross with 205 names implies materially larger
  positions per name; no participation-rate cap or market-impact model is applied.
