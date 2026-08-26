---
name: signal-review
description: The questions a portfolio manager asks before allocating to a signal — economic mechanism, decay, capacity, crowding, correlation to the existing book, regime stability, and failure mode. Use when evaluating whether a working signal is worth trading, preparing to present research, deciding between candidate signals, or reviewing someone else's alpha proposal. Complements alpha-research (is it real?) by asking whether it is worth capital.
---

# Signal review

`alpha-research` asks whether a result is real. This asks whether a real result
deserves capital. A signal can be genuine and still be uninvestable — and the
distinction is what separates a research note from a trade.

Answer every question below with a number and a chart, not an adjective.

## 1. Mechanism

- What is the economic hypothesis in one sentence?
- **Who is on the other side, and why do they keep losing?** A signal with no
  identifiable counterparty is a statistical artifact until proven otherwise.
- Why has this not been arbitraged? Acceptable answers: a capacity ceiling, a
  data-acquisition cost, a career-risk or mandate constraint, a balance-sheet
  cost, genuine novelty in the data. Unacceptable: "it is a well-known anomaly."
- What would have to change in the world for it to stop working?

## 2. Strength and shape

- Mean IC, ICIR, and t-statistic on **independent** periods — not overlapping
  observations counted as independent.
- Is the quantile response **monotone**? A high IC with non-monotone deciles
  usually means the signal is driven by the tails, which is a different and less
  robust trade than a smooth cross-sectional tilt.
- What fraction of the return comes from the long side versus the short side? A
  book that only works short has borrow and capacity problems the long-only
  version does not.
- Hit rate and skew. A signal that wins 45% of the time with fat right tails
  needs different sizing from one that wins 55% smoothly.

## 3. Decay

- Plot IC against forward horizon: 1, 5, 21, 63, 126 days. The shape tells you
  the rebalance frequency and the cost budget.
- How fast does the signal itself change? **Selection turnover** — the fraction
  of the selected book that changes per period — bounds how cheaply the signal
  can be traded.
- Is the alpha front-loaded into an event (an earnings date, an index rebalance,
  a filing)? Event-driven alpha has different capacity and different competition
  from continuous alpha.

## 4. Capacity

- At what dollar AUM does the alpha halve? Estimate it: position size as a
  fraction of trailing ADV, an impact model, and the turnover requirement.
- What is the market cap and liquidity profile of where the alpha lives? If it
  is concentrated in the smallest quintile, the paper Sharpe is not the tradable
  Sharpe.
- What is the gross exposure needed to reach a target volatility, and is that
  gross financeable?

## 5. Crowding and correlation

- Correlation of the signal to standard risk factors: market, size, value,
  momentum, quality, low-vol, and the sector map. Report the **exposure**, not
  just the residual.
- What survives after neutralising each? A signal whose alpha disappears under
  sector neutralisation is a sector timing bet, which is a legitimate trade but
  a different one with different capacity.
- Correlation to the **existing book**. A 0.6 Sharpe uncorrelated with what is
  already on is worth more than a 1.0 Sharpe that is 0.8 correlated with it.
  This is usually the question that decides the allocation.
- Is the data source widely licensed? Alpha in a dataset a hundred funds
  subscribe to decays on the vendor's sales schedule.

## 6. Stability

- Performance by year and by regime. Which regime kills it — rising rates, high
  dispersion, low dispersion, high volatility, a momentum crash?
- Dispersion across model seeds and specification choices, if the model has any
  stochastic element.
- Does it survive a different universe, a different exchange, a different
  horizon? Signals that only work in exactly one configuration are usually
  overfit to it.

## 7. Implementation

- **Transfer coefficient**: how much of the theoretical IR does the constrained
  portfolio actually capture? A large gap between `IC × √breadth` and realised IR
  is an implementation problem, not a signal problem, and it is usually fixable.
- Effective breadth, honestly. `√N` assumes independent bets; correlated
  large-caps are nowhere near `N` independent bets, so the theoretical IR is an
  upper bound that is rarely approached.
- What does it cost to trade — commission, spread, impact — as a fraction of the
  gross edge? If costs are the same order as the edge, the signal is a research
  finding, not a strategy.
- Borrow availability and cost for the short book.
- Operational load: data latency, vendor reliability, what happens when the feed
  is late on a rebalance day.

## 8. Failure mode

- How would you know, in live trading, that it has stopped working — before the
  drawdown tells you? Define the monitoring statistic and the threshold up front.
- What is the maximum tolerable drawdown before the position is cut, and does
  the historical drawdown distribution make that plausible?
- What is the correlation of its worst months with the firm's worst months? A
  signal that fails precisely when everything else fails is worth far less than
  its standalone Sharpe suggests.

## Presenting it

A research presentation that survives this review leads with the mechanism, not
the Sharpe. Structure:

1. The hypothesis and the counterparty, in two sentences.
2. What is novel — the data, the construction, or the horizon. If nothing is
   novel, explain why the known thing is mispriced anyway.
3. Signal evidence: IC, monotonicity, decay, with independent-period t-stats.
4. What survives neutralisation, and what that says about the mechanism.
5. Implementation: capacity, cost, transfer coefficient, correlation to the book.
6. The failure mode and the monitoring plan.
7. Everything that did not work, and how many things were tried.

Expect the first question to be "who is losing money to you, and why do they
keep doing it." Have the answer ready.
