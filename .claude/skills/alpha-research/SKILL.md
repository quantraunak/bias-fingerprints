---
name: alpha-research
description: The protocol for proposing, testing, and reporting a new alpha signal — hypothesis before data, point-in-time discipline, purged validation, and multiple-testing correction. Use when designing a new factor or signal, when starting a research question ("does X predict returns"), before running a backtest on a new idea, or when deciding whether a measured result is real. Not for portfolio construction or execution questions.
---

# Alpha research protocol

The default failure mode in signal research is not a coding bug. It is running
many specifications, reporting the best one, and mistaking a search result for a
measurement. Everything below exists to make that failure visible.

## 1. Write the hypothesis before touching the data

State, in writing, before the first query:

- **The mechanism.** What causes the mispricing? "Value works" is not a
  mechanism. "Investors extrapolate recent growth and underweight mean reversion
  in capital intensity" is.
- **Who is on the other side.** Every dollar of alpha is somebody's loss. Name
  them: a forced seller, a constrained index fund, a retail flow, an inattentive
  analyst. If no plausible counterparty exists, the edge is probably a data
  artifact.
- **Why it survives.** If the mechanism has been public since 1993, explain what
  stops it from being arbitraged — a capacity limit, a career-risk constraint, a
  balance-sheet cost, a data-acquisition cost.
- **The falsification.** What result would make you drop this? Write the number
  down now, not after seeing the output.

A hypothesis that cannot be falsified in advance produces a result that cannot
be believed afterward.

## 2. Point-in-time or it does not count

Every input carries the timestamp at which it was **knowable**, not the
timestamp it describes.

| Input | Correct key | The trap |
|---|---|---|
| Fundamentals | filing date (`filed`, not `period_end`) | joining on period end leaks 30-90 days |
| Estimates | announcement datetime | revision history is restated in most vendor files |
| Index membership | membership spells with entry and exit | current membership backfilled = survivorship |
| Corporate actions | ex-date | adjusting the whole history retroactively |
| Restatements | first reported value | vendors overwrite with the revised number |
| Ratings, classifications | effective date | GICS reclassifications are backfilled |

Restatements deserve special care: keep the **earliest** reported figure, because
that is the number the market saw. The revised one is information from the
future.

## 3. Validation that respects overlapping labels

For a signal with an `h`-day forward return label:

- **Purge.** Training data must end `h` days before the prediction date, or the
  label window overlaps the prediction period.
- **Embargo.** Add a further gap (commonly `h` again) after the purge, because
  serial correlation in features leaks across the boundary even without label
  overlap.
- **Uniqueness weights.** `n` overlapping daily observations of an `h`-day label
  are worth roughly `n/h` independent ones. Weight by average uniqueness or thin
  the sample; unweighted fitting overstates the effective sample by an order of
  magnitude and every t-statistic downstream inherits the error.
- **Expanding, not rolling, unless you can justify forgetting.** A rolling window
  discards information; use it only if you believe the relationship is
  non-stationary, and say so.

Reference: López de Prado, *Advances in Financial Machine Learning*, ch. 7.

## 4. Count every specification you tried

Keep a running log of every variant: each feature set, each horizon, each
universe filter, each hyperparameter sweep, each smoothing window. The count
matters more than any single result.

**Thresholds, given the search:**

- A single pre-registered test: `|t| > 2.0` is the conventional bar.
- A new factor claim in a literature that has already mined this data:
  **`|t| > 3.0`** (Harvey, Liu & Zhu 2016). Most published anomalies do not
  clear this out of sample.
- After an explicit search of `N` specifications, apply a **deflated Sharpe
  ratio** (Bailey & López de Prado 2014), which discounts the observed Sharpe by
  the number of trials and the non-normality of returns. Reporting the maximum of
  20 trials as though it were one test inflates the t-statistic by roughly the
  expected maximum of 20 draws.

If you cannot reconstruct how many things you tried, the result is not
interpretable and the honest report is "exploratory, uncorrected".

## 5. Hold out-of-sample genuinely out

- Choose the OOS split **before** looking at results, and touch it once.
- Every time you look at OOS and go back to change the model, OOS has become
  training data. Track how many times this has happened and report it.
- Prefer a **hold-out period** over a hold-out random split: financial data is
  serially correlated and regime-dependent, so a random split leaks.

## 6. Report the search, not the winner

A result section should contain:

- The full specification table, including the variants that failed. Failures are
  evidence about the hypothesis; hiding them destroys the reader's ability to
  calibrate.
- **Seed and ordering dispersion.** Re-fit under several random seeds with
  identical economics. If the headline metric moves materially across seeds, it
  is a draw from a distribution — report mean and spread, never a single draw.
- **Sub-period results.** One number over a long window hides regime structure.
  Report by year or by regime.
- Sensitivity to the parameters you chose freely — the horizon, the smoothing
  window, the universe cut.

## Red flags to check before believing your own result

- Sharpe above ~1.5 on daily large-cap equity data with free data sources
- An equity curve that is smooth relative to its drawdown profile
- Performance concentrated in a handful of days, names, or one regime
- IC that is high but decile returns that are not monotone
- Results that improve when you add costs (a sign of an inverted sign somewhere)
- A metric that changes materially when you reorder the input columns
- Coverage or observation counts that do not match the calendar span

## The standard to hold

Publishing a negative result with the search fully disclosed is a stronger
research contribution than a positive result with the search hidden — and it is
the one a reviewer can actually verify. Write it so a skeptical reader with the
code can reproduce every number and find the places you were uncertain.
