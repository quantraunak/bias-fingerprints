# Bias fingerprinting: the framework

**A method for auditing factor research from the outside.** A data-handling
defect is not a scalar amount of inflation but a *direction* in factor space, and
the direction is a property of the defect rather than of the study it damages.
Measure the direction once on a pipeline you own; then test for it in tables you
do not.

The framework has four parts, specified below:

1. **A generator** — one factor study, re-run under alternative data conventions
   with everything the convention does not touch held fixed.
2. **A signature** `delta_b` — the vector of t-statistic shifts a defect induces
   across the factor cross-section, measured on that generator.
3. **An inference model** — an observed table is `tau + delta_b + epsilon`,
   identified through exclusion, within-family contrast and projection rather
   than a fitted classifier.
4. **A validation protocol** — four gates, in order, each of which ends the
   method rather than downgrading it.

`BIAS.md` measures two signatures on the generator this repository ships. This
document is the method they instantiate, and it applies to any defect a correct
pipeline can be made to reproduce.

## The problem it solves

The literature on look-ahead and survivorship bias is prescriptive and
self-directed: here is what the bias is, here is how to avoid it in your own
backtest, here is how to test your own pipeline by shifting a feature a bar
forward. All of it assumes you control the code.

The situation that actually recurs is the opposite. An allocator reads a
manager's factor table. A researcher reads a published paper. A desk evaluates a
vendor's signal. In each case the biased artifact is visible and the pipeline
that produced it is not. Nothing in the literature answers *which bug does this
table look like*, and that is the question being asked.

## The generator, and the two signatures measured on it

The generator fixes a factor registry, an evaluation window, a forward-return
definition and a walk-forward split, then produces the resulting table of factor
t-statistics under any data convention, sharing every component the convention
does not touch. Prices, tradability screen, sector neutralisation,
rank-normalisation and evaluation code are common across arms; the arm changes
one construction decision. Without it the effect of a defect can only be argued;
with it the effect is a difference between two runs of the same code.

Both signatures below come from `reports/pit_vs_naive.csv` and
`reports/survivorship.csv`, as t-statistic shifts from the correct point-in-time
baseline.

**Dating bias** (joining fundamentals on period end):

| factor | shift |
|---|---|
| accruals | **+1.77** |
| roe | +1.07 |
| cash_flow_yield | +1.02 |
| earnings_yield | +0.98 |
| gross_profitability | +0.49 |
| ...seven more fundamentals | +0.15 to +0.44 |
| **eleven price-only factors** | **exactly 0.000** |
| turnover_1m | −0.11 |

**Survivorship bias** (current index membership applied to history):

| factor | shift |
|---|---|
| amihud_illiquidity | **+3.91** |
| gross_profitability | +1.17 |
| ...mid-pack | −0.9 to +0.3 |
| roe | −1.24 |
| earnings_yield | −1.28 |
| vol_60d | −1.41 |
| operating_margin | −1.88 |
| idio_vol_60d | −2.02 |
| turnover_1m | −2.05 |
| **factors exactly unchanged** | **none** |

Two properties make the inversion plausible.

**The signatures are nearly orthogonal.** Correlation across the 22 factors is
**−0.088**. They are not two versions of "results look better"; they are
different directions in factor space.

**They move value in opposite directions.** Dating inflates earnings yield and
ROE (+0.98, +1.07); survivorship deflates them (−1.28, −1.24). A table cannot be
explained by both.

**One property is a hard constraint rather than a tendency.** Under dating bias,
eleven price-and-volume factors are unchanged to machine precision -- not
approximately, exactly, because they never read a filed figure. So an
anomalously strong illiquidity result *cannot* be dating bias. That is a logical
exclusion, not a probabilistic one, and it does most of the discriminating work.

## Why the inference is not a learning problem

The obvious framing -- learn a classifier mapping factor tables to bias labels --
does not survive contact with the sample size.

The out-of-sample window is 2014 to 2026. Non-overlapping two-year windows give
six independent observations. Three bias conditions gives eighteen labelled
examples for a twenty-two dimensional feature vector. Any classifier fit on that
is fitting noise, and cross-validation over overlapping windows would hide it
rather than reveal it.

The sound formulation is different: **the signature is measured once from a
pipeline we control, and then used as a known quantity in a hypothesis test.**
There is nothing to learn. There is something to estimate -- the signature's
sampling variability -- and something to test.

## The inference problem, stated

An observed table is a vector of t-statistics `t` over the factor registry.
Model it as

    t = tau + delta_b + epsilon

where `tau` is the study's true factor performance, `delta_b` is the signature of
bias `b` (measured, with `delta_none = 0`), and `epsilon` is sampling noise.

`tau` is unknown and twenty-two dimensional, so `b` is not identified without
further structure. Three ways to get it, in increasing order of assumption:

1. **Exclusion.** Dating bias cannot move a price-only factor. If the
   price-only block carries the study's strongest results, dating bias is ruled
   out regardless of anything else.
2. **Contrast.** Assume `tau` is exchangeable within factor families -- value
   factors share a true level, momentum factors share a different one. Then
   *within-family* deviations are attributable to `delta_b`, and the family
   means absorb `tau`. This is the workhorse assumption and it is testable
   against the factor-zoo literature.
3. **Projection.** Score a table by its projection onto each signature after
   removing the family means. Under the null that `tau` is exchangeable, the
   projection has a computable reference distribution, so the output is a
   p-value rather than a label.

The output should be a posterior or a set of p-values over `{none, dating,
survivorship, both}`, never a bare classification. A diagnostic that says
"survivorship, confidence 0.9" without a calibration behind it is worse than no
diagnostic.

## The validation protocol

Four measurements, in order. Each is a gate, and failing one ends the method
rather than downgrading its claims. Nothing is reported against a real table
until all four have passed.

1. **Signature stability.** Everything rests on `delta_b` being a property of
   the bias rather than of the 2014-2026 sample. Re-measure it on disjoint
   subperiods and on random half-universes. If `amihud_illiquidity` does not
   move sharply positive under survivorship in *every* subsample, there is no
   signature and the project ends here.
2. **Signature uncertainty.** A point estimate of `delta_b` cannot support a
   hypothesis test. Bootstrap over dates to get a covariance, and report the
   signature with error bars.
3. **Separability under realistic noise.** Simulate tables from known `tau`
   priors, apply each bias, add sampling noise at the level a twelve-year study
   actually has, and measure the confusion matrix. If dating and survivorship
   are not separable at realistic noise, report that and stop.
4. **Only then, application.** Run the diagnostic on published factor tables and
   report what it says, with its uncertainty.

Step 3 is where the labelled data comes from, and it is free: biased panels are
generated by the pipeline itself, so there is no annotation and no external
dataset. That is the property that makes this feasible for one person.

## Novelty and scope

The individual biases are old -- Banz and Breen (1986) for look-ahead, and
survivorship is textbook. `BIAS.md` says so in its first paragraph. What does
not appear in the literature is the inversion: treating the *pattern* of
distortion as a diagnostic signature and inferring the bug from the reported
cross-section.

The contribution is a method built on known facts rather than a new fact, and it
is best described as a diagnostic built on measured signatures. Its value is
practical: it is a tool someone applies to a table they cannot otherwise check,
and it extends to any defect a correct pipeline can reproduce. Delisting-return
assumptions, restatement handling and index-reconstitution timing are each a
convention this generator already has an opinion about, which means each already
has a signature waiting to be measured.

A literature search focused on prescriptive and self-diagnostic work found
nothing doing this inversion. That search was not exhaustive and must be
repeated properly before any claim of priority.

## Scope conditions

The framework transfers wherever a correct pipeline can be re-run under a
defective convention. The two signatures measured here transfer less far.

- **One index, one country, 2010-2026.** The signatures are S&P 500 artifacts
  until shown otherwise, and the whole method assumes the target study uses a
  comparable universe. Re-measuring them elsewhere is a run of the generator,
  not a new study.
- **Twenty-two factors.** A study reporting five factors gives a five
  dimensional observation, and the exclusion constraint may not apply at all if
  none of them are price-only.
- **Two biases.** Real studies contain several bugs at once, and the signatures
  may not compose additively. `BIAS.md` already notes the two arms are not
  additively decomposable because the survivorship arm rebuilds forward returns.
- **Membership spells come from a wiki revision history**, not CRSP. The
  survivorship signature inherits whatever that approximation gets wrong.
