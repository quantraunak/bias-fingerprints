# Two Bugs, Two Fingerprints: Identifying Data-Handling Errors from the Cross-Section of Reported Factor Performance

**Raunak Sood**

---

## Abstract

Look-ahead and survivorship bias are known to inflate backtested performance,
and the standard treatment of both is prescriptive: here is the error, here is
how to avoid it in your own pipeline. That framing assumes the reader controls
the code. The situation that recurs in practice is the opposite — an allocator
reads a manager's factor table, a researcher reads a published result, a desk
evaluates a vendor's signal — and the biased artifact is visible while the
pipeline that produced it is not.

We measure what two common data-handling errors do to a twenty-two factor
cross-sectional study built on point-in-time US equity data, holding everything
else fixed. Joining fundamentals on fiscal period end rather than filing date
inflates mean information coefficient by **59%** among the eleven factors that
read a filed figure, and manufactures **four** t-statistics above 2.0 from
factors that are insignificant under correct dating. Conditioning the universe on
current index membership does not uniformly inflate: mean IC *falls* by 0.0043,
while `amihud_illiquidity` flips sign and becomes significant (t: −1.11 → **+2.80**)
and `turnover_1m` becomes significant with the wrong sign (t: −0.48 → **−2.53**).

The two distortions are nearly orthogonal across the factor cross-section
(correlation **−0.088**) and move value factors in opposite directions. Under
dating bias, eleven price-and-volume factors are unchanged to machine precision —
exactly, not approximately, because they never read a filed figure. We argue that
this structure makes the *pattern* of reported factor performance diagnostic of
the underlying error, and we set out the conditions under which that inversion
would be valid. Those conditions are not yet tested, and we say so.

**Keywords:** point-in-time data, look-ahead bias, survivorship bias, factor
replication, research forensics

---

## 1. Introduction

That look-ahead bias inflates backtested returns has been known since Banz and
Breen (1986). That survivorship bias does the same is textbook. Neither
observation is a contribution, and this paper does not claim otherwise.

What is less established is *how* each error distorts a factor study — which
factors move, by how much, and in which direction. This matters because the
consumer of factor research is rarely in a position to audit the pipeline. A
Sharpe ratio carries no information about which bug produced it. A cross-section
of twenty-two factor t-statistics might.

This paper makes three contributions, in decreasing order of confidence:

1. **A controlled measurement.** We quantify what each of two errors does to the
   same factor study, with the same prices, universe, tradability screen and
   forward returns, changing one thing at a time. Eleven factors that cannot
   respond to the dating treatment serve as controls and confirm the harness
   introduces nothing: their IC is identical to machine precision across all
   three dating conventions.

2. **The observation that the distortions are structurally distinct.** They are
   nearly orthogonal, they disagree in sign on value factors, and one of them
   leaves an entire block of factors provably untouched.

3. **A proposed inversion, with its conditions.** If the distortion patterns are
   stable properties of the errors rather than artifacts of our sample, they
   support inference from a reported table to the error that produced it. We set
   out what would have to be true, and report that we have not yet tested it.

The third is a proposal. We are explicit about that separation throughout,
because the first two are measurements and the third is not.

---

## 2. Data and construction

**Universe.** S&P 500 point-in-time membership, reconstructed as dated
constituent spells from the revision history of a public reference page. This is
a free approximation to a point-in-time constituent file and is not CRSP; §6
treats the consequences.

**Prices.** Daily adjusted closes, 2010–2026, with a tradability screen on price,
volume and membership.

**Fundamentals.** SEC XBRL company facts, keyed on the date each figure was
filed. Restatements resolve to the earliest filing. Cumulative cash-flow spans
are differenced to true quarters. Rows whose filing date is an XBRL-adoption
artifact — a period ending long before the filing that first reports it — are
flagged rather than silently trusted.

**Factors.** Twenty-two published cross-sectional factors, each signed to its
published direction, sector-neutralised and rank-normalised. Eleven read a filed
fundamental. Eleven use only price and volume.

**Evaluation.** Information coefficient against 21-day forward returns,
out-of-sample from 2014-01-01, with a purged and embargoed walk-forward. The
evaluation code is shared across every condition below, so the standard cannot
drift between arms.

---

## 3. Bug 1: joining fundamentals on fiscal period end

A quarter ending 31 March is not public on 31 March. It becomes public when the
10-Q is filed, a median of 34 days later. Joining on period end hands the
backtest a month of information nobody had.

We compare three conventions, holding everything else fixed:

| convention | assumption |
|---|---|
| **PIT** | visible the day the filing landed — correct |
| **LAG45** | every company files 45 days after quarter end — common practice |
| **NAIVE** | visible the instant the quarter closed — the error |

### 3.1 The controls hold exactly

Eleven factors read no filed figure. Across all three conventions their IC is
identical, with maximum absolute drift **0.0e+00** — exact equality in floating
point, not a small number.

This is what licenses the rest of the section. It establishes that nothing in
the harness responds to the filing calendar except the factors that must.

One factor required care. `turnover_1m` is built from price and volume but
divides by shares outstanding, so it inherits the filing calendar through its
denominator. It is the only price-family factor that does. We determine dating
sensitivity by measurement and check it against a declared set, so a factor that
quietly acquires a filed input is surfaced rather than hidden among the controls.

### 3.2 What the error is worth

Among the eleven affected factors:

| convention | mean IC inflation | look better | cross t = 2 spuriously |
|---|---|---|---|
| fixed 45-day lag | +9% of true IC | 8 of 11 | **0 of 11** |
| period-end join | **+59% of true IC** | 10 of 11 | **4 of 11** |

The four manufactured significances are `earnings_yield`, `cash_flow_yield`,
`roe` and `accruals` — each insignificant under correct dating and above t = 2.0
under the period-end join.

The comparison between the two rows is the practical result. A fixed 45-day lag
is wrong — it is right on average and wrong for every filing that arrives later
than that — but it manufactures no false discoveries in this sample. The
period-end join manufactures four. The distinction between "approximately right"
and "wrong" is the distinction between a bias and a bug.

Inflation is summarised over affected factors only. Averaging in eleven controls
that cannot move by construction would halve the reported figure for free.

---

## 4. Bug 2: conditioning the universe on current membership

Download today's constituents, pull their price history, backtest. The resulting
panel conditions on index membership as of the end of the sample.

### 4.1 The dominant channel is not the one usually named

Survivorship is normally described as dropping losers — the acquired, the
bankrupt, the demoted. In index research that is the smaller half.

Daily averages, 2010–2015:

| | names/day | median $ volume | median market cap |
|---|---|---|---|
| genuine members | 386 | $95M | $10.1B |
| wrongly **included** — not yet admitted | **203** | **$28M** | **$3.5B** |
| wrongly **excluded** — later deleted | 88 | $54M | $5.3B |

The biased panel wrongly includes more than twice as many names as it wrongly
excludes, and the added names are roughly a third the size and a third the
liquidity of a genuine constituent. These are companies that were small early in
the sample and grew enough to be admitted later.

Conditioning on current membership is therefore not principally a survival
filter. It is a **forward-looking growth filter aimed at the small and illiquid
cohort**.

### 4.2 The distortion relocates rather than inflates

Mean IC change across all twenty-two factors is **−0.00432**, and only 8 of 22
improve. Two factors cross |t| = 2:

| factor | PIT IC | biased IC | PIT t | biased t |
|---|---|---|---|---|
| `amihud_illiquidity` | −0.0105 | **+0.0224** | −1.11 | **+2.80** |
| `turnover_1m` | −0.0047 | −0.0232 | −0.48 | **−2.53** |
| `idio_vol_60d` | +0.0059 | −0.0179 | 0.49 | −1.53 |
| `vol_60d` | −0.0013 | −0.0234 | −0.08 | −1.49 |

Illiquidity flips sign and becomes significant. In a survivorship-biased panel,
being illiquid predicts returns — because the illiquid names in today's index are
precisely those that were small years ago and then grew into it. Turnover becomes
significant with the wrong sign. The low-volatility anomaly inverts.

A biased panel says: buy the illiquid, high-turnover, high-volatility names. That
is not a risk premium. It is a description of a company on its way into the index,
read backwards.

---

## 5. The two fingerprints

The distortions do not overlap.

| | period-end join | current-membership universe |
|---|---|---|
| price/volume factors | **untouched, exactly** | heavily distorted |
| value and quality | inflated; 4 false positives | mixed, mostly weakened |
| liquidity and volatility | untouched | sign flips, false positives |
| direction | uniform inflation | relocation, not inflation |

Quantitatively, the two shift vectors across the twenty-two factors correlate at
**−0.088**. They are not two versions of "results look better"; they are close to
independent directions in factor space. They also disagree in sign on value:
dating inflates `earnings_yield` and `roe` by +0.98 and +1.07 t-units,
survivorship deflates them by −1.28 and −1.24.

One property is a hard constraint rather than a tendency. Under dating bias the
eleven price-only factors are unchanged **to machine precision**, because they
never read a filed figure. An anomalously strong illiquidity result therefore
cannot be produced by dating bias, whatever else is true of the study.

### 5.1 The proposed inversion, and what it would require

This structure suggests reading the implication backwards: a study with a
suspiciously strong illiquidity premium and a dead low-volatility anomaly has a
universe problem; a study whose value and quality factors clear t = 2 while its
price factors look ordinary has a dating problem.

We state plainly that this inversion is **proposed and not tested**. Four
conditions would have to hold, and each is a gate rather than a caveat:

1. **Signature stability.** The shift vectors must be properties of the errors,
   not of the 2014–2026 sample. This requires re-measurement on disjoint
   subperiods and random half-universes.
2. **Signature uncertainty.** A point estimate cannot support a test. The
   sampling covariance of each shift vector must be estimated.
3. **Separability under realistic noise.** With a study's true factor
   performance unknown and twenty-two dimensional, identification requires
   structure — exclusion from the exact-zero constraint, and within-family
   contrasts that absorb the unknown level. Whether the two errors remain
   separable at the noise level of a twelve-year study is an empirical question.
4. **Calibration before application.** Any output must be a p-value or a
   posterior with a reference distribution behind it, never a bare label.

We note that labelled data for (3) is free: biased panels are generated by the
pipeline itself, so the inversion can be validated at scale without any external
dataset or annotation.

---

## 6. Limitations

- **One index, one country, 2010–2026.** The fingerprints are measured on the
  S&P 500 and should not be assumed to transfer to small caps or non-US markets.
- **Membership spells are reconstructed from a public wiki revision history**,
  not CRSP. The survivorship arm inherits whatever that approximation gets wrong.
- **Fundamental coverage begins in 2010** because of the XBRL mandate phase-in,
  so the dating arm cannot speak to the pre-XBRL era.
- **The two arms are not additively decomposable.** The survivorship arm rebuilds
  forward returns per universe, because a biased study never observes the returns
  of the names it excludes. Real studies contain several errors at once and the
  signatures may not compose.
- **Twenty-two factors.** A study reporting five gives a five-dimensional
  observation, and the exclusion constraint does not apply at all if none of them
  are price-only.
- **The inversion in §5.1 is untested.** Sections 3 and 4 are measurements;
  §5.1 is a proposal.

---

## 7. Conclusion

Two common data-handling errors distort a factor study in different and
identifiable ways. The period-end join inflates fundamental factors uniformly by
59% of true IC and manufactures four false discoveries while leaving price
factors provably untouched. Current-membership conditioning relocates rather than
inflates, flipping illiquidity to significance and inverting the low-volatility
anomaly, through a channel that is better described as a forward-looking growth
filter than as a survival filter.

Neither distortion is visible in a Sharpe ratio. Both are visible in the
cross-section of what a study reports. Whether that visibility supports reliable
inference from a table to its underlying error is the open question, and §5.1
states what would have to be established before it could be claimed.

---

## Reproduction

All results reproduce from a public repository with `make bias`. The dating arm
is `scripts/pit_vs_naive.py`; the survivorship arm is `scripts/survivorship.py`.
Per-factor outputs are written to `reports/pit_vs_naive.csv` and
`reports/survivorship.csv`. Prices are fetched per user rather than
redistributed. The point-in-time fundamentals layer and the membership spells are
released under CC BY 4.0.

## References

Banz, R. and Breen, W. (1986). Sample-Dependent Results Using Accounting and
Market Data: Some Evidence. *Journal of Finance*.
