# Two calibrated signatures

Every factor in this repository, scored three ways under a wrong filing calendar
and two ways under a wrong universe. This is the measurement half of the
framework in `FINGERPRINT.md`: the generator produces one factor study under
alternative data conventions, and the difference between two arms is the
signature of the convention that changed.

That look-ahead inflates results has been known since Banz and Breen (1986). What
is measured here is the *shape* of each distortion — which factors move, by how
much, in which direction, and which cannot move at all — because the shape is
what a diagnostic can test for in a study whose pipeline is not available.

All figures are out-of-sample from 2014-01-01, 21-day forward returns, S&P 500
point-in-time membership, sector-neutral rank-normalised scores. Reproduce with
`make bias`.

---

## Signature 1: fundamentals dated on fiscal period end

A quarter ending 31 March is not public on 31 March. It becomes public when the
10-Q is filed, a median of 34 days later. Joining on period end hands the
backtest a month of information nobody had.

Three conventions, everything else held fixed:

| | what it assumes |
|---|---|
| **PIT** | visible the day the filing landed — the truth |
| **LAG45** | every company files 45 days after quarter end — common practice |
| **NAIVE** | visible the instant the quarter closed — the bug |

### The controls hold exactly

Eleven of the twenty-two factors read no filed figure. Their IC is **identical to
machine precision** across all three conventions — maximum drift `0.0e+00`. That
is what licenses the rest of this document: nothing in the pipeline responds to
the filing calendar except the factors that should.

The twelfth expected mover is the one worth naming. `turnover_1m` is a
price-and-volume factor, but it divides volume by shares outstanding, so it
inherits the filing calendar through its denominator. Dating sensitivity here is
measured and checked against a declared set rather than assumed, so a factor that
quietly acquires a filed input shows up rather than hiding among the controls.

### What the bug is worth

| convention | mean IC inflation | look better | cross t = 2 spuriously |
|---|---|---|---|
| fixed 45-day lag | +9% of true IC | 8 of 11 | **0 of 11** |
| period-end join | **+59% of true IC** | 10 of 11 | **4 of 11** |

The four manufactured significances are `earnings_yield`, `cash_flow_yield`,
`roe` and `accruals`. Accruals is the worst case: **t = 1.23 becomes t = 2.99**,
which is the difference between a null result and a publishable one.

The 45-day row is the honest part. The shortcut most practitioners actually use
is approximately fine — it inflates a little and promotes nothing. The damage is
specific to one join, which means the finding has a precise target rather than a
general warning.

### The bug is decaying

| era | PIT IC | naive IC | inflation | % of true IC |
|---|---|---|---|---|
| 2010–2015 | 0.00617 | 0.01212 | +0.00595 | **61%** |
| 2016–2020 | 0.00271 | 0.00737 | +0.00466 | **45%** |
| 2021–2026 | 0.01084 | 0.01455 | +0.00370 | **33%** |

Monotone in both the ratio and the absolute inflation, which matters because the
PIT denominator moves between eras and could otherwise drive the ratio on its
own. The mechanism is the filing calendar: median lag fell from 35 days in 2010
to 25 in 2025, and the fraction of company-quarters filed more than 90 days after
period end collapsed from 22% to 1.6% as accelerated-filer deadlines took hold.

The implication is uncomfortable. **The same one-line bug was worth roughly twice
as much in 2010–2015 as it is today**, so anomaly research is contaminated
unevenly by vintage, with older work more exposed. Multiple-testing corrections
do not address this: the entire t-hurdle debate assumes the t-statistics are
correctly measured.

---

## Signature 2: a universe conditioned on current membership

Download today's 500 constituents, pull their price history, backtest. The
resulting panel conditions on index membership *as of the end of the sample*.

### The dominant channel is not the one everyone names

Survivorship is usually described as dropping the losers — the acquired, the
bankrupt, the demoted. In index research that is the smaller half of the problem.

Daily averages, 2010–2015:

| | names/day | median $ volume | median market cap |
|---|---|---|---|
| genuine index members | 386 | $95M | $10.1B |
| wrongly **included** — not yet admitted | **203** | **$28M** | **$3.5B** |
| wrongly **excluded** — later deleted | 88 | $54M | $5.3B |

The biased panel wrongly includes more than twice as many names as it wrongly
excludes, and the ones it adds are a third the size and a third the liquidity of
a real constituent. These are companies that were small in 2012 and grew enough
to be admitted later. Conditioning on current membership is therefore not mainly
a survival filter — it is a **forward-looking growth filter aimed squarely at the
small and illiquid cohort**.

### The fingerprint

Mean IC change across all 22 factors is **−0.00432**, and only 8 of 22 look
better. Survivorship does not uniformly inflate. It relocates:

| factor | PIT IC | survivor IC | PIT t | survivor t |
|---|---|---|---|---|
| `amihud_illiquidity` | −0.01047 | **+0.02238** | −1.11 | **+2.80** |
| `turnover_1m` | −0.00472 | −0.02316 | −0.48 | **−2.53** |
| `idio_vol_60d` | +0.00593 | −0.01793 | 0.49 | −1.53 |
| `vol_60d` | −0.00130 | −0.02336 | −0.08 | −1.49 |

Illiquidity **flips sign and becomes significant at t = 2.80**. In a
survivorship-biased panel, being illiquid predicts returns — because the illiquid
names in today's index are precisely the ones that were small years ago and then
grew into it. Turnover becomes significant with the wrong sign. The low-volatility
anomaly inverts.

A biased panel says: buy the illiquid, high-turnover, high-volatility names. That
is not a risk premium. It is a description of a company on its way into the S&P
500, read backwards.

---

## Separability of the two signatures

The distortions do not overlap.

| | period-end join | current-membership universe |
|---|---|---|
| price/volume factors | **untouched, exactly** | heavily distorted |
| value and quality | inflated, 4 false positives | mixed, mostly weakened |
| liquidity and volatility | untouched | sign flips, false positives |
| direction | uniform inflation | relocation, not inflation |

So the pattern is diagnostic. A study with a suspiciously strong illiquidity
premium and a dead low-volatility anomaly has a universe problem. A study whose
value and quality factors clear t = 2 while its momentum factors look ordinary
has a dating problem. Neither is visible from a Sharpe ratio, and both are
visible from the cross-section of what the study claims to have found.

`FINGERPRINT.md` states the inference model that turns that observation into a
test, and the four gates that have to pass before it may report anything.

## Scope conditions

- One index, one country, 2010–2026. The signatures are measured on the S&P
  500 and should not be assumed to transfer to small caps or non-US markets.
- Membership spells are reconstructed from the revision history of a Wikipedia
  page, which is the best free approximation of a point-in-time constituent file
  and is not CRSP.
- Fundamental coverage begins in 2010 for the reason documented above, so the
  dating arm cannot speak to the pre-XBRL era at all.
- The survivorship arm rebuilds forward returns per universe, because a biased
  study never sees the returns of the names it excludes. That is deliberate, and
  it means the two arms are not additively decomposable.
