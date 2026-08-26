# Pre-registration: economic links from filing text

Written before the extraction corpus exists, so the design cannot be
reverse-engineered from the result. Committed ahead of the data on purpose —
the git history is the evidence that it was.

## Hypothesis

**Returns propagate along supply-chain links with a delay, because investors
do not track links that no database makes visible.**

Cohen and Frazzini (2008) showed that when a firm's *customer* has a good month,
the firm itself tends to have a good month roughly 30 days later. Their finding
rests on the customer-segment data mandated by ASC 280, which only forces
disclosure when a customer exceeds 10% of revenue — a sparse, high-threshold
sample of the real link structure.

The claim tested here is narrower and, if it holds, more interesting: the effect
should be **stronger on links that are disclosed in prose but absent from
structured data**, because those are the links no screen surfaces and no risk
model neutralises.

## Mechanism

- **Who is on the other side.** Investors with finite attention. Following a
  semiconductor firm means following its foundry, its three largest OEM
  customers, and their end markets. Analysts are organised by sector, so a
  link that crosses a sector boundary falls between two desks and is covered by
  neither.
- **Why it is not arbitraged.** The link structure is not in Compustat, not in
  a factor model, and not on a screen. Constructing it requires reading ten
  thousand filings, which until recently cost more than the alpha was worth.
  That cost is the barrier, and it is exactly the barrier that has just fallen.
- **What would kill it.** Vendors now sell supply-chain graphs (FactSet Revere,
  Bloomberg SPLC). To the extent the tradable links are the ones those vendors
  already cover, the edge should be gone. The prediction is that alpha
  concentrates in links those products do *not* carry: second-order links,
  smaller counterparties, and relationships described qualitatively rather than
  quantified.

## Falsification, written in advance

The hypothesis is rejected if any of the following holds on out-of-sample data:

| Test | Rejection threshold |
|---|---|
| Link-lagged return IC | `t < 2.0` over independent months |
| Decile monotonicity | non-monotone across the middle deciles |
| Second-order vs first-order | second-order IC not greater than first-order |
| Placebo (shuffled edges) | real IC within one standard error of placebo |
| Survives beta + sector neutralisation | alpha `t < 2.0` after neutralising |

The **placebo test is the one that matters**. Randomly rewiring the graph while
holding degree distribution fixed must destroy the signal. If a shuffled graph
predicts returns about as well as the real one, the result is a mechanical
artifact of the return panel — cross-sectional autocorrelation, sector
clustering — and not economic-link propagation at all.

## Design, fixed before the data

- **Universe.** Signal is computed for S&P 500 point-in-time members. A
  *counterparty* may be any US-listed name with a price series — restricting
  counterparties to index members would discard most of the graph.
- **Edge timing.** An edge becomes visible on the **filing date** of the 10-K
  that disclosed it, never the fiscal period it covers. Edges persist until
  superseded by a later filing from the same issuer, and expire after 3 years
  without reconfirmation.
- **Signal.** For each firm on each date, the edge-weighted average return of
  its counterparties over the prior month, lagged one day. Computed separately
  for first-order and second-order neighbourhoods.
- **Label.** Cross-sectional rank of the 21-day forward return, matching the
  existing pipeline so the two signals are directly comparable.
- **Validation.** The existing purged and embargoed walk-forward. No new
  evaluation code, so the standard cannot drift between signals.
- **Out-of-sample.** 2014-01-31 onward, the same window already used. It is
  looked at once, at the end.

## Extraction quality is a measured quantity, not an assumption

Extraction runs on a local 8B model, so its error rate is a first-class part of
the result rather than a footnote. Before any signal is computed:

- A hand-checked sample establishes precision and recall against the model.
- Every extracted claim carries a verbatim quote, checked as a literal substring
  of the source. Claims failing that check are discarded and **counted**, so
  hallucination is a reported rate.
- The edge count that survives resolution into the price universe is reported
  before any return is computed. If in-universe edge density is too low, the
  study stops there and says so.

## Specifications tried

Maintained in `docs/SPECIFICATIONS.md`, appended to as the research runs, so the
multiple-testing correction at the end has an honest denominator. Given a search
over `N` specifications, the reported threshold is a deflated Sharpe ratio
rather than a raw t-statistic.

## What a negative result looks like

If link-lagged returns carry no information once the placebo is subtracted, that
is the finding, and it is reported with the same prominence a positive result
would get. The infrastructure — a point-in-time economic-link graph built from
unstructured filings, with measured extraction error — stands on its own
regardless of which way the test comes out.
