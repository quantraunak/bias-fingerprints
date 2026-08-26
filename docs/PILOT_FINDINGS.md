# Pilot findings: can a supply-chain graph be built from 10-K text?

Measured before committing to the full extraction run, as the pre-registration
requires. The answer changed the design.

## The constraint: large caps do not name counterparties

Random sample of 300 filings from 107 S&P 500 issuers, 2010-2025. Counting
capitalised names appearing near relationship language and resolving to a listed
ticker:

| | |
|---|---|
| Filings with at least one resolvable named counterparty | **41.7%** |
| Named counterparties per filing | mean **0.62**, median **0** |
| Distribution | 175 filings with 0, 86 with 1, 27 with 2, 12 with 3+ |

The median large-cap 10-K names **zero** counterparties. Two semiconductor firms
with famously deep supply chains illustrate it:

- **Applied Materials** (2016): zero. Customers described only as "manufacturers
  of semiconductor chips" and "third party providers".
- **Analog Devices** (2016): zero. Customers given only as end-market shares —
  "Industrial 44%, Automotive 16%, Consumer 20%".
- **Caterpillar** (2016): names Generac, Kohler and Aggreko — all *competitors*.
  Everything else is Cat Financial, its own subsidiary.

This is not an extraction failure. The information is not in the document.
ASC 280 compels disclosure of customers above 10% of revenue but does not compel
naming them, and a company with no customer above 10% discloses nothing at all —
which describes most of the S&P 500.

## The model's yield is inflated by product names

Local 8B extraction over the first 25 filings returned 2.12 claims per filing,
well above the 0.62 ceiling the regex scan implies. The gap is false positives:
the most-named "counterparties" were **App Store, Mac App Store, iTunes Store,
iBooks Store, TV App Store** — Apple's own products, which the prompt explicitly
excludes and the 8B model emitted anyway. They resolve to no ticker and die at
the resolution stage, so they cost coverage rather than corrupting the graph,
but the effective yield is close to the regex ceiling rather than above it.

## Where the disclosure actually lives

Named customers appear where concentration is *material* — suppliers selling
into a few large OEMs. Same measurement, on 15 such firms:

| | large-cap S&P 500 | concentrated suppliers |
|---|---|---|
| Named counterparties per filing | 0.62 | **1.50** |
| Filings with at least one | 42% | **65%** |

And the edges are correct on inspection: Amkor names Intel, Micron, STMicro and
TI; Jabil names Cisco, Ericsson, GE, HP and Dell; Hexcel names Boeing; Sanmina
names HP; Lear names Ford.

## What this means for the design

A broad S&P 500 cross-sectional signal is **not supportable** on this data. At
under one edge per firm-year, most firms have no counterparty on most dates, and
a "weighted average counterparty return" computed over one link is a noisy pair
trade rather than a portfolio.

The study is supportable, and the hypothesis is cleaner, on a universe selected
for customer concentration. That is also where Cohen-Frazzini's mechanism should
be strongest: a supplier whose revenue depends on one named, listed customer is
exactly the case where inattention to the economic link is costly, and small and
mid caps are less arbitraged than the large-cap cross-section.

The infrastructure is unchanged by this. What changes is the universe it points
at, and that decision is now made on a measurement rather than an assumption.
