# Dataset release

What this project can publish, in what form, under what licence — and, as
important, what it cannot publish and why.

Counts below are **measured**, not projected.

The released graph is the E5 pilot: 163 filings extracted under the most accurate
configuration on the frontier (`qwen3:32b`, F1 0.857 at 270s per filing), stopped
on 2026-09-08 at the coverage report the pre-registration commits to. A separate
corpus run under E10 (`qwen3:30b-a3b`, F1 0.792 at 13s per filing) is under way to
supply the pre-registered signal test, which needs coverage rather than precision.
The release stays on E5, where a false edge is a permanent defect; see
`EXTRACTION.md` for why the two artefacts take different points on the same
frontier.

## What is actually new here

`HYPOTHESIS.md` concluded, after checking the literature, that the return result
is a replication rather than a discovery. That makes the construction protocol
and its artefacts the primary contribution, and it fixes what the release is for:

1. **An extraction benchmark with auditable negatives.** Ten filings, 55
   links, five of them deliberately empty-but-rich, and every *excluded* name
   listed with the rule that excluded it. Benchmarks for this task normally
   publish positives only, which makes precision unfalsifiable.
2. **A measured cost/quality frontier for local extraction.** Six model
   configurations scored on that benchmark, with the errors decomposed into
   hard negatives, direction errors and enumeration recall. The headline
   findings — reasoning eliminates relation inversions, a 20x-faster
   mixture-of-experts model fails only on long enumerations, and parameter
   count does not order the results — are in `EXTRACTION.md`.
3. **A point-in-time link graph from filing prose**, 163 filings across 38
   issuers, every edge keyed to the disclosure date with an explicit validity
   interval. A pilot at this size.

The first two are what someone else can build on. The third demonstrates that
the protocol produces what it claims to, at the scale currently extracted.

---

## Layers, and whether each can be redistributed

| Layer | Source | Ship it? | Why |
|---|---|---|---|
| Extracted link claims | derived from 10-K prose | **Yes** | Facts about who deals with whom; not expression |
| Evidence quotes | verbatim from 10-K prose | **Yes, one sentence each** | Necessary for audit; de minimis. See the copyright note |
| Resolved edge panel | claims + EDGAR registry | **Yes** | Derived |
| Extraction log | our run | **Yes** | Ours |
| Filing index | EDGAR metadata | **Yes** | Accessions and dates, not documents |
| Full filing text | EDGAR | **No** | Registrant-authored; large; fetchable |
| Gold benchmark | our annotation | **Yes** | The point of the release |
| PIT fundamentals | SEC XBRL | **Yes** | Already shipped by `export_dataset.py` |
| Membership spells | public wiki history | **Yes** | Already shipped |
| Price panel | Yahoo via yfinance | **No** | Terms forbid redistribution |
| `features.parquet`, `target.parquet` | price-derived | **No** | See the trap below |

### The copyright note, stated properly

The shorthand "SEC filings are public domain" is not quite right, and the
release should not lean on it. 17 U.S.C. §105 removes copyright from works *of
the U.S. Government*. A 10-K is authored by the registrant, a private company;
filing it with the Commission does not place it in the public domain. EDGAR is
freely accessible and freely redistributed in practice, and the SEC imposes no
restriction of its own, but the underlying text carries whatever rights the
filer holds.

The position taken here does not need that question resolved:

- **The structured output is facts.** That Boeing named Spirit AeroSystems a
  supplier on a given date is a fact, and facts do not carry copyright
  (*Feist*). The claim and edge tables are safe on their own terms.
- **Evidence quotes are one sentence, and load-bearing.** Every claim carries
  the sentence it came from, checked as a literal substring of the source, so
  the hallucination rate is auditable rather than asserted. A single sentence
  quoted for verification is about as clean a fair-use posture as exists, and
  removing them would gut the release's main methodological claim.
- **Full documents are not shipped.** The filing index carries accession
  numbers; `scripts/build_graph.py --stage filings` re-fetches the text from
  EDGAR in about an hour. This is also the honest engineering answer — the
  corpus is tens of gigabytes and EDGAR serves it better than we would.

### The trap: price-derived tables

`prices/` is obviously excluded. The less obvious exclusion is anything computed
*from* prices — the returns panel, `features.parquet`, `target.parquet`, and any
per-name return series used in the signal test. Publishing 21-day forward
returns per ticker per date is publishing Yahoo's price history in a rotated
basis; a redistribution restriction that can be defeated by a subtraction was
never a restriction. Users fetch their own prices, exactly as the existing
factor study already requires.

The consequence is that the signal result will be **reproducible but not
re-derivable from the release alone**: a user runs `make prices` first. That is
the same contract `export_dataset.py` already ships under, so it introduces no
new friction.

### The model-licence question, which is not obvious

The claims are model output, and the model's licence follows its output.

| Model | Licence | Consequence |
|---|---|---|
| Qwen 3 32B (E5) | Apache 2.0 | Clean. No attribution or field-of-use restriction on outputs |
| Llama 3 8B (E1) | Meta Llama 3 Community License | Requires "Built with Meta Llama 3" attribution, and restricts using outputs to train other models |

**Recommendation: the headline corpus is the Qwen 3 extraction only.** It is
also the better extractor by a wide margin — precision 0.900 / recall 0.818
against 0.569 / 0.745 — so nothing is lost by making it the release. The Llama 3
outputs stay in the repository as the model-comparison evidence behind E1 and
E5, and if they are shipped at all they go in a separate file carrying the Meta
attribution notice.

This matters more than it looks. A benchmark released under an Apache-2.0-clean
provenance can be used to train and evaluate anything; one contaminated with
Llama-3-derived rows cannot, and the restriction would attach silently.

*Both licence terms should be re-read against their current text before release
rather than trusted to this table.*

---

## Contents and format

Parquet is the working format; CSV is the shipped one. The tables are small
enough that CSV costs little, and a release that opens in a spreadsheet gets
read by people a Parquet release does not.

```
dist/
  links/
    link_claims.csv.gz          one row per extracted claim, pre-resolution
    link_edges.csv.gz           resolved, point-in-time validity intervals
    extraction_log.csv          one row per filing attempted
    filing_index.csv            accession -> issuer, form, filing date
  benchmark/
    gold_links.json             50 annotated links over 10 filings
    gold_exclusions.md          every excluded name, with the rule that excluded it
    README.md                   how to score against it
  pit/
    pit_fundamentals.csv.gz     unchanged from the existing export
    sp500_membership_spells.csv unchanged
  DATASHEET.md
  LICENCE-DATA
  CITATION.cff
```

### `link_claims.csv.gz` — what the model said

One row per surviving claim. **Measured: 933 claims over 163 filings** (5.72
per filing), 38 distinct issuers, filing dates 2010-02 to 2025-11. 46% of
filings yield no claim at all. By relation: 358 `customer`, 315 `competitor`,
152 `supplier`, 108 `partner`. 59 claims carry a disclosed revenue share.

| column | type | notes |
|---|---|---|
| `ticker` | str | filer |
| `accession` | str | EDGAR accession, undashed; joins to `filing_index` |
| `filed` | date | **the disclosure date; the only date the graph uses** |
| `counterparty` | str | name as written in the filing, unresolved |
| `relation` | enum | `customer` / `supplier` / `partner` |
| `revenue_pct` | float | disclosed share of revenue, else null |
| `confidence` | enum | model self-report. **Degenerate — see caveat** |
| `evidence` | str | verbatim sentence, verified substring of the source |

Claims failing the substring check are dropped and *counted* in
`extraction_log.csv` rather than silently discarded, so the hallucination rate
is a published number.

### `link_edges.csv.gz` — the graph

One row per edge-interval.

| column | type | notes |
|---|---|---|
| `source` | ticker | filer |
| `target` | ticker | resolved counterparty |
| `relation` | enum | direction is the flow of goods |
| `valid_from` | date | filing date |
| `valid_to` | date | next filing from the same issuer, capped at 550 days |
| `weight` | float | disclosed revenue share, else equal |
| `confidence` | enum | carried through; see caveat |

Intervals are half-open, `[valid_from, valid_to)`. **Measured: 264 edges, 35
source firms, 100 counterparties**, from the 400 claims (42.9%) that resolve to
a listed ticker. Competitor relations are extracted and kept in the claims table
but excluded from the graph by `build.edges`, which is why 400 resolved claims
become 264 edges. Verified: no self-loops, no edge visible before its filing
date, no malformed intervals.

### `extraction_log.csv` — the error rate, shipped with the data

Per filing: status, raw claim count, kept count, seconds, and the rejection
counts (`rej_no_evidence`, `rej_anonymous`, `rej_empty_name`). This is what
makes the extraction quality a measured quantity rather than a footnote, as
the pre-registration requires. It ships as a first-class table, not an appendix.

### Caveats that travel with the data

Two are non-negotiable, because a user who does not know them will misuse the
tables:

- **`confidence` is degenerate.** The model returned "high" for essentially
  every claim — 1,051 of 1,060 for Llama 3, all 74 for Qwen 3 in the pilot. The
  column is retained for provenance and **must not be used as a weight**. It is
  documented as degenerate in the datasheet and in the column comment.
- **The registry has no historical names.** Resolution runs against EDGAR's
  *current* company names, so Raytheon, Navistar, Chrysler and Alcatel-Lucent
  resolve to nothing (R4). Coverage is therefore biased against firms that were
  renamed or acquired — a survivorship-flavoured gap in the graph, in the same
  direction as the biases `BIAS.md` documents elsewhere. It is a stated
  limitation, not a fixed defect.

---

## Licence

Three-way split, because the layers have genuinely different provenance:

| Component | Licence |
|---|---|
| Code | MIT (unchanged, already in `pyproject.toml`) |
| Derived tables and benchmark annotations | **CC BY 4.0** |
| Evidence quotes within those tables | Quoted from registrant filings; not licensed by this project |

**CC BY 4.0 rather than CC0** for the data. The usual argument for CC0 is
frictionlessness, and for a pure facts table it would be right. But the benchmark
is the part most likely to be reused, and a benchmark whose results are cited
back is a benchmark that stays honest: attribution is how a later paper's
precision number can be checked against the annotation rules in
`gold_exclusions.md`. The cost is one line in a bibliography.

`LICENCE-DATA` states the split explicitly, including the carve-out for quoted
material, so a downstream user is not left to infer that a CC BY grant covers
text this project never owned.

## Datasheet

`DATASHEET.md` follows Gebru et al., and the sections that carry real content
here are motivation, composition, collection, preprocessing, uses and
limitations. It is mostly assembly rather than new writing: `HYPOTHESIS.md`,
`PILOT_FINDINGS.md` and `SPECIFICATIONS.md` already contain the substance,
including the failures. The one section that needs writing from scratch is
**Uses** — specifically the anti-uses:

- Not a commercial supply-chain graph substitute. Coverage is roughly three
  counterparties per filing on the filings that name any, and half of all
  filings name none.
- Not a complete link structure. It contains what firms *chose to disclose in
  prose*, which is a selected sample of the real graph, and the selection is
  correlated with materiality thresholds under ASC 280.
- Not usable for point-in-time work before renaming is handled (R4).

## Release checklist

- [x] Replace projected counts with measured ones
- [x] Report in-universe edge density **before** any return is computed, per the
      pre-registration. Done, and the study stopped there: see the Signal
      section of `SPECIFICATIONS.md`
- [x] Ship the extraction frontier (`EXTRACTION.md`, `extraction_table.md`)
- [ ] Expand the benchmark toward ~100 filings and report inter-annotator
      agreement on a double-annotated subset (`scripts/annotator_agreement.py`)
- [ ] Run the passage-splitting experiment on the MoE
- [ ] Confirm Qwen 3 and Llama 3 licence text against current versions
- [ ] Strip every price-derived column; assert it in a test, not by inspection
- [ ] `scripts/export_dataset.py` extended to emit `links/` and `benchmark/`
- [ ] Datasheet, `LICENCE-DATA`, `CITATION.cff`
- [ ] Decide the archive: Zenodo gives a DOI and a citable version, which suits
      a benchmark better than a bare repository tag

## Open questions for the author

1. **Is the negative result released the same way as a positive one?** The
   pre-registration says yes and the release plan above assumes yes: the graph
   and benchmark ship regardless of how the signal test comes out. Worth
   confirming, because it is much easier to agree to in advance.
2. **Does the benchmark ship at ten filings, or wait for more?** Ten is small.
   It is also annotated to a written rule set with auditable negatives, which
   nothing comparable is. Shipping it now and versioning it (v2 → v3) is
   probably better than holding it back.
3. **Zenodo DOI, or repository release only?**
