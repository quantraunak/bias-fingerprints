# Specifications tried

`docs/HYPOTHESIS.md` commits to maintaining this file so the multiple-testing
correction at the end has an honest denominator. Every choice that could have
gone another way is appended here as it is made, including the ones that failed.

**No signal specification has been run yet.** Nothing below has touched a
forward return. The count that matters for deflating a final t-statistic is the
number of entries in the *signal* section, and that section is empty on purpose:
the log starts before the first test rather than being reconstructed after it.

---

## Universe

| # | Specification | Decided on | Outcome |
|---|---|---|---|
| U1 | S&P 500 point-in-time members | measured edge yield | Rejected on a biased measurement, then **reinstated**. Re-measured at 3.03 counterparties/filing and 85.7% coverage with the fixed resolver — statistically level with U2. Current. |
| U2 | Concentrated suppliers, selected by SIC industry | structure knowable before reading any filing | Withdrawn as the primary universe. The gap that motivated it (3.18 vs 3.03) does not survive R3. Retained as a robustness cut. |

## Extraction

| # | Specification | Decided on | Outcome |
|---|---|---|---|
| E1 | Local Llama 3 8B, JSON-schema constrained decoding | cost | Precision **0.22**, recall **0.95** on the 10-filing annotated sample. Recall is adequate; precision is not. |
| E2 | Prompt v1 — named counterparties, verbatim evidence, direction from the filer | pre-registration | Superseded. Correctly excluded anonymous references; silent on the failure modes E1 actually produced. |
| E3 | Prompt v2 — adds explicit rejection of executive biographies, one-off transactions (acquisitions, IP sales), litigation, regulators and government bodies, financial boilerplate, and geographies; states direction as the flow of goods | four observed E1 failure classes | Current. Not yet scored. |
| E4 | Claude Opus 5 via the Batch API | E1 precision | Built, not yet run. |

Every extraction specification is scored against the same annotated sample, so
the numbers are comparable. The sample is currently 10 filings and 20 links,
which is too small to separate 0.22 from 0.30 with any confidence; expanding it
is a prerequisite for treating E3 or E4 as an improvement over E1.

## Resolution

| # | Specification | Decided on | Outcome |
|---|---|---|---|
| R1 | Tiered exact → token-set → subset matching, no edit distance | precision over recall | Retained. Edit distance stays rejected: "Delta Air Lines" and "Delta Apparel" are close in edit distance and unrelated. |
| R2 | Subset tier gated on token count | — | Rejected. Merged Semiconductor Manufacturing International into TSM, and blocked "John Deere" → DE. A count cannot distinguish a token that names one company from one that names an industry. |
| R3 | Subset tier gated on registry document frequency of the shared tokens | R2 failures, both directions | Current. Claim match rate 34.3% → 39.9%. |
| R4 | Historical names from EDGAR `formerNames` | Raytheon, Navistar, Chrysler and Alcatel-Lucent resolve to nothing because the registry holds only current names | **Not built.** Resolving twenty years of filing text against a current-name registry is itself a survivorship problem. |

## Annotation rules

Four cases recur across filings and cannot be decided per-filing without the
benchmark becoming inconsistent. Decided 2026-09-04, applied to every annotation
including the five already written. Each is reversible: annotations carry the
evidence span and a `related_party` tag, so a different ruling can be applied by
re-filtering rather than re-reading.

| # | Case | Ruling | Reasoning |
|---|---|---|---|
| A1 | Government buyers (NASA, DoD, Homeland Security) | **Include** as `customer` | Extraction should be faithful to the filing; resolution is the right stage to drop what cannot be traded. Excluding at extraction would hide 43% of Boeing's revenue from the benchmark and conflate two different judgements. |
| A2 | Joint ventures the filer part-owns (AMD's ATMP JV, Boeing's ULA and Sea Launch) | **Include**, tagged `related_party` | A supplier by function and a related party by structure. Tagging keeps both readings available, so a related-party-excluded robustness cut costs a filter rather than a re-annotation. |
| A3 | A real relationship named only inside an executive biography (First Solar's 8point3 yieldco with SunPower) | **Exclude** | The biography rule has to be mechanical or it stops being a rule. A model cannot be expected to distinguish a true corporate fact inside a résumé from a false one, and the failure mode this guards against — Lucent, Ericsson, Medtronic, GE as counterparties — is the dominant source of 8B false positives. |
| A4 | A historical agreement surfaced for an unrelated purpose (NVIDIA's 2000 Xbox agreement, cited in 2018 to explain a change-of-control provision) | **Exclude** | The filing offers no evidence the relationship is live on the filing date, and edge validity intervals start at the filing date. |

## Signal

*Empty. No forward return has been regressed on any graph-derived quantity.*

When the first entry lands here, the falsification table in `HYPOTHESIS.md`
applies as written, and the reported threshold is deflated by the number of
entries in this section — not by the number of specifications overall, since the
choices above were made on extraction quality and coverage rather than on
anything correlated with the return test.
