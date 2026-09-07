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
| E1 | Local Llama 3 8B, JSON-schema constrained decoding | cost | Precision **0.569**, recall **0.745**, F1 0.646 on benchmark v2. (The 0.22 first reported was a v1 measurement artifact, not a property of the model.) |
| E2 | Prompt v1 — named counterparties, verbatim evidence, direction from the filer | pre-registration | Superseded. Correctly excluded anonymous references; silent on the failure modes E1 actually produced. |
| E3 | Prompt v2 — adds explicit rejection of executive biographies, one-off transactions (acquisitions, IP sales), litigation, regulators and government bodies, financial boilerplate, and geographies; states direction as the flow of goods | four observed E1 failure classes | Current. Not yet scored. |
| E4 | Claude Opus 5 via the Batch API | E1 precision | Built, not yet run. |

| E5 | Qwen 3 32B, run locally | E4 needs API credit this project does not have | **Precision 0.900, recall 0.818, F1 0.857** on benchmark v2, against E1's 0.569 / 0.745 / 0.646. Free, but 152s and 22GB per filing. |

Every extraction specification is scored against the same annotated sample, so
the numbers are comparable.

**Benchmark v2** (`docs/gold_links.json`, 2026-09-04): 10 filings, 50 links,
5 of them empty. Replaces v1 (archived as `gold_links_v1.json`), which had
20 links and four filings annotated empty that were never verified — the model
predicted 11, 15, 6 and 8 links on them, and precision 0.22 rested entirely on
whether those were right.

v2 is harder on purpose. Five filings are rich-looking and genuinely empty:
First Solar names eleven companies, all inside executive biographies; Qualcomm
names Veoneer, Arriver, SSW Partners, Magna, PwC and the European Commission,
all acquisitions, transaction counterparties or regulators. Every excluded name
is listed in `gold_exclusions.md` with the rule that excluded it, so the
benchmark's negatives are auditable rather than implicit.

One number falls out of building it. Across the sample, 85.7% of filings carry a
resolvable name near relationship language, but only half disclose a real
relationship. That gap is the noise any extractor has to filter, and it is why
precision rather than recall is the binding constraint.

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

## Graph

| # | Specification | Decided on | Outcome |
|---|---|---|---|
| G1 | Edge validity from filing date to the issuer's next filing, capped at 550 days | how 10-Ks restate relationships annually | Current. Verified on 322 edges: no self-loops, no edge visible before its filing date, no malformed intervals. |
| G2 | Edge weight = disclosed revenue share, else a confidence-graded prior | — | **Withdrawn as described.** Confidence is degenerate: Llama 3 returned "high" for 1,051 of 1,060 claims, Qwen 3 for all 74. The prior is a constant in practice. |
| G3 | Edge weight = disclosed revenue share, else equal | G2's measurement | Current, and named accurately. Alternatives that would actually vary — repeated mention across filings, relation type — are unexplored. |

## Signal

*Empty. No forward return has been regressed on any graph-derived quantity.*

Blocking observation from G1: on the 197 filings extracted so far, only 17-38
edges are active on any given date, covering 8-13 source firms against a
tradable cross-section of roughly 450. That is a pair trade, not a
cross-sectional signal, and it is the condition `PILOT_FINDINGS.md` warned
about. The corpus run covers 2,364 filings, twelve times as many, so the
question is whether density scales with it. If it does not, the study stops at
the coverage report exactly as the pre-registration says it should.

When the first entry lands here, the falsification table in `HYPOTHESIS.md`
applies as written, and the reported threshold is deflated by the number of
entries in this section — not by the number of specifications overall, since the
choices above were made on extraction quality and coverage rather than on
anything correlated with the return test.
