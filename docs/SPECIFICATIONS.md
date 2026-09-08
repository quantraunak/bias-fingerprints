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
| E6 | Qwen 3 32B with reasoning disabled (`--no-think`) | corpus throughput, not quality | **Rejected.** Precision 0.486, recall 0.655, F1 0.558 against E5's 0.900 / 0.818 / 0.857, for a 24% speedup. Scored on the same ten filings, both settings read from cache. |
| E7 | `num_predict` 6000 inside an 8192-token context | — | **Withdrawn.** A ~3,400-token prompt plus a 6,000-token budget exceeds the window, so long reasoning traces trigger context shift and re-prefill. Median 116s, mean 781s, worst filing 11,843s; the eight slowest of 93 filings consumed 62% of all wall time. |
| E8 | Qwen 3 30B-A3B (mixture of experts, ~3B active), `think=False` | 32B decode is memory-bandwidth-bound; a 30B MoE reads ~1/10th the weights per token | **Rejected on quality, at a bar fixed before the measurement.** Precision 0.824, recall 0.764, F1 0.792 against E5's 0.900 / 0.818 / 0.857 — below the 0.82 threshold set in advance. But **13s per filing against 270s**, a 20x speedup that turns the corpus from ~13 days into ~7 hours. Loses on long enumerations: 14 of NVIDIA's 24 links, dropping the whole outsourced-assembly list (ASE, Hon Hai, Siliconware, King Yuan, Unimicron, Ibiden, Nanya). Beats E5 on Boeing, 12 true positives against 6. |
| E9 | Qwen 3 30B-A3B with reasoning enabled, recovered from Ollama's `thinking` field | E6 measured reasoning as worth 0.30 F1 on the 32B; E8's gap to the bar was 0.065 | **Rejected. Byte-identical to E8** — precision 0.824, recall 0.764, F1 0.792, the same 42/9/13 split and the same missed and spurious lists, from a separate cache key. Under schema-constrained decoding `think` only changes which field Ollama files the tokens under; `eval_count` is unchanged. The MoE has no reasoning mode to unlock on this task. |

Every extraction specification is scored against the same annotated sample, so
the numbers are comparable.

E6 is the clearest case so far of the benchmark overruling a plausible argument.
Disabling reasoning was proposed on throughput grounds and looked defensible --
it returned *more* links, 81 against 53 on the twelve filings held in both
caches. Scoring showed those extra links were false positives. On First Solar,
where the gold annotation is empty because all eleven named companies sit inside
executive biographies (rule A3), reasoning-on returns nothing and reasoning-off
returns twenty spurious links. On NVIDIA it recovers Hon Hai, Quanta and
Siliconware and labels every one of them `customer` rather than `supplier`.
Reasoning is buying the hard negatives and the direction of the relation, which
is precisely what benchmark v2 was built to detect.

E8's first run scored zero -- ten filings, `raw=0` on every one -- and that was a
plumbing failure, not a measurement. For this model Ollama files the entire
structured response under `thinking` and returns `response: ""` unless `think`
is explicitly `False`, and `local_model.generate` reads `payload["response"]`.
The failure is silent and its signature is indistinguishable from a model that
genuinely found nothing: ten empty extractions, cached as legitimate zero-link
filings. Had the corpus been started on the MoE without scoring first, it would
have produced two thousand cached blanks and a graph with no edges.

Two consequences. First, `local_model.generate` should treat an empty
`response` alongside a non-empty `thinking` as an error rather than as a result.
Second, and easy to misread: `--no-think` on the MoE is **not** the setting E6
rejected. On the 32B, disabling reasoning cost 0.30 F1. On the MoE it is how the
output is retrieved at all, and `eval_count` is identical across `think` unset,
`True` and `False` -- the flag changes which field Ollama files the tokens
under, not how many are generated. The two findings share a flag name and
nothing else.

E9 ran that configuration and it changed nothing: the output is identical to
E8's on all ten filings. Under schema-constrained decoding this model does not
reason regardless of the flag, so the 0.065 gap to the bar does not close and
the extractor search ends here. Qwen 3 32B (E5) remains the only extractor that
meets the standard, at 270s per filing.

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

**Density does scale, measured 2026-09-08 on 98 filings extracted under E5.**
6.06 validated claims per filing, of which 2.59 resolve to a listed ticker
(42.8%). The corpus is 2,364 filings across 160 distinct issuers, roughly 145 of
which file in any given year, so with G1's validity intervals the projection is
**~145 covered source firms carrying ~375 active edges, about 2.6 counterparties
each**. Against the 17-38 edges and 8-13 firms measured earlier that is a
twelve-fold improvement, and it clears the condition the pre-registration set.

At n≈145 per month over ~138 months the standard error on mean IC is about
0.007, so the `t > 2.0` threshold in the falsification table is reachable at an
IC of roughly 0.014 — below published customer-supplier spillover effects. The
test would be adequately powered *if the corpus were complete*.

Three caveats travel with that number, and the first is load-bearing. The
extraction reached 163 filings covering **38 of the 160 issuers** before it was
stopped, so the projection extrapolates from under a quarter of the issuer
cross-section, and issuer identity is exactly what determines counterparty
yield — a semiconductor firm names foundries, a utility names nobody. The
projection is therefore an estimate with a wide and unquantified interval, not a
measurement. Second, 160 issuers is not the 450-name cross-section the
twenty-two factors are measured on, so the two signals are comparable in method
but not in breadth. Third, the relation mix is uneven: of 933 validated claims,
358 are `customer`, 315 `competitor`, 152 `supplier` and 108 `partner`, and
`build.edges` correctly drops competitors from the graph, so a third of what the
model extracts does not become an edge at all.

No forward return has been touched to produce any of the above.

When the first entry lands here, the falsification table in `HYPOTHESIS.md`
applies as written, and the reported threshold is deflated by the number of
entries in this section — not by the number of specifications overall, since the
choices above were made on extraction quality and coverage rather than on
anything correlated with the return test.
