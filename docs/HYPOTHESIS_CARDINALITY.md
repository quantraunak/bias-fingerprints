# Pre-registration: output cardinality as an extraction evaluation axis

Written 2026-09-10, before any configuration was run against a gold set built
for this question. `RELATED_WORK.md` was written first and records the
literature check; this document fixes the design, the estimator, and the reading
of every outcome before the data exists.

## Hypothesis

**Structured-extraction recall degrades as a function of the number of items the
model must emit, holding schema and input length fixed.**

Call that number the *cardinality* `k`. The claim is that `k` is an evaluation
axis in its own right — not a proxy for input length, not a proxy for schema
breadth, both of which are already measured in the literature and both of which
are held constant here.

The mechanism claim, weaker and tested separately: degradation is a property of
*items per call*. If so, partitioning the input so each call carries a smaller
`k` recovers the lost items.

## Proposed framework

Report extraction quality as a curve, not a scalar.

1. **Cardinality-controlled evaluation.** For an extraction task with a repeated
   array under a fixed schema, report recall `R(k)` and precision `P(k)` against
   gold, with `k` the gold item count for that document. A single F1 over a
   benchmark whose `k` distribution is unstated is not comparable across
   benchmarks, because it silently averages over a `k` mix.
2. **Orthogonality discipline.** `k` must be shown decorrelated from input
   length `L` in the evaluation set, or `L` enters as a covariate. Here
   `passages.select` caps the prompt at 14,000 characters, and the measured
   correlation on benchmark v2 was `r = -0.296`.
3. **Per-document estimation.** The primary estimate is per-document, regressed
   on `k`. Bucket means over `k` strata are for display only. This project has
   already been burned once by a bucket average over two filings, recorded in
   `RELATED_WORK.md`, and was nearly burned a second time at n=86 where the
   aggregate ratio moved from 1.46 to 0.59 while the per-document rank
   correlation was +0.002.
4. **Mitigation as mechanism test.** If `R(k)` declines, splitting the input into
   `m` partitions and unioning the results must raise total recall. If it does
   not, the effect is not items-per-call and the framework's mechanism claim is
   wrong even if its measurement claim survives.

## Design

**Factors.** 2x2 fully crossed, one model family:

|  | think | no-think |
|---|---|---|
| dense (`qwen3:32b`) | cell A | cell B |
| MoE (`qwen3:30b-a3b`) | cell C | cell D |

The existing E5/E10 comparison is cells A and D only, which confounds
architecture with reasoning mode. Cells B and C are what make the 2x2 identify
either factor. A second family (`llama3` dense / a Mixtral-class MoE) replicates
or refutes Qwen-specificity, and runs only if the primary test rejects.

**Evaluation set.** 60 filings drawn from the 1,989-filing corpus, stratified to
span `k`: target 10 filings in each of `k` = 1, 2-3, 4-6, 7-9, 10-14, 15+.
Strata are assigned by `max` over already-cached model outputs, which is a
*selection* device only — `k` for analysis is the gold count, never a model
count. Selection is seeded and the seed is recorded.

**Annotation.** Seeded from model output and corrected, following the rules in
`gold_exclusions.md` (A3 executive biography, A4 historical agreement, and the
rest), with every exclusion recorded by rule rather than by taste. The annotator
is blind to which configuration produced a seed. A second annotator scores >= 20
of the 60 and agreement is reported before any recall number is.

## Estimator, fixed in advance

- **Primary:** Spearman rank correlation between gold `k` and per-filing recall,
  pooled over the four cells, filings as the unit. `n >= 40`.
- **Effect size:** OLS slope of recall on `log2(k)`, standard errors clustered by
  issuer, since one issuer contributes several filings.
- **Covariate:** prompt length `L`. Entered only if `|corr(k, L)| > 0.3` in the
  realised set.
- **Secondary, explicitly exploratory:** the same fit within each of the four
  cells, and the architecture and reasoning-mode main effects. These are not the
  test; they are what generates the next hypothesis.

## Falsification table

Fixed before the data exists. Rejection requires **both** the rank test at
`p < 0.05` **and** an effect size of at least 0.15 recall between the `k <= 3`
and `k >= 10` strata. A bar moved after seeing the number is not a bar.

| outcome | reading |
|---|---|
| Recall declines in `k` in all four cells | Cardinality is an evaluation axis. Report `R(k)`, then run the split mitigation and the second family. |
| Recall flat in `k` (rank test null, or effect < 0.15) | **Null, and the study ends.** The output is a short negative note: bucket averages over small benchmarks manufacture trends, with the n=2 and n=86 cases as the worked example. |
| Declines in MoE cells only | Architecture-specific. Routing instability under repeated same-type tokens becomes the mechanism hypothesis; a different paper. |
| Declines in no-think cells only | A reasoning-budget effect, not an architecture one. Still publishable, different claim, and the framework's axis survives while its mechanism story changes. |
| Declines, but splitting does not recover it | The measurement claim survives and the mechanism claim dies. Report both; do not quietly drop the mitigation. |
| Precision rises as recall falls with `k` | The model is trading recall for precision under load, not failing. Reframes the axis as a decision boundary, not a defect. |

## Controls, run before the experiment

1. **Known answer.** Re-score the existing configurations against benchmark v2
   and reproduce the recorded F1 (dense 0.857, MoE 0.792). If the harness does
   not reproduce its own numbers, everything downstream is unreadable and the
   study stops until it does.
2. **Ceiling.** No cell may be token-limited at high `k`. Measured 2026-09-10 on
   the corpus: high-`k` responses average 5,876 characters against a ~16,000
   character ceiling, 4 of 29 above 12,000. Re-checked per cell, since the
   thinking cells emit more.
3. **Orthogonality.** Report `corr(k, L)` on the realised evaluation set.

## Stopping rules

The study ends, and is written up as it stands, if: the control in (1) fails and
cannot be repaired; or the primary test is null at `n >= 40`. Neither outcome is
a reason to add cells, change the estimator, or search strata for an effect.

## Realised evaluation set — recorded before any cell was run

Seed `20260910`. 60 filings, 10 per stratum, 51 distinct issuers, no issuer
contributing more than 2. Proxy `k` spans 1 to 33.
`reports/cardinality_eval_set.csv` holds the selection.

Two design choices worth their own record, because both could have gone
otherwise:

**Selection was not allowed to prefer already-cached filings.** A first selector
preferred filings already extracted under the dense model, which would have cut
4.4h of compute. It was discarded: those 163 filings come from `pilot_universe`,
ranked by index tenure, so the evaluation set would have been almost entirely
large-cap semiconductor and hardware issuers and the result would license a
claim only about those. Stratified random draw from the whole corpus instead,
at a cost of 4h of unattended local compute.

**Strata are assigned by the MoE's own item count**, since that is the only
count available for all 1,989 filings. This selects, at high `k`, filings on
which the MoE emitted many items — i.e. filings where it did *not* stop early.
If the cardinality effect is real, this biases the MoE cells *upward* at high
`k` and works against the hypothesis. The direction is conservative, which is
the acceptable direction for a selection artifact, but it is a reason the
measured effect is a lower bound rather than an estimate. `k` for analysis
remains the gold count.

## Controls, executed 2026-09-10, before any new cell was run

| control | required | measured | verdict |
|---|---|---|---|
| dense `qwen3:32b` reproduces recorded F1 | 0.857 | **0.857** (P 0.900 / R 0.818) | pass |
| MoE `qwen3:30b-a3b --no-think` reproduces recorded F1 | 0.792 | **0.792** (P 0.824 / R 0.764) | pass |

The harness reproduces its own numbers, so downstream measurements are readable.

Both control runs also reproduce the original contradiction at full strength.
NVDA, gold `k`=24: dense recall 1.00, MoE 0.54. BA, gold `k`=14: dense 0.43, MoE
0.86. The MoE's eleven NVDA misses are all suppliers from a single long
enumeration — Micron, Hon Hai, Nanya, Siliconware, Unimicron, King Yuan and
others — of which it emitted 14 of 24 and stopped. That is the predicted
mechanism visible in one filing, and it is not evidence; n=2 filings pointing in
opposite directions is what this study exists to replace.

## Correction to the annotation protocol — 2026-09-10, before annotation began

The design above says annotation is "seeded from model output and corrected."
**That is wrong and is replaced**, before any filing was annotated.

Seeding from model output makes gold a function of what models found. If the MoE
omits eleven suppliers from a long enumeration — which is precisely the
behaviour under test — an annotator reviewing only model-proposed candidates
never sees those eleven, they never enter gold, and measured recall at high `k`
is inflated. The bias runs directly against the hypothesis in a way that would
look like a null. An instrument that cannot record the failure it is built to
measure is not an instrument.

Annotation therefore seeds from `scripts/gold_view.py`, which is
model-independent: it keeps every sentence containing a capitalised run, plus
one sentence either side, and drops only sentences that can carry neither a name
nor relationship language. Its completeness argument is stated in that file — a
named counterparty must appear inside a sentence containing a capitalised run,
because that is what a company name is — so it is a superset of what any
extractor working from the same passages could legitimately find.

Scope is unchanged and follows `sample_gold.py`: gold is defined over
`passages.select` output, not the whole 10-K, so a link in a dropped section is a
selector failure rather than an extraction failure and is not scored as a false
negative.

Blinding survives the change and gets stronger: candidate spans carry no model
attribution at all, because no model produced them.

Agreement is reported with `scripts/annotator_agreement.py` on >= 20 of the 60
filings before any recall number is reported — Cohen's kappa on the disclosure
decision as the headline, pairwise link-set F1 alongside it.

---

## Superseding design: controlled-k documents — 2026-09-10

The observational design above requires 60 hand-annotated filings. It is
**replaced as the primary study** by a constructed one, before any annotation was
done. The observational arm survives as an external-validity anchor, scored
against the existing 10-filing gold, and its 2x2 cells are already running.

### Why construct rather than annotate

`k` is the treatment. In the observational design `k` is a property of whichever
filings happened to be sampled, must be recovered by a human, and is measured
with error that is itself correlated with the outcome — the annotator's
attention flags on exactly the long enumerations where the effect should live.
Constructing documents makes `k` exact, free, and manipulable, and turns an
association into an ablation.

### Construction

- **Item bank.** Evidence spans from the union of the dense (E5) and MoE (E10)
  claim tables — real filing sentences, each verbatim-validated against its
  source, each carrying exactly one counterparty. Spans carrying several
  counterparties are excluded so that items and sentences do not covary.
- **Levels.** `k` in {1, 2, 4, 8, 16, 24, 32}, drawn without replacement per
  document, counterparties deduplicated within a document.
- **Filler.** Passages from filings where no configuration found any claim, with
  every sentence containing a capitalised run removed using `gold_view.py`'s
  candidate detector, so the filler cannot carry a legitimate link.
- **Length.** Every document padded to 14,000 characters, the cap
  `passages.select` already enforces. Input length is therefore constant by
  construction rather than merely uncorrelated.
- **Position.** Injection points drawn uniformly at random, so position effects
  are averaged over rather than confounded with `k`.
- **Gold.** The injected set. Recall is the share of injected counterparties
  recovered; anything emitted that was not injected is a false positive.

### Estimator

Unchanged in spirit, better powered. Primary: OLS of per-document recall on
`log2(k)`, errors clustered by document batch, run within each of the four 2x2
cells and pooled. The pre-registered rejection rule is unchanged — `p < 0.05`
**and** >= 0.15 recall between `k <= 3` and `k >= 16`. The falsification table
above applies as written.

### Threats this design carries, stated before results

1. **Recombined text is not a coherent 10-K.** A model may treat stitched
   passages differently from real prose. This is why the observational arm is
   kept: if the two disagree, the constructed result is a claim about
   constructed documents and must be reported as one.
2. **The item bank is model-found.** Sentences entered it because some
   configuration extracted them, so absolute recall is inflated. The bank's
   composition is identical across `k` levels, so the *slope* — the estimand —
   is unaffected. Absolute recall levels are not interpretable; differences in
   slope are.
3. **Filler may contain an unlabelled company name.** Stripping capitalised-run
   sentences is a conservative filter; residual false positives inflate the
   precision denominator equally at every `k`.

### Literature check owed before this goes further

Multi-needle needle-in-a-haystack variants exist for retrieval QA and are the
closest prior art to a constructed multi-item probe. `RELATED_WORK.md` does not
cover them, because it was written against the observational design. That search
is owed **before the pilot is scaled**, and the claim of novelty is provisional
until it is done. If multi-needle work already characterises recall as a
function of item count under structured output, this study is a replication and
should be described as one.

### Construction validity gate — added 2026-09-10 after the k=1 pilot failed it

The first constructed pilot returned **recall 0.333 at k=1** under the MoE
no-think cell. One item, in a document the model itself supplied the sentence
for, found a third of the time. That is not a finding about cardinality; it is a
broken instrument, and a slope fitted through it would have been meaningless.

**Cause.** `extract.user_prompt` asks for the counterparties *of the filer*. The
item bank contained spans naming their original filer in the third person — "BDS
faces strong competition from Lockheed Martin Corporation" — which, in a document
attributed to a placeholder filer, answers a question the document does not ask.
The model was right to decline them. The construction, not the model, was wrong.

**Fix.** The bank is restricted to first-person spans (`we` / `our` / `us`),
which are filer-agnostic and transplant without changing meaning: "Wal-Mart
Stores, Inc. and its affiliates accounted for 22 percent of our consolidated net
sales" is true of whoever files it. 427 spans survive, against 1,002 before.

**Gate, now binding on every cell.** If recall at `k = 1` is below **0.80**, the
construction is rejected for that cell and no slope from it is interpretable.
`k = 1` is a known-answer input: one item, one document, an extractor that found
that exact sentence in the wild. A cell that cannot clear it is measuring
something other than cardinality.

This is the control that should have run before the first document was ever
scored, and it is recorded here as having been added *after* the failure rather
than before it.

### The count-density-length trilemma — measured 2026-09-10

The k=1 gate failed three times. Diagnosis, holding the injected sentence and
`k=1` fixed and varying only the filler:

| filler chars | recall | items emitted |
|---|---|---|
| 1,500 | 0.667 | 1.00 |
| 4,000 | 0.583 | 0.75 |
| 14,000 | 0.417 | 0.67 |

Two findings, and the second is a design problem rather than a bug.

**1. Dilution costs 0.25 recall at constant `k`.** The 14,000-character documents
were measuring haystack size. The filler was built to contain no company names,
which made it far more inert than a real prompt: `passages.select` keeps only
paragraphs scoring above 0.5, so a real prompt is 14,000 characters of
relationship-dense prose, not boilerplate. Filler must be rebuilt from real
selected passages with entity names masked.

**2. Count, density and length cannot all be held fixed.** They are
mechanically linked — density = (count x span length) / total length. Fixing
total length, as this design does, makes item density *rise* with `k`: a `k=32`
document is 6,400 characters of items and 7,600 of filler, a `k=1` document is
200 and 13,800. Since dilution demonstrably suppresses recall, high-`k`
documents receive a density advantage that partially cancels the effect under
test.

Any study of item count must choose two of the three to fix:

| fixed | varies | residual confound |
|---|---|---|
| length + count (this design) | density | density aids high `k`; effect is a **lower bound** |
| density + count | length | length suppresses high `k`; effect is an **upper bound** |
| length + density | count cannot vary | not a design |

Multi-needle NIAH takes the second row, which is why its degradation is
entangled with context length. This design takes the first, which is the
conservative direction, and the measured 0.25 dilution penalty quantifies how
much the choice is worth rather than leaving it as a caveat.

**Design consequence.** The primary arm stays at fixed length, reported as a
lower bound. A second arm at fixed filler length brackets it from the other
side. Reporting both, with the trilemma stated, is a better contribution than
either arm alone — and it is the part of this study that multi-needle does not
have.

**Gate consequence.** The 0.80 gate stands, but it cannot be met until the
filler is realistic. Absolute recall of 0.667 at minimal dilution also says the
item bank still contains spans that do not survive transplanting; that filter
needs to be empirical — keep only spans the extractor recovers at `k=1` in a
short document — rather than syntactic.

### Stop condition on the instrument — fixed 2026-09-10, before the rebuild

The `k=1` gate has failed three times (0.333, 0.400, 0.400). Each failure was
followed by a fix aimed at a cause diagnosed by inspection, and each diagnosis
was wrong: the first-person filter, the four-defect batch, and the matching
relaxation all moved precision or nothing. Only controlled variation — holding
the item and `k` fixed and varying filler alone — produced a real cause. A
fourth round of inspect-and-patch is how a project becomes a sunk cost.

**One more attempt, with both known causes addressed together:**

1. **Filler rebuilt from realistic prose.** Source from the selected passages of
   *productive* filings, keeping their name-free sentences, rather than from
   filings with no claims — which are the least relationship-dense text in the
   corpus and made the haystack far more inert than any real prompt.
2. **Item bank screened empirically.** Every candidate span is tested at `k=1`
   in a short document and kept only if the extractor recovers it. The bank is
   then defined by measurement rather than by syntactic guesses about which
   spans transplant, which have now been wrong three times.

**If the `k=1` gate still fails after that, the constructed design is
abandoned**, and the study is written up as: a measured dilution penalty at
constant `k`, the count-density-length trilemma, and a negative report on
building controlled-cardinality documents from transplanted domain sentences.
That is a publishable note on its own and it is the honest terminal state if the
instrument will not come up. No fourth iteration.

### The second factor was wrong — replaced 2026-09-11

The 2x2's reasoning-mode factor does not exist. Cells C and D came back
identical to three decimals across all 300 documents, which is impossible for
two configurations. Direct test on one k=4 document at temperature 0:

| format | think | seconds | eval_count | thinking chars | response chars |
|---|---|---|---|---|---|
| yes | True | 37.1 | 268 | 965 | 0 |
| yes | False | 58.6 | 268 | 0 | 965 |
| no | True | 139.8 | 4914 | 18631 | 840 |
| no | False | 163.8 | 4914 | 0 | 19481 |

`think` changes only which response field the same tokens are filed under —
eval_count is identical within each format setting. It was never a treatment.
Cells A and C are discarded; 62 dense documents were computed before this was
caught.

**The real second factor is the one the format rows show: constrained versus
unconstrained decoding, worth 18x the generation budget** (268 vs 4,914 tokens
on the same prompt). Under the schema the model emits 965 characters and no
reasoning; without it, a full `<think>` block and 19,481 characters.

This is the explanation `RELATED_WORK.md` named and could not rule out: *"If
constrained decoding interacts with list length ... the effect would be a
property of the decoding constraint rather than of the architecture. No
configuration has been run without constrained decoding, so this is currently
unfalsified and is the strongest competing explanation."* It is now the factor
under test rather than a caveat.

Arms: constrained (the completed cell D) versus unconstrained with post-hoc
parsing, matched documents, k in {1, 4, 16}. Both are re-parsed from raw cached
text at scoring time, so a parser fix applies retroactively rather than costing
a re-run — which mattered immediately, since the first loose parser returned
zero links on output that contained perfectly good JSON behind a `<think>`
block, and that would have scored as total recall failure.

### The unconstrained arm was context-bound — corrected 2026-09-14

Control 2 of this pre-registration says no cell may be token-limited at high `k`.
It was verified for the constrained arm, which emits 74-268 tokens, and then not
re-checked for the unconstrained arm, which emits 4-6x more. It should have been.

`local_model.CONTEXT_TOKENS` was 8,192 and the prompt is ~2,270 tokens. Measured
on the 30B MoE, unconstrained generation produced:

| k | output tokens | prompt + output vs 8,192 window |
|---|---|---|
| 1 | ~4,700 | 6,970, fits |
| 4 | ~5,900 | 8,148, at the wall |
| 16 | 7,100-12,000 | over, 4 of 6 truncated mid-JSON |

Truncated documents parse to zero links and score as total recall failure, so the
arm would have reported constrained decoding as hugely superior at exactly the
level that decides the study. **The earlier unconstrained numbers (+0.033 at k=1,
-0.014 at k=4) are withdrawn**; only k=1 was clean.

Raising the window on the 30B is not possible here. `llama-server` holds 18.7 GB
with that model loaded on a 36 GB machine, and both 16k and 32k context attempts
were killed by memory pressure.

**The decoding comparison moves to `qwen3:14b` at `num_ctx` 32,768.** Verified on
a k=16 document: constrained 752 tokens and unconstrained 448, both with
`done_reason=stop`, both parsing (10 and 7 links against 16 gold). The context no
longer binds either arm.

Two consequences to state rather than bury. The model is dense rather than MoE,
so this arm speaks to decoding mode and not to architecture, and it is a different
model from cell D. And the 18x token ratio reported earlier is a property of the
30B MoE's unconstrained verbosity, not of unconstrained decoding: the same
document on the 14B costs 448 tokens against 752 constrained, a ratio below one.
