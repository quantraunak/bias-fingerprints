# Related work: does list length break structured extraction?

Written before the experiment, so the design cannot be reverse-engineered from a
result, and so the record shows the literature was checked first. This reverses
the ordering failure recorded at the bottom of `HYPOTHESIS.md`, where prior work
was checked after the design was fixed and materially changed what the study
could claim.

**Status of the finding that motivates this: not established.** See the last
section. The observation is two filings pointing in opposite directions.

## The observation

Scoring six extractor configurations against benchmark v2 produced a pattern
that looked like an architectural difference:

| filings naming | Qwen 3 32B (dense) | Qwen 3 30B-A3B (MoE) |
|---|---|---|
| 1-9 counterparties | 0.882 | 1.000 |
| 10 or more | 0.789 | 0.658 |

Read as a bucket average this says a mixture-of-experts model is perfect on
short lists and loses a third on long ones while the dense model degrades
gracefully. Per filing it says something much weaker:

| ticker | gold links | dense recall | MoE recall |
|---|---|---|---|
| BWA | 4 | 1.000 | 1.000 |
| AMD | 6 | 0.667 | 1.000 |
| LRCX | 7 | 1.000 | 1.000 |
| BA | 14 | 0.429 | **0.857** |
| NVDA | 24 | 1.000 | **0.542** |

The "10 or more" bucket contains two filings and they disagree: on Boeing the
MoE doubles dense recall, on NVIDIA it halves it. The bucket average
manufactured a trend out of two contradictory points.

## What the observation does rule out

One confound is already dead, and it is the one the literature would reach for
first. Prompt length is effectively constant across these filings -- 12,844 to
13,984 characters, because `passages.select` caps the budget at 14,000 -- and
its correlation with gold link count is **-0.296**, slightly negative. Whatever
is happening is not driven by input length, so the position and
context-length literature below does not explain it.

## Prior work

### Long-list extraction is a recognised problem

*Recall Them All* (arXiv 2405.02732) is the closest work. It targets extraction
of long object lists from long documents, notes that "cues for relevant objects
can be spread across many passages", and proposes L3X: recall-oriented
generation with retrieval augmentation, then precision-oriented pruning. It
establishes that the problem exists and offers a method that beats LLM-only
generation.

What it does not do: quantify how recall degrades as a function of the number of
ground-truth items, or compare architectures. It treats low recall as a
condition to be fixed rather than a behaviour to be characterised.

### Position and context length are well covered, and are a different axis

Lost-in-the-middle is established: models degrade on evidence in middle
positions (arXiv 2511.13900), and extraction over long documents shows
higher redundancy for middle content (arXiv 2404.04068). "Context rot" describes
accuracy declining as input grows even with evidence fixed and favourably
placed.

All of these vary **input length or evidence position**. The observation here
holds input length fixed and varies the number of items to be *produced*. That
is an output-side quantity, and the measured correlation of -0.296 says the two
axes are not confounded in this sample.

### Schema breadth, which is adjacent but not the same

*ExtractBench* (arXiv 2602.12247, KDD '26, Contextual AI) evaluates PDF-to-JSON
extraction on 35 documents and 12,867 fields, and reports that performance
"degrades sharply with schema breadth" -- 0% valid output on a 369-field
financial reporting schema. It separates omission from hallucination as failure
types, and evaluates frontier models.

Schema breadth is the number of *distinct fields* in the schema. Cardinality is
the number of *items in one repeated array* under a fixed schema. A 369-field
schema strains a model differently from a 24-element list under a two-field
schema. The finding is suggestive that output-side complexity degrades
extraction, and it is not a measurement of the axis here.

### Constrained decoding has a measured cost

Forcing structured output costs accuracy: reported drops of 3-9 percentage
points across open-weight models, and one function-calling result where
unconstrained generation with post-hoc parsing (93.63%) beat constrained
decoding (91.37%) -- always-valid JSON that is less often right. The mechanism
offered is distributional: masking tokens the model wants renormalises the
remainder.

This matters here because every configuration in the observation runs under
schema-constrained decoding. If constrained decoding interacts with list
length -- if the cost of staying inside the grammar grows with the number of
array elements -- then the effect would be a property of the decoding
constraint rather than of the architecture. **No configuration has been run
without constrained decoding**, so this is currently unfalsified and is the
strongest competing explanation.

### Mixture-of-experts behaviour

MoE-versus-dense work concerns efficiency, capacity and long-range dependency
handling, with one reported result that dense models catch up to MoE as
sequence length grows (arXiv 2509.10530). Token-level routing is described as
producing experts specialised on shallow syntactic features against
sequence-level routing capturing semantic concepts (EMNLP 2023).

Nothing found evaluates MoE against dense on *structured output completeness*.

## The gap, stated narrowly

Cardinality -- how many items a model is asked to emit under a fixed schema and
fixed input length -- does not appear to be treated as an evaluation axis in its
own right. Long-list extraction is treated as a problem to solve, schema breadth
is measured, position and input length are measured. The number of items,
holding everything else fixed, is not.

That is a gap in the evaluation literature. It is a narrow one, and filling it
is characterisation rather than method: the honest description is "a measurement
axis nobody isolated", not a new technique or a boundary being pushed.

## What has to be true for this to be worth weeks

Three conditions, in order. Failing any one of them ends it.

1. **The effect exists.** At n >= 40 filings spanning 1 to 30+ counterparties,
   recall must vary systematically with item count. Two filings disagreeing is
   not evidence. This is a real chance of a null and Boeing already hints at it.
2. **It is not the decoding constraint.** The same curve must be measured with
   constrained decoding off and post-hoc parsing on. If unconstrained generation
   is flat, the finding is about grammar masking, not architecture -- still
   publishable, and a different paper.
3. **It is not Qwen-specific.** A second architecture family, dense and MoE, has
   to show it. One model family is an anecdote.

If all three hold there is a mechanism question worth asking -- routing
instability under repeated same-type tokens, attention dilution across many
identical structures -- and a mitigation with an effect size, since splitting
passages should recover recall if the driver is count.

If (1) fails, the honest output is a short negative note: bucket averages over
small benchmarks manufacture trends, here is the case study.

---

## Literature check owed to the constructed design — done 2026-09-10

`HYPOTHESIS_CARDINALITY.md` records this search as owed before the pilot scaled.
It was run, and **it makes the claim smaller.**

### Multi-needle NIAH already reports the phenomenon

LangChain's multi-needle in a haystack work varies the number of inserted facts
and reports directly that as the number of needles increases, retrieval
decreases, and that reasoning over retrieved needles is worse than retrieval
alone. It also reports the degradation beginning at shorter contexts in the
multi-needle case (~25k tokens) than the single-needle case (~73k for GPT-4).

That is the same qualitative claim as the hypothesis here. **"Nobody has isolated
cardinality" is therefore wrong as stated in the section above, and is
withdrawn.** The phenomenon is known.

### What the search did not turn up

No work found characterises recall as a function of item count for **structured
extraction under constrained decoding**, holding input length fixed:

* ExtractBench (arXiv 2602.12247) measures **schema breadth** — number of
  distinct fields — not items in one repeated array.
* JSONSchemaBench (arXiv 2501.10868) measures validity, coverage and efficiency
  of constrained-decoding frameworks over 10k schemas, not recall against gold
  as a function of array length.
* Multi-needle NIAH is retrieval QA over synthetic inserted facts with context
  length varying alongside needle count. The output is an answer, not a
  schema-constrained array, so the grammar-masking mechanism this project flags
  as its strongest competing explanation cannot arise there.

### Honest positioning, replacing the claim above

This is **a replication in a different output modality plus a
characterisation**, not the discovery of an unmeasured axis:

1. Replicates a known retrieval-QA effect in schema-constrained extraction.
2. Holds input length exactly constant by construction, which the multi-needle
   result does not — its degradation is entangled with context length.
3. Estimates a **slope** (recall per doubling of `k`) rather than reporting that
   a decline exists.
4. Decomposes it across architecture x reasoning mode, which no found work does.
5. Tests a mitigation that follows from the mechanism — input splitting.

Items 2 through 5 are the contribution and they are real, but the paper's
framing must lead with replication, not discovery. A reviewer who knows the
multi-needle result and reads a discovery framing will stop reading there.

### Framing, corrected again — 2026-09-10

The section immediately above overstated the downgrade. "Replication plus
characterisation, never discovery" reads as a retreat, and that is the wrong
posture.

The multi-needle result is an engineering blog post: GPT-4 and a handful of
comparators, synthetic needles inserted into essays, retrieval QA, with needle
count and context length varying together. It is good work and it establishes
that the phenomenon is real. It is not a controlled study of the axis, and it is
not in this modality.

Building on it is what the literature is for. The standing position:

* **Cite multi-needle prominently and early**, as the result that motivates the
  question. Do not bury it, and do not claim the phenomenon as new.
* **Claim the controlled measurement**, which is what is actually new here:
  input length constant by construction rather than co-varying; a slope in
  recall per doubling of `k` rather than an existence claim; real domain
  sentences rather than inserted trivia; schema-constrained output, where
  grammar masking is a candidate mechanism that cannot arise in free-form QA;
  and a 2x2 that separates architecture from reasoning mode.
* **Claim the mitigation**, if input splitting recovers the items, because a
  measured axis with a working intervention is a usable result rather than a
  characterisation.

An extension that cites its antecedent clearly is a normal contribution. The
failure mode to avoid is a discovery framing that a reviewer can puncture with
one link, not the act of building on someone else's finding.
