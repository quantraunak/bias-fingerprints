# Experiment 1: does extraction recall depend on how many items are asked for?

Pre-registered before the data. `RELATED_WORK.md` states the gap and the three
conditions that have to hold for this to be worth continuing; this is the test
of the first one.

## Question

Holding input length, schema and prompt fixed, does recall vary systematically
with the number of ground-truth items in a document?

## Why it is not already answered

The observation that motivates it rests on two filings that disagree -- Boeing
at 14 items where the MoE beats the dense model, NVIDIA at 24 where it loses a
third. Benchmark v2 has exactly two filings above nine items, so the "10+"
bucket in `extraction_table.md` is those two averaged. That is not a curve.

## The binding constraint, and what it forces

Recall requires ground truth. The corpus has plenty of documents at the
interesting cardinalities:

| extracted items | filings available |
|---|---|
| 0 | 75 |
| 1-3 | 35 |
| 4-9 | 24 |
| 10-19 | 12 |
| 20+ | 17 |

but only ten are annotated, and only two of those exceed nine gold items:

    gold counts across benchmark v2:  0 0 0 0 0 4 6 7 14 24

So **the experiment is gated on annotation, not on GPU**. Week 1 is annotation
work. This is the opposite of every other stage of this project, where compute
was the constraint and labels were assumed.

One favourable property, measured rather than assumed: across the 163 extracted
filings the correlation between item count and prompt length is **0.032**.
`passages.select` caps the prompt at 14,000 characters, so long-list filings are
not systematically longer inputs. The confound the position-and-context-length
literature would predict is naturally absent, and the design does not have to
engineer it away.

## Design

**Sample.** 40 filings, stratified by *extracted* item count into five bands
(0, 1-3, 4-9, 10-19, 20+), eight per band, drawn with a fixed seed from the 163
already extracted. Stratifying on the model's own output is a known bias -- the
sample is conditioned on what one extractor found -- and it is accepted
deliberately: drawing uniformly would put almost nothing above nine items, which
is precisely the region under test. The bias is recorded here rather than
discovered later, and the 0-item band is retained as the control that keeps
precision measurable.

**Annotation.** Same rules as benchmark v2, in `SPECIFICATIONS.md`: verbatim
evidence, the four annotation rulings A1-A4, every excluded name recorded with
the rule that excluded it. Annotation is over `passages.select` output, not
whole filings, matching the existing benchmark's scope.

**Double annotation.** 20% of the sample (8 filings) annotated twice, with
agreement reported by `scripts/annotator_agreement.py`. Without this the whole
curve rests on one person's reading, and a reviewer stops there. Cohen's kappa
on the disclosure decision, pairwise F1 on link sets, direction agreement on
shared counterparties.

**Measurement.** Recall per filing against gold, plotted against gold item
count, for the dense model and the MoE. Both already have cached responses for
some of these filings; the rest is one extraction pass each.

## What counts as an answer

Fixed now, so the result cannot be read favourably after the fact.

| outcome | reading |
|---|---|
| Both models flat across bands | **Null.** The bucket average was an artifact. Write the short negative note and stop. |
| Both models decline together | Real, but architecture-independent -- a property of extraction under this prompt, not of MoE. Condition 3 becomes irrelevant; condition 2 becomes the whole question. |
| MoE declines, dense does not | The motivating observation survives. Proceed to conditions 2 and 3. |
| Dense declines, MoE does not | The observation was backwards, which is a finding and is reported as one. |

Monotonicity is not required. A cliff at a threshold and a smooth decline are
different mechanisms, and distinguishing them is the point of a curve rather
than two buckets.

**Power.** Eight filings per band is enough to see a difference of roughly 0.25
in mean recall between the extreme bands, not enough to resolve 0.05. The
experiment is built to detect a cliff, not to measure its slope. If the effect
is real but small, the honest output is "underpowered, effect direction
consistent, needs more annotation" rather than a claimed curve.

## Threats, in the order they would invalidate the result

1. **Constrained decoding.** Every configuration runs schema-constrained
   decoding, which has a measured accuracy cost, and that cost may itself grow
   with the number of array elements. This is condition 2 in `RELATED_WORK.md`
   and it is the strongest competing explanation. It is not tested in this
   experiment, and no architectural claim survives without it.
2. **Selection on extractor output.** The sample is stratified on what the dense
   model found. A filing where both models fail entirely is invisible to the
   sample, which biases toward documents that are extractable at all.
3. **Item count confounded with document character.** A filing naming twenty
   counterparties may name them in one dense enumeration or scattered across
   twenty pages. These are different tasks. Recording which, per filing, is
   cheap during annotation and lets the two be separated afterwards.
4. **One annotator.** Mitigated by the double-annotated subset, not removed.

## Cost

Annotation of 40 filings, plus 8 re-annotated. No GPU beyond one extraction pass
for filings not already cached. This is days of human work and hours of machine
work, which is the inverse of everything attempted so far.

## What this is not

Not a method, not a new technique, and not a boundary being pushed. It is a
measurement of an axis the evaluation literature has left unisolated. If it
comes out flat, that is the finding and it gets written up as one.
