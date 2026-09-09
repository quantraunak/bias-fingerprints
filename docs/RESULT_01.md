# Result 1: extraction recall does not depend on how many items are present

The cardinality hypothesis in `EXPERIMENT_01.md` is rejected. Recall does not
decline as a document discloses more relationships. The observation that
motivated the question was a bucket average over two filings, and it does not
survive measurement across fourteen.

## What was predicted, and what happened

`EXPERIMENT_01.md` fixed four possible outcomes in advance. This is the first
row of that table:

> Both models flat across bands → **Null.** The bucket average was an artifact.
> Write the short negative note and stop.

| filings | items disclosed | mean recall |
|---|---|---|
| AEE, ETR, HUM x3, GE, CAH, BWA | 3-4 | **0.323** |
| NVDA, CAT, ON, VIAV, NVDA, CMI | 34-51 | **0.478** |

Recall is *higher* at high cardinality, not lower. The correlation between item
count and recall is **+0.400** (Spearman, p = 0.156) -- positive, and not
distinguishable from zero. A two-sample test between the bands gives p = 0.442.

Nothing here supports a cardinality effect in either direction.

## Where the variance actually is

Within-band spread exceeds between-band difference: standard deviation 0.394 at
3-4 items and 0.313 at 34-51, against a 0.155 gap between the band means. The
dominant factor is **which filing**, not how many items it holds.

The pattern is sectoral. Utilities and health insurance (AEE, ETR, HUM) return
zero; industrials and semiconductors (GE, CAT, ON) return 0.72 to 1.00. That
holds at both ends of the cardinality range, which is why it cannot be a
cardinality effect wearing a sector disguise.

## What the original observation was

`extraction_table.md` reports MoE recall of 0.658 on filings naming ten or more
counterparties against 0.789 for the dense model. That cell is two filings --
Boeing at 14 items where the MoE scores 0.857, NVIDIA at 24 where it scores
0.542 -- averaged together. `RELATED_WORK.md` flagged the arithmetic before this
experiment ran; the experiment confirms there is nothing underneath it.

A bucket average over two observations is not a measurement, and presenting it
as a curve was an error in reading the table rather than in producing it.

## The instrument

Ablation of real filings, in `src/graph/ablate.py`. A filing's relationship
sentences are located and a subset replaced with neutral prose from the same
filing, so cardinality varies while length, voice and section structure hold.
The control passes: ablating to full cardinality returns the document
byte-identical, and length holds within 8% from one item to fifty-one.

This replaced a synthetic-document approach that failed for an instructive
reason. Shuffling real sentences from many filers into one document produces
text that is not a 10-K and that the model does not read as one: recall at a
single planted item was 1/6, against 0.66-1.00 for the same model on real
filings. The construction dominated the effect under study. Two rounds of
filtering did not fix it, because incoherence was the method rather than a
defect in it.

The failure was caught by a baseline control -- extract from an unmodified
document and check the number is plausible -- run before the curve. Every
earlier instrument in this project ran its control afterwards, or not at all.

## Limits on this null

The measurement is honest about what it cannot exclude.

- **Fourteen filings.** Enough to reject a large effect, not enough to exclude a
  small one. A 0.1 decline across the range would not be visible here.
- **The reference set is model-derived.** Ground truth is the dense model's
  validated extractions, so this measures MoE-against-dense agreement rather
  than recall against annotated truth. It is the same reference type at both
  ends of the range, which is what makes the comparison internally valid, but
  it is not gold.
- **One model pair, one architecture family.** Qwen 3 32B dense against Qwen 3
  30B-A3B. Nothing here transfers to another family without measurement.
- **The corpus caps at 51 items.** Larger enumerations are untested, and a
  ceiling above that range would be invisible.

## The finding that replaced it

The data says something the hypothesis did not anticipate: **the two models
disagree about what counts as a relationship, and the disagreement is
sectoral.**

VIAV is the clean case. The dense model returns 42 items -- Adva,
Alcatel-Lucent, Ciena, Cisco, Ericsson, Fujitsu, Huawei, Infinera, Nokia
Siemens, Tellabs -- which is a competitive-landscape section, not a customer or
supplier list. The MoE returns nothing at all.

This is not a capacity difference. It is a taxonomy difference, and it is
material to the pipeline: 312 of 933 corpus claims carry the `competitor`
relation, and `build.edges` discards every one of them before the graph is
built. A third of what one extractor produces is not an edge, and the two
models differ on how much of it to produce.

Testing that costs one re-measurement with the reference set restricted to
`customer`, `supplier` and `partner`. It is not run here, and it is recorded as
the open question rather than as a result.
