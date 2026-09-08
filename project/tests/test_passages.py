"""Guards on passage selection.

`passages.select` is the only thing that decides what the model ever reads. A
defect here is invisible downstream: the extractor returns valid, well-formed,
correctly grounded claims about the passages it was given, and simply never sees
the sentence that named the counterparty. Recall is capped before the model
starts, and no extraction metric can distinguish that from a model that missed
the link -- which is why the scoring regressions below are tested directly
rather than inferred from precision and recall on the benchmark.

The failure that motivated the current scorer is recorded in `score`'s
docstring: keyword-density selection ranked generic risk-factor boilerplate --
saturated with "customers" and "suppliers" while naming nobody -- above the
paragraph that named the foundry, and every counterparty of interest in the
NVIDIA filing was pushed out of the budget. Those cases are the tests.
"""

from __future__ import annotations

import pytest

from src.graph import passages


# --------------------------------------------------------------------------
# sections: get the body of Items 1, 1A and 7, or give up honestly
# --------------------------------------------------------------------------

def test_heading_is_taken_from_the_body_not_the_table_of_contents():
    """Item headings appear twice. Anchoring on the first keeps the TOC."""
    document = (
        "TABLE OF CONTENTS\nItem 1. Business .... 3\nItem 1A. Risk Factors .... 12\n"
        + "FRONT MATTER " * 200
        + "\nItem 1. Business\n" + "REAL BODY " * 4000
        + "\nItem 1A. Risk Factors\n" + "RISK BODY " * 4000
    )
    kept = passages.sections(document)
    assert "FRONT MATTER" not in kept
    assert "REAL BODY" in kept and "RISK BODY" in kept


def test_an_unparsed_filing_falls_back_to_the_whole_document():
    """No heading match must widen the net, never empty it."""
    document = "This filer uses no recognisable item headings at all."
    assert passages.sections(document) == document


def test_a_late_cross_reference_falls_back_rather_than_keeping_a_sliver():
    """"...as described in Item 1A" near the end is a reference, not a section.

    Anchoring on it would keep the last few percent of the filing and discard
    Item 1 entirely -- a silent near-total loss of the passages that matter.
    """
    document = "BODY " * 20000 + " as described in Item 1A. Risk Factors above."
    assert passages.sections(document) == document


def test_a_real_section_is_narrower_than_the_filing():
    """The fallback must not be so eager that sectioning never happens."""
    document = (
        "COVER PAGE " * 5000
        + "\nItem 1. Business\n" + "BUSINESS BODY " * 5000
        + "\nItem 8. Financial Statements\n" + "TABLES " * 5000
    )
    kept = passages.sections(document)
    assert len(kept) < len(document)
    assert "COVER PAGE" not in kept


# --------------------------------------------------------------------------
# paragraphs: units that do not depend on how a filer wrapped its HTML
# --------------------------------------------------------------------------

def test_a_filing_with_no_blank_lines_still_splits():
    """The Walmart case: one 65k-char block, no blank line anywhere in it.

    Blank-line splitting produced a single unit far outside any length bound,
    so nothing was ever selected and the filing contributed no claims at all.
    """
    blob = " ".join(
        f"This is sentence number {i} about our customers and suppliers." for i in range(400)
    )
    assert "\n" not in blob
    units = passages.paragraphs(blob)
    assert len(units) > 1


def test_units_are_length_bounded_so_scores_are_comparable():
    """A score threshold only means the same thing everywhere if units do."""
    blob = " ".join(f"Sentence {i} concerns our suppliers and our customers." for i in range(600))
    units = passages.paragraphs(blob, min_chars=120, target_chars=1200)
    body = units[:-1]  # the tail is whatever is left over
    assert all(1200 <= len(u) < 1600 for u in body)
    assert all(len(u) >= 120 for u in units)


def test_boilerplate_units_are_dropped():
    assert passages.paragraphs("Forward-looking statements. " + "filler " * 200) == []
    assert passages.paragraphs("See note 7 to the financial statements. " + "filler " * 200) == []


def test_short_fragments_are_dropped():
    assert passages.paragraphs("Too short.") == []


# --------------------------------------------------------------------------
# company_candidates: capitalised runs that could name a company
# --------------------------------------------------------------------------

def test_a_sentence_initial_capital_is_not_a_company():
    """Without this the scorer cannot tell Samsung from "However, our...".

    A single capitalised token opening a sentence is sentence furniture. It is
    also the single most common capitalised run in a filing, so admitting it
    would swamp the real names.
    """
    found = passages.company_candidates(
        "Our foundry partner is important. However our customers vary widely by region."
    )
    assert found == []


def test_a_corporate_suffix_rescues_a_single_token():
    names = [name for _, name in passages.company_candidates(
        "We buy wafers. Intel Corporation supplies them to us under contract."
    )]
    assert any("Intel" in n for n in names)


def test_section_furniture_and_geographies_are_not_companies():
    for phrase in ("Item 1A. Risk Factors follow below.",
                   "Our customers in China and Japan and Taiwan are many."):
        names = [n for _, n in passages.company_candidates(phrase)]
        assert not any(n in {"Risk Factors", "China", "Japan", "Taiwan"} for n in names)


def test_a_multi_token_name_mid_sentence_is_kept():
    names = [n for _, n in passages.company_candidates(
        "We depend on Taiwan Semiconductor Manufacturing Company for wafer supply."
    )]
    assert "Taiwan Semiconductor Manufacturing Company" in names


@pytest.mark.xfail(
    strict=True,
    reason="CAPITALISED_RUN spans sentence boundaries: '...by Intel. Samsung also...' "
           "yields the single candidate 'Intel. Samsung', which resolves to neither. "
           "Measured on 60 corpus filings: 438 candidates span a boundary and 16 "
           "become resolvable once split, but no filing's prescreen verdict changes, "
           "so this costs recall at the margin rather than skipping filings. Not fixed "
           "while extraction is running -- rebuild_claims re-runs select() to "
           "re-validate cached responses, so changing selection retroactively changes "
           "which cached claims survive the verbatim-quote check.",
)
def test_a_name_is_not_merged_across_a_sentence_boundary():
    names = [n for _, n in passages.company_candidates(
        "We are supplied by Intel. Samsung also serves as a supplier to us today."
    )]
    assert "Intel" in names or "Samsung" in names
    assert not any("." in n and " " in n for n in names)


# --------------------------------------------------------------------------
# score: named entities near relationship language, not keyword density
# --------------------------------------------------------------------------

BOILERPLATE = (
    "Our customers and suppliers and distributors and resellers are OEMs and ODMs. "
    "We depend on customers. Our suppliers and vendors and partners are third parties. "
) * 4

NAMED = (
    "We rely on Taiwan Semiconductor Manufacturing Company as our principal foundry "
    "partner, and Samsung Electronics supplies memory to us under a long term agreement. "
) * 3


def test_boilerplate_that_names_nobody_scores_zero():
    """Saturated with relationship vocabulary, naming no one. The whole point."""
    assert passages.score(BOILERPLATE) == 0.0


def test_a_named_counterparty_outranks_boilerplate():
    """The NVIDIA regression: this comparison is what the scorer exists to get right."""
    assert passages.score(NAMED) > passages.score(BOILERPLATE)


def test_a_name_with_no_relationship_language_scores_zero():
    """A name in a lease schedule or a director's biography is not a link."""
    assert passages.score(
        "The board appointed Susan Green of Lucent Technologies to the audit committee "
        "in March, and she previously served at Ericsson and at Medtronic before that."
    ) == 0.0


def test_quantified_concentration_outweighs_a_generic_mention():
    """A 10%-of-revenue disclosure is the highest-yield sentence in a 10-K."""
    strong = "Sales to Apple Computer accounted for 23% of net revenue in fiscal 2019."
    weak = "Sales to Apple Computer are made through our normal distributors and resellers."
    assert passages.score(strong) > passages.score(weak)


def test_generic_plural_density_taxes_the_score():
    """Two passages naming the same company; the one padded with categories loses."""
    clean = "We rely on Intel Corporation as our largest customer for server parts."
    padded = clean + " " + "Our customers and suppliers and distributors and OEMs vary. " * 3
    assert passages.score(padded) < passages.score(clean)


# --------------------------------------------------------------------------
# select: the budget, the ordering, and the recall guarantee that matters
# --------------------------------------------------------------------------

FILLER = "The quick brown fox jumped over the lazy dog in the meadow yesterday. "


def test_a_named_counterparty_survives_selection():
    """End-to-end recall guard.

    If a filing names a counterparty next to relationship language and `select`
    drops the passage, every extractor downstream is capped and nothing reports
    it. This is the assertion the rest of the module exists to support.
    """
    document = FILLER * 60 + NAMED * 4 + FILLER * 60
    selected = passages.select(document)
    assert any("Taiwan Semiconductor Manufacturing Company" in p for p in selected)


def test_selected_passages_are_returned_in_document_order():
    """The model reads them as prose; score order would scramble the argument."""
    document = FILLER * 40 + NAMED * 20 + FILLER * 40 + NAMED * 20
    selected = passages.select(document, budget_chars=3000)
    positions = [document.index(p) for p in selected]
    assert positions == sorted(positions)


def test_the_character_budget_is_respected():
    document = FILLER * 40 + NAMED * 60
    selected = passages.select(document, budget_chars=3000)
    assert sum(len(p) for p in selected) <= 3000


def test_a_filing_naming_nobody_selects_nothing():
    """Better to skip the call than to send the model a page of lease terms."""
    assert passages.select(FILLER * 200) == []


# --------------------------------------------------------------------------
# prescreen: is a model call worth fourteen seconds
# --------------------------------------------------------------------------

def _resolver(known: dict):
    def resolve(name, lookup):
        return known.get(name), "exact"
    return resolve


TSMC = _resolver({"Taiwan Semiconductor Manufacturing Company": "TSM",
                  "Intel Corporation": "INTC"})


def test_a_resolving_name_near_relationship_language_passes():
    passed, hits = passages.prescreen(
        ["We depend on Taiwan Semiconductor Manufacturing Company for wafer supply."],
        None, TSMC, "NVDA",
    )
    assert passed and hits == 1


def test_a_resolving_name_far_from_relationship_language_does_not_pass():
    """The version without this check skipped only 4% of filings.

    Every filing mentions some capitalised name that collides with a ticker.
    Proximity is what makes the screen mean "a relationship may be described
    here" rather than "a ticker-shaped string occurs somewhere".
    """
    far = ["Taiwan Semiconductor Manufacturing Company is mentioned here. "
           + "padding " * 400 + " and separately our customers vary by region."]
    assert passages.prescreen(far, None, TSMC, "NVDA") == (False, 0)


def test_the_filer_itself_is_not_a_counterparty():
    """A self-loop is not an economic link, and G1 verifies the graph has none."""
    passage = ["We purchases from Intel Corporation regularly and rely upon them for wafers."]
    assert passages.prescreen(passage, None, TSMC, "NVDA") == (True, 1)
    assert passages.prescreen(passage, None, TSMC, "INTC") == (False, 0)


def test_quantified_concentration_passes_without_a_resolving_name():
    """Deliberately generous: the model reads names the regex cannot catch."""
    passed, hits = passages.prescreen(
        ["One customer accounted for 23% of net revenue during fiscal 2019."],
        None, _resolver({}), "NVDA",
    )
    assert passed and hits == 0


def test_a_filing_with_neither_is_skipped():
    assert passages.prescreen(
        ["Our customers are many and varied across a range of end markets."],
        None, _resolver({}), "NVDA",
    ) == (False, 0)
