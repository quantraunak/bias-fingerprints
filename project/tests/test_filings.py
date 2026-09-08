"""Guards on filing acquisition and text extraction.

Every failure this module can have is a *quiet* one. A dropped connection, a
stale ticker-to-CIK mapping or an over-eager table stripper does not raise: it
produces a corpus that is simply smaller than it should be, and a graph built
from a smaller corpus looks exactly like a graph built from a complete one. The
XOM case is the record of that -- sixteen 10-Ks missing for months because the
symbol resolved to a post-reorganisation CIK with no filing history, and nothing
anywhere said so.

Nothing here touches the network. `_get` is replaced with a stub in every test
that would otherwise reach EDGAR, and the cache directory is a tmp_path, so the
suite is deterministic and runs offline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.graph import filings


# --------------------------------------------------------------------------
# fixtures: a fake EDGAR
# --------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _block(rows: list[dict]) -> dict:
    """Shape a list of filings the way EDGAR's submissions index does."""
    return {
        "form": [r["form"] for r in rows],
        "filingDate": [r["filed"] for r in rows],
        "reportDate": [r.get("period") for r in rows],
        "accessionNumber": [r["accession"] for r in rows],
        "primaryDocument": [r.get("document", "d.htm") for r in rows],
    }


RECENT = _block([
    {"form": "10-K", "filed": "2020-02-14", "period": "2019-12-31", "accession": "0000-20-000001"},
    {"form": "10-K/A", "filed": "2020-05-01", "period": "2019-12-31", "accession": "0000-20-000002"},
    {"form": "10-Q", "filed": "2020-08-01", "period": "2020-06-30", "accession": "0000-20-000003"},
    {"form": "8-K", "filed": "2020-09-01", "period": None, "accession": "0000-20-000004"},
])

# The overflow page: older filings that fall outside the `recent` window.
OLDER = _block([
    {"form": "10-K", "filed": "2011-03-01", "period": "2010-12-31", "accession": "0000-11-000001"},
    {"form": "10-K", "filed": "2012-03-01", "period": None, "accession": "0000-12-000001"},
])


@pytest.fixture
def edgar(monkeypatch):
    """A stubbed `_get` covering submissions, overflow pages and documents."""
    calls = []

    def fake_get(url, user_agent, timeout=60, retries=4):
        calls.append(url)
        # Checked before the index URL: real overflow names begin with "CIK" too.
        if "-submissions-" in url:
            return FakeResponse(payload=OLDER)
        if "submissions/CIK" in url:
            return FakeResponse(payload={
                "filings": {"recent": RECENT,
                            "files": [{"name": "CIK0000001234-submissions-001.json"}]}
            })
        return FakeResponse(text="<html><body><p>" + "filing body " * 2000 + "</p></body></html>")

    monkeypatch.setattr(filings, "_get", fake_get)
    return calls


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(filings, "FILINGS_DIR", tmp_path / "filings")
    monkeypatch.setattr(filings, "INDEX_PATH", tmp_path / "filings" / "_index.parquet")
    return tmp_path / "filings"


# --------------------------------------------------------------------------
# to_text: keep the prose, drop the tables, keep the headings
# --------------------------------------------------------------------------

def test_long_data_tables_are_dropped():
    """Tables are most of a 10-K's bulk and almost none of its relationship prose."""
    rows = " ".join(f"<td>row{i}</td><td>123.45</td>" for i in range(80))
    text = filings.to_text(f"<html><body><p>Keep this.</p><table><tr>{rows}</tr></table></body></html>")
    assert "Keep this." in text
    assert "123.45" not in text


def test_short_tables_are_kept_because_headings_live_in_them():
    """Caterpillar's "Item 1. Business" is a table cell, not a paragraph.

    Dropping every table erased the section headings for those filers, which
    left `passages.sections` with nothing to anchor on and silently fell the
    whole filing back to unsectioned text.
    """
    html = "<html><body><table><tr><td>Item 1. Business</td></tr></table><p>Body.</p></body></html>"
    assert "Item 1. Business" in filings.to_text(html)


def test_scripts_and_styles_are_removed():
    html = "<html><body><script>var x=1;</script><style>p{color:red}</style><p>Body.</p></body></html>"
    text = filings.to_text(html)
    assert "var x" not in text and "color:red" not in text
    assert "Body." in text


def test_whitespace_is_normalised():
    """Non-breaking spaces survive naively and break every downstream regex."""
    text = filings.to_text("<html><body><p>Intel&nbsp;&nbsp;Corporation   supplies\tus.</p></body></html>")
    assert "\xa0" not in text
    assert "Intel Corporation supplies us." in text


# --------------------------------------------------------------------------
# list_filings: the right forms, the right window, and the older pages
# --------------------------------------------------------------------------

def test_only_annual_reports_are_kept(edgar):
    found = filings.list_filings("AAA", 1234, "ua", "2000-01-01", "2030-01-01")
    assert {f.form for f in found} == {"10-K", "10-K/A"}


def test_overflow_pages_are_followed(edgar):
    """`recent` covers ~1000 filings; long histories spill into paginated files.

    Not following them drops the early years of exactly the large-cap issuers
    the study is built on, and drops them without any error.
    """
    found = filings.list_filings("AAA", 1234, "ua", "2000-01-01", "2030-01-01")
    assert "2011-03-01" in {str(f.filed.date()) for f in found}


def test_the_date_window_is_applied_to_the_filing_date(edgar):
    """Filing date, not period end. The whole point-in-time claim rests on it."""
    found = filings.list_filings("AAA", 1234, "ua", "2012-01-01", "2021-01-01")
    filed = sorted(str(f.filed.date()) for f in found)
    assert filed == ["2012-03-01", "2020-02-14", "2020-05-01"]


def test_results_are_sorted_by_filing_date(edgar):
    found = filings.list_filings("AAA", 1234, "ua", "2000-01-01", "2030-01-01")
    assert [f.filed for f in found] == sorted(f.filed for f in found)


def test_accession_numbers_are_stripped_of_dashes(edgar):
    """The archive URL and the cache filename both want the undashed form."""
    found = filings.list_filings("AAA", 1234, "ua", "2000-01-01", "2030-01-01")
    assert all("-" not in f.accession for f in found)


def test_a_missing_report_date_is_none_not_a_crash(edgar):
    found = filings.list_filings("AAA", 1234, "ua", "2000-01-01", "2030-01-01")
    periods = {f.accession: f.period for f in found}
    assert periods["000012000001"] is None


def test_an_unreachable_issuer_yields_nothing_rather_than_raising(monkeypatch):
    """A persistent failure must cost one issuer, not the whole run."""
    monkeypatch.setattr(filings, "_get", lambda *a, **k: None)
    assert filings.list_filings("AAA", 1234, "ua", "2000-01-01", "2030-01-01") == []


# --------------------------------------------------------------------------
# fetch_text: the cache is the expensive thing
# --------------------------------------------------------------------------

def _filing(**kwargs) -> filings.Filing:
    base = dict(ticker="AAA", cik=1234, accession="000020000001", form="10-K",
                filed=pd.Timestamp("2020-02-14"), period=pd.Timestamp("2019-12-31"),
                document="d.htm")
    base.update(kwargs)
    return filings.Filing(**base)


def test_a_cached_document_is_not_refetched(cache, edgar):
    filing = _filing()
    filing.path.parent.mkdir(parents=True, exist_ok=True)
    filing.path.write_text("cached body")
    assert filings.fetch_text(filing, "ua") == "cached body"
    assert edgar == []  # no request was made


def test_force_refetches_over_a_cached_document(cache, edgar):
    filing = _filing()
    filing.path.parent.mkdir(parents=True, exist_ok=True)
    filing.path.write_text("stale body")
    text = filings.fetch_text(filing, "ua", force=True)
    assert text is not None and "filing body" in text
    assert edgar  # a request was made


def test_a_stub_document_is_rejected_and_not_cached(cache, monkeypatch):
    """A short body is an exhibit or a cover stub, not the filing.

    Caching it would be worse than failing: the run would never retry, and the
    issuer would carry a filing that contributes no passages forever.
    """
    monkeypatch.setattr(filings, "_get", lambda *a, **k: FakeResponse(text="<html><p>stub</p></html>"))
    filing = _filing()
    assert filings.fetch_text(filing, "ua") is None
    assert not filing.path.exists()


def test_a_failed_download_returns_none(cache, monkeypatch):
    monkeypatch.setattr(filings, "_get", lambda *a, **k: None)
    assert filings.fetch_text(_filing(), "ua") is None


# --------------------------------------------------------------------------
# _get: EDGAR drops connections under sustained pulling
# --------------------------------------------------------------------------

@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(filings.time, "sleep", lambda *_: None)


def test_a_transient_server_error_is_retried(monkeypatch, no_sleep):
    """The first bulk run died on a reset after 1,665 documents."""
    attempts = []

    def flaky(url, headers=None, timeout=None):
        attempts.append(url)
        return FakeResponse(status_code=503) if len(attempts) < 3 else FakeResponse(text="ok")

    monkeypatch.setattr(filings.requests, "get", flaky)
    assert filings._get("http://x", "ua").text == "ok"
    assert len(attempts) == 3


def test_a_transport_error_is_retried(monkeypatch, no_sleep):
    import requests as rq
    attempts = []

    def flaky(url, headers=None, timeout=None):
        attempts.append(url)
        if len(attempts) == 1:
            raise rq.exceptions.ConnectionError("reset by peer")
        return FakeResponse(text="ok")

    monkeypatch.setattr(filings.requests, "get", flaky)
    assert filings._get("http://x", "ua").text == "ok"


def test_a_missing_document_is_not_retried(monkeypatch, no_sleep):
    """404 is an answer. Retrying it burns the rate limit for nothing."""
    attempts = []

    def gone(url, headers=None, timeout=None):
        attempts.append(url)
        return FakeResponse(status_code=404)

    monkeypatch.setattr(filings.requests, "get", gone)
    assert filings._get("http://x", "ua") is None
    assert len(attempts) == 1


def test_a_persistent_failure_gives_up_and_returns_none(monkeypatch, no_sleep):
    monkeypatch.setattr(filings.requests, "get",
                        lambda url, headers=None, timeout=None: FakeResponse(status_code=500))
    assert filings._get("http://x", "ua", retries=2) is None


# --------------------------------------------------------------------------
# build_index: the index must survive the run that builds it
# --------------------------------------------------------------------------

@pytest.fixture
def cik_map(monkeypatch):
    monkeypatch.setattr(filings, "load_cik_map", lambda ua: {"AAA": 1234, "XOM": 9999999, "GONE": 5555})


def test_the_index_is_written_and_keyed_by_filing_date(cache, edgar, cik_map):
    index = filings.build_index(["AAA"], "ua", "2000-01-01", "2030-01-01")
    assert filings.INDEX_PATH.exists()
    assert set(index.columns) >= {"ticker", "cik", "accession", "form", "filed", "period", "chars", "path"}
    assert index.filed.is_monotonic_increasing


def test_the_index_is_flushed_per_issuer_not_at_the_end(cache, edgar, cik_map, monkeypatch):
    """A run that crashes mid-corpus must leave a usable index behind.

    The first attempt wrote nothing until the end, crashed after 1,665
    downloads, and left the documents on disk with nothing to say what they
    were.
    """
    flushes = []
    original = filings._flush
    monkeypatch.setattr(filings, "_flush", lambda rows: (flushes.append(len(rows)), original(rows))[1])
    filings.build_index(["AAA", "GONE"], "ua", "2000-01-01", "2030-01-01")
    assert len(flushes) > 1


def test_an_issuer_with_no_cik_is_reported(cache, edgar, cik_map):
    index = filings.build_index(["AAA", "NOSUCH"], "ua", "2000-01-01", "2030-01-01")
    assert index.attrs["no_cik"] == ["NOSUCH"]


def test_an_issuer_that_resolved_but_returned_nothing_is_reported(cache, cik_map, monkeypatch):
    """The XOM symptom, surfaced instead of left to a coverage check nobody runs.

    A stale symbol-to-CIK mapping resolves fine and returns an empty filing
    history. Without this report the issuer just quietly contributes no edges.
    """
    def empty_submissions(url, user_agent, timeout=60, retries=4):
        if "submissions" in url:
            return FakeResponse(payload={"filings": {"recent": _block([]), "files": []}})
        return FakeResponse(text="<html><body>" + "body " * 2000 + "</body></html>")

    monkeypatch.setattr(filings, "_get", empty_submissions)
    index = filings.build_index(["AAA"], "ua", "2000-01-01", "2030-01-01")
    assert index.attrs["no_filings"] == ["AAA"]


def test_a_hand_checked_cik_override_wins_over_the_ticker_file(cache, cik_map, monkeypatch):
    """XOM's symbol points at a 2024 entity with no history; 34088 has sixteen 10-Ks."""
    seen = []

    def record(url, user_agent, timeout=60, retries=4):
        seen.append(url)
        if "submissions" in url:
            return FakeResponse(payload={"filings": {"recent": _block([]), "files": []}})
        return FakeResponse(text="x")

    monkeypatch.setattr(filings, "_get", record)
    filings.build_index(["XOM"], "ua", "2000-01-01", "2030-01-01")
    assert any("0000034088" in url for url in seen)
    assert not any("9999999" in url for url in seen)


def test_rerunning_does_not_duplicate_or_refetch(cache, edgar, cik_map):
    """Resumability: the corpus is measured in days and the run dies often."""
    first = filings.build_index(["AAA"], "ua", "2000-01-01", "2030-01-01")
    before = len(edgar)
    second = filings.build_index(["AAA"], "ua", "2000-01-01", "2030-01-01")
    assert len(second) == len(first)
    assert second.accession.is_unique
    # Only the submissions listings are re-requested; no document is refetched.
    assert all("submissions" in url for url in edgar[before:])
