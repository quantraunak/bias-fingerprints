"""Guards on the local model client, and on the failure that returns nothing.

Every defect this module can have is silent by construction. `generate` returns
a dataclass, the caller parses whatever text is in it, and an empty string
parses cleanly into zero links -- which is indistinguishable from a filing that
genuinely discloses no relationship. Roughly half of all filings genuinely
disclose none, so there is no downstream signal that anything went wrong.

That is not hypothetical. Scoring Qwen 3 30B-A3B produced ten blank extractions
and an F1 of zero, because Ollama files a hybrid reasoning model's entire
structured answer under `thinking` and returns an empty `response` unless
`think` is explicitly False. Had the corpus been started on it, it would have
cached two thousand blanks as legitimate zero-link filings and produced a graph
with no edges and no error.
"""

from __future__ import annotations

import io
import json

import pytest

from src.graph import local_model


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(local_model.time, "sleep", lambda *_: None)


def _reply(monkeypatch, **payload):
    """Stub Ollama's response, capturing the request body for inspection."""
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["body"] = json.loads(request.data)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(local_model.urllib.request, "urlopen", fake_urlopen)
    return sent


SCHEMA = {"type": "object"}


def _generate(**kwargs):
    return local_model.generate("sys", "prompt", SCHEMA, **kwargs)


def test_a_normal_response_is_returned(monkeypatch):
    _reply(monkeypatch, response='{"links": []}', eval_count=12)
    result = _generate()
    assert result.ok and result.text == '{"links": []}'


def test_an_answer_filed_under_thinking_is_recovered(monkeypatch):
    """The Qwen 3 30B-A3B case: the whole answer lands in the wrong field."""
    _reply(monkeypatch, response="", thinking='{"links": [{"counterparty": "Intel"}]}',
           eval_count=36)
    result = _generate()
    assert result.ok
    assert json.loads(result.text)["links"][0]["counterparty"] == "Intel"


def test_a_blank_after_real_generation_fails_loudly(monkeypatch):
    """Tokens generated but nothing readable is an error, never an empty result.

    Returning ok=True with text="" here is what would have written two thousand
    blank filings into the cache.
    """
    _reply(monkeypatch, response="", thinking="", eval_count=36)
    result = _generate()
    assert not result.ok
    assert "no readable text" in (result.error or "")


def test_a_genuinely_empty_generation_is_not_an_error(monkeypatch):
    """Zero tokens generated is a different condition and must not be conflated."""
    _reply(monkeypatch, response="", thinking="", eval_count=0)
    result = _generate()
    assert result.ok and result.text == ""


def test_whitespace_only_counts_as_blank(monkeypatch):
    _reply(monkeypatch, response="   \n  ", thinking="", eval_count=20)
    assert not _generate().ok


def test_a_populated_response_wins_over_thinking(monkeypatch):
    """When the model uses both fields, the answer is the one in `response`."""
    _reply(monkeypatch, response='{"links": []}', thinking="let me consider...",
           eval_count=40)
    assert _generate().text == '{"links": []}'


def test_hitting_the_token_ceiling_is_flagged(monkeypatch):
    """A truncated response is a filing we did not read, not a filing with fewer links."""
    _reply(monkeypatch, response='{"links": [{"counterp', eval_count=500)
    assert _generate(num_predict=500).truncated


def test_a_response_under_the_ceiling_is_not_flagged(monkeypatch):
    _reply(monkeypatch, response='{"links": []}', eval_count=499)
    assert not _generate(num_predict=500).truncated


def test_think_is_only_sent_when_set(monkeypatch):
    """`think` unset and `think=False` are different requests to Ollama.

    They are also different cache keys in `score_extraction.py`, so conflating
    them would silently score one configuration as though it were the other.
    """
    sent = _reply(monkeypatch, response="{}", eval_count=1)
    _generate()
    assert "think" not in sent["body"]

    sent = _reply(monkeypatch, response="{}", eval_count=1)
    _generate(think=False)
    assert sent["body"]["think"] is False


def test_decoding_is_deterministic(monkeypatch):
    """Sampling variance would make edges appear and disappear between runs."""
    sent = _reply(monkeypatch, response="{}", eval_count=1)
    _generate()
    assert sent["body"]["options"]["temperature"] == 0
    assert sent["body"]["format"] == SCHEMA


def test_a_transport_failure_is_retried_then_reported(monkeypatch):
    attempts = []

    def flaky(request, timeout=None):
        attempts.append(1)
        raise OSError("connection refused")

    monkeypatch.setattr(local_model.urllib.request, "urlopen", flaky)
    result = _generate(retries=2)
    assert not result.ok and len(attempts) == 3
    assert "connection refused" in (result.error or "")
