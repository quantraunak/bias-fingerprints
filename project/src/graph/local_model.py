"""Run extraction against a local Ollama model.

Extraction over the full corpus is thousands of calls. Doing that against a
frontier API costs real money for a result whose value is unknown until the
graph exists, so the volume runs on a local 8B model and a small hand-checked
sample establishes what that costs in quality. `evaluate.py` measures the gap
rather than assuming it is small.

Ollama's `format` parameter takes a JSON schema and constrains decoding, which
matters more for a small model than a large one: without it, an 8B model spends
most of its failures on malformed JSON rather than on wrong content.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3:latest"
CONTEXT_TOKENS = 8192

# Generation has to be bounded or a single filing can consume hours. Measured
# over the first 37 corpus filings: median 100s, mean 440s, worst 7,005s -- and
# the worst one produced 3,715 characters of JSON while an 18-minute one
# produced 16,763. Time does not track output; it tracks how long the model
# ruminates before answering, and that is unbounded. A token ceiling is the only
# reliable bound, since the HTTP timeout is per-read and never fires on a model
# that is still slowly emitting.
DEFAULT_NUM_PREDICT = 6000


@dataclass(frozen=True)
class Response:
    text: str
    seconds: float
    ok: bool
    error: str | None = None
    eval_count: int = 0
    truncated: bool = False


def available(timeout: float = 5.0) -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout)
        return True
    except (urllib.error.URLError, OSError):
        return False


def generate(
    system: str,
    prompt: str,
    schema: dict,
    model: str = DEFAULT_MODEL,
    timeout: float = 600.0,
    retries: int = 2,
    think: bool | None = None,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> Response:
    """Extract with a local model. `think` disables reasoning where supported.

    Qwen 3 reasons before answering by default, and on this task most of the
    270 seconds a filing costs is spent there rather than on the extraction
    itself. Whether that reasoning earns its keep is a measurable question, not
    an assumption, so it is a parameter and both settings get benchmarked.
    """
    payload = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            # Temperature 0: this is extraction, not generation. Any sampling
            # variance here shows up as graph edges that appear and disappear
            # between runs, which would make the backtest irreproducible.
            "options": {"temperature": 0, "num_ctx": CONTEXT_TOKENS,
                        "num_predict": num_predict},
    }
    # Omitted entirely when schema is None, which is how the unconstrained
    # decoding arm runs. Sending "format": null is not the same as sending
    # nothing, and the arm under test is exactly this key's presence.
    if schema is not None:
        payload["format"] = schema
    if think is not None:
        payload["think"] = think
    body = json.dumps(payload).encode()

    last_error = None
    for attempt in range(retries + 1):
        start = time.time()
        try:
            request = urllib.request.Request(
                ENDPOINT, data=body, headers={"Content-Type": "application/json"}
            )
            payload = json.loads(urllib.request.urlopen(request, timeout=timeout).read())
            used = int(payload.get("eval_count", 0))
            # Ollama files a hybrid reasoning model's whole structured answer
            # under `thinking` and returns an empty `response` unless `think` is
            # explicitly False. Reading `response` alone turned Qwen 3 30B-A3B
            # into ten blank extractions that cached as legitimate zero-link
            # filings -- a silent failure indistinguishable from a model that
            # found nothing. Recover the answer, and never return a blank while
            # the model demonstrably said something.
            answer = payload.get("response", "") or ""
            reasoning = payload.get("thinking", "") or ""
            if not answer.strip() and reasoning.strip():
                answer = reasoning
            if not answer.strip() and used > 0:
                raise ValueError(
                    f"{model} generated {used} tokens but returned no readable text"
                )
            return Response(
                text=answer,
                seconds=time.time() - start,
                ok=True,
                eval_count=used,
                # Hitting the ceiling means the JSON is cut mid-structure. The
                # claim is dropped rather than half-parsed: a truncated
                # extraction is a filing we did not read, not a filing with
                # fewer links.
                truncated=used >= num_predict,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the returned record
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))

    return Response(text="", seconds=0.0, ok=False, error=last_error)
