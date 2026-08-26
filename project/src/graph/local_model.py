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


@dataclass(frozen=True)
class Response:
    text: str
    seconds: float
    ok: bool
    error: str | None = None


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
) -> Response:
    body = json.dumps(
        {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            # Temperature 0: this is extraction, not generation. Any sampling
            # variance here shows up as graph edges that appear and disappear
            # between runs, which would make the backtest irreproducible.
            "options": {"temperature": 0, "num_ctx": CONTEXT_TOKENS},
        }
    ).encode()

    last_error = None
    for attempt in range(retries + 1):
        start = time.time()
        try:
            request = urllib.request.Request(
                ENDPOINT, data=body, headers={"Content-Type": "application/json"}
            )
            payload = json.loads(urllib.request.urlopen(request, timeout=timeout).read())
            return Response(text=payload.get("response", ""), seconds=time.time() - start, ok=True)
        except Exception as exc:  # noqa: BLE001 - surfaced in the returned record
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))

    return Response(text="", seconds=0.0, ok=False, error=last_error)
