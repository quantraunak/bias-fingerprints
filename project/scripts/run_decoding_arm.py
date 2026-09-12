"""Constrained vs unconstrained decoding, at matched k. The factor RELATED_WORK.md
flagged as the strongest unfalsified competing explanation.

    python3 scripts/run_decoding_arm.py --arm unconstrained --levels 1,4,16

Measured 2026-09-11 on one k=4 document, qwen3:30b-a3b, temperature 0:

    with schema     268 tokens     965 chars
    without schema 4914 tokens  19481 chars

An 18x difference in generation budget. If recall degrades with k only under the
grammar, the effect belongs to constrained decoding rather than to the model.

The `think` flag is not a factor: at fixed format it changes only which field
the same tokens are filed under. Verified, same eval_count either way.
"""
from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED
from src.graph import extract, local_model

DOCS = PROCESSED / "cardinality_docs.json"
JSON_INSTRUCTION = (
    '\n\nReturn ONLY a JSON object of the form '
    '{"links": [{"counterparty": "...", "relation": "customer|supplier|partner|competitor", '
    '"evidence": "..."}]} and nothing else.'
)
OBJECT = re.compile(r"\{.*\}", re.S)


def parse_loose(text: str) -> list[dict]:
    """Post-hoc parse of free-form output. Deliberately tolerant: a parse failure
    here would score as a recall failure and confound the arm under test.

    Unconstrained, the model emits a full <think> block and then the JSON. A
    greedy brace match spans from a brace inside the reasoning to the final one
    and yields nothing parseable, so the reasoning is dropped first and the
    object is then scanned balanced from the end.
    """
    if not text:
        return []
    body = text.rsplit("</think>", 1)[-1]
    body = re.sub(r"^```(?:json)?|```$", "", body.strip(), flags=re.M).strip()

    def links_from(candidate: str) -> list[dict] | None:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        found = payload.get("links", [])
        return [l for l in found if isinstance(l, dict)] if isinstance(found, list) else None

    direct = links_from(body)
    if direct is not None:
        return direct

    # Balanced scan backwards from the last closing brace.
    for end in range(len(body) - 1, -1, -1):
        if body[end] != "}":
            continue
        depth = 0
        for start in range(end, -1, -1):
            if body[start] == "}":
                depth += 1
            elif body[start] == "{":
                depth -= 1
                if depth == 0:
                    got = links_from(body[start:end + 1])
                    if got is not None:
                        return got
                    break
        break
    return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["constrained", "unconstrained"])
    p.add_argument("--model", default="qwen3:30b-a3b")
    p.add_argument("--levels", default="1,4,16")
    p.add_argument("--num-predict", type=int, default=12000)
    p.add_argument("--max-per-level", type=int, default=None,
                   help="cap replicates per k. Measurement noise falls as k grows -- "
                        "recall at k=16 averages over 16 items, at k=1 it is a coin "
                        "flip -- so the per-document recall SD in the constrained arm "
                        "runs 0.481 at k=1 down to 0.152 at k=16. Detecting the "
                        "pre-registered 0.15 at 80%% power needs 106 documents at k=1 "
                        "and 17 at k=16. Equal n per level is the wrong allocation.")
    args = p.parse_args()

    tag = f"{args.model.replace(':', '_')}_{args.arm}"
    out = PROCESSED / "decoding_runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    want = {int(x) for x in args.levels.split(",")}
    docs = [d for d in json.loads(DOCS.read_text()) if d["k_intended"] in want]
    if args.max_per_level:
        capped = []
        for level in sorted({d["k_intended"] for d in docs}):
            capped += [d for d in docs if d["k_intended"] == level][:args.max_per_level]
        docs = capped
    todo = [d for d in docs if not (out / f"{d['doc_id']}.json").exists()]
    print(f"{args.arm}  {args.model}  {len(todo)}/{len(docs)} to run", flush=True)

    began, done = time.time(), 0
    for doc in todo:
        prompt = extract.user_prompt("ACME", "ACME", "2020-01-01", doc["paragraphs"])
        if args.arm == "unconstrained":
            prompt += JSON_INSTRUCTION
        r = local_model.generate(
            extract.SYSTEM, prompt,
            extract.Extraction.model_json_schema() if args.arm == "constrained" else None,
            model=args.model, timeout=1800, think=False, num_predict=args.num_predict)
        if not r.ok:
            print(f"  {doc['doc_id']}: {r.error}", flush=True); continue
        links = extract.parse_response(r.text) if args.arm == "constrained" else parse_loose(r.text)
        (out / f"{doc['doc_id']}.json").write_text(json.dumps(
            {"text": r.text, "links": links, "seconds": r.seconds,
             "eval_count": r.eval_count, "truncated": bool(r.truncated)}))
        done += 1
        left = (len(todo) - done) * (time.time() - began) / done / 3600
        print(f"  [{done:>3}/{len(todo)}] {doc['doc_id']} k={doc['k_actual']:>2} "
              f"{r.seconds:>5.0f}s {r.eval_count:>6}tok {len(links):>2} links  ~{left:.1f}h",
              flush=True)
    sys.exit(0 if done >= len(todo) else 1)


if __name__ == "__main__":
    main()
