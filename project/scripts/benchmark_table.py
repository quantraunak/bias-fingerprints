"""Score every cached extractor against benchmark v2 and decompose its errors.

    python scripts/benchmark_table.py [--markdown docs/EXTRACTION.md]

Reads only from `data/processed/extract_cache`, so it runs offline, touches no
GPU, and can be re-run while an extraction is in flight. A configuration with no
cached responses for the annotated filings is skipped rather than generated --
this reports what has been measured, it does not measure.

Aggregate precision and recall answer "how good is this model". They do not
answer "good at what", and on this task the second question is the one that
decides whether an extractor is usable. Three decompositions are reported
because each corresponds to a failure that costs something different downstream:

* **Hard negatives.** Five of the ten filings carry an empty gold annotation
  while naming plenty of companies -- First Solar's eleven names sit inside
  executive biographies, Qualcomm's six are acquisitions and regulators. False
  positives here are the ones aggregate precision hides, because a model can
  buy precision on the link-rich filings and still be unusable on the half of
  the corpus that discloses nothing.
* **Direction errors.** The same counterparty extracted with the relation
  reversed. Worse than a miss: a missed edge costs coverage, an inverted one
  puts a real return series on a backwards relationship and propagates with the
  wrong sign. Nothing downstream flags it.
* **Enumeration recall.** Recall on the filings that name ten or more
  counterparties, against those that name fewer. Filings that list an entire
  outsourced-assembly chain are where the graph's density comes from, and a
  model that extracts three names from a list of twelve fails precisely there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config, PROCESSED  # noqa: E402
from src.graph import evaluate, extract, filings, passages, resolve  # noqa: E402

CACHE = PROCESSED / "extract_cache"
LONG_LIST = 10  # gold links at or above this count is an "enumeration" filing


def configurations() -> list[tuple[str, str, Path]]:
    """(model, reasoning, directory) for every populated cache directory."""
    out = []
    for path in sorted(CACHE.iterdir()):
        if not path.is_dir() or not any(path.glob("*.json")):
            continue
        name = path.name
        if name.endswith("_nothink"):
            model, reasoning = name[: -len("_nothink")], "off"
        elif name.endswith("_think"):
            model, reasoning = name[: -len("_think")], "on"
        else:
            model, reasoning = name, "default"
        out.append((model.replace("_", ":", 1), reasoning, path))
    return out


def predictions(directory: Path, gold: dict, index: pd.DataFrame,
                selection: dict[str, list[str]]) -> tuple[pd.DataFrame, float, int]:
    """Validated claims for the annotated filings held in one cache directory."""
    rows, seconds, n = [], 0.0, 0
    for accession in gold:
        path = directory / f"{accession}.json"
        if not path.exists() or accession not in index.index:
            continue
        payload = json.loads(path.read_text())
        seconds += payload.get("seconds", 0.0)
        n += 1
        kept, _ = extract.validate(
            extract.parse_response(payload["text"]), selection[accession]
        )
        for link in kept:
            rows.append({"accession": accession, "counterparty": link["counterparty"],
                         "relation": link["relation"]})
    return pd.DataFrame(rows, columns=["accession", "counterparty", "relation"]), seconds, n


def decompose(frame: pd.DataFrame, gold: dict) -> dict:
    """Hard negatives, direction errors, and recall split by list length."""
    empty = {a for a, links in gold.items() if not links}
    rich = {a for a, links in gold.items() if len(links) >= LONG_LIST}

    hard_fp = int(frame[frame.accession.isin(empty)].false_positive.sum())

    direction = 0
    for row in frame.itertuples():
        missed = {m.rsplit(":", 1)[0]: m.rsplit(":", 1)[1] for m in row.missed}
        for item in row.spurious:
            name, relation = item.rsplit(":", 1)
            if name in missed and missed[name] != relation:
                direction += 1

    def recall(subset: set) -> float | None:
        part = frame[frame.accession.isin(subset)]
        tp, fn = part.true_positive.sum(), part.false_negative.sum()
        return round(float(tp / (tp + fn)), 3) if tp + fn else None

    return {
        "hard_negative_fp": hard_fp,
        "direction_errors": direction,
        "recall_enumerations": recall(rich),
        "recall_short": recall(set(gold) - rich - empty),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--markdown", help="also write a markdown table here")
    args = parser.parse_args()

    config = Config.load(args.config)
    gold = evaluate.load_gold()
    index = filings.load_index().set_index("accession")
    lookup = resolve.build_lookup(resolve.load_reference(config.data.sec_user_agent))

    # Passage selection is deterministic and identical across models, so it is
    # computed once and reused rather than per configuration.
    selection = {
        a: passages.select(Path(index.loc[a].path).read_text(encoding="utf-8", errors="ignore"))
        for a in gold if a in index.index
    }

    rows = []
    for model, reasoning, directory in configurations():
        frame, seconds, n = predictions(directory, gold, index, selection)
        if n < len(gold):
            print(f"skipping {directory.name}: {n}/{len(gold)} annotated filings cached")
            continue
        scored, agg = evaluate.score(frame, gold, lookup)
        rows.append({"model": model, "reasoning": reasoning, **agg,
                     "seconds": round(seconds / n), **decompose(scored, gold)})

    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    empty = sum(1 for links in gold.values() if not links)
    rich = sum(1 for links in gold.values() if len(links) >= LONG_LIST)
    print(f"\nbenchmark v2: {len(gold)} filings, {sum(len(v) for v in gold.values())} links, "
          f"{empty} annotated empty, {rich} naming {LONG_LIST}+\n")
    print(table[["model", "reasoning", "precision", "recall", "f1", "seconds",
                 "hard_negative_fp", "direction_errors",
                 "recall_enumerations", "recall_short"]].to_string(index=False))

    if args.markdown:
        Path(args.markdown).write_text(render(table, gold, empty, rich))
        print(f"\nwrote {args.markdown}")


def render(table: pd.DataFrame, gold: dict, empty: int, rich: int) -> str:
    lines = [
        "# Extraction frontier",
        "",
        "Generated by `scripts/benchmark_table.py` from cached responses. Every",
        "configuration is scored against the same annotated sample, so the numbers",
        "are comparable.",
        "",
        f"**Benchmark v2**: {len(gold)} filings, {sum(len(v) for v in gold.values())} links, "
        f"{empty} annotated empty on purpose, {rich} naming {LONG_LIST} or more counterparties.",
        "",
        "| model | reasoning | P | R | F1 | s/filing | FP on empty filings | direction errors | recall on 10+ | recall on 1-9 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        lines.append(
            f"| `{r.model}` | {r.reasoning} | {r.precision} | {r.recall} | **{r.f1}** | "
            f"{r.seconds} | {r.hard_negative_fp} | {r.direction_errors} | "
            f"{r.recall_enumerations} | {r.recall_short} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
