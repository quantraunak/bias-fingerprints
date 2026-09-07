"""How much of the corpus actually needs the model?

Extraction is the expensive stage -- 270s per filing on the model that scores
well -- so the size of the queue decides whether the run is days or weeks. A
filing whose passages contain no resolvable name near relationship language
cannot yield an edge no matter what reads it, and paying a 32B model to
establish that is the most expensive way to learn nothing.

Writes a queue ordered by prescreen yield, so an interrupted extraction run has
covered the filings most likely to carry edges rather than an alphabetical
prefix of the corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config, PROCESSED  # noqa: E402
from src.data import universe  # noqa: E402
from src.graph import filings, passages, resolve  # noqa: E402

SECONDS_PER_FILING = 270  # measured for qwen3:32b on this benchmark
OUT = PROCESSED / "extract_queue.parquet"


def main() -> None:
    config = Config.load("configs/default.yaml")
    lookup = resolve.build_lookup(resolve.load_reference(config.data.sec_user_agent))
    index = filings.load_index()
    members = set(universe.load_spells().ticker)

    rows = []
    for n, row in enumerate(index.itertuples(), 1):
        try:
            text = Path(row.path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        selected = passages.select(text)
        if selected:
            passes, hits = passages.prescreen(selected, lookup, resolve.resolve_one, row.ticker)
            chars = sum(len(p) for p in selected)
        else:
            passes, hits, chars = False, 0, 0
        rows.append({"accession": row.accession, "ticker": row.ticker, "filed": row.filed,
                     "passes": passes, "hits": hits, "chars": chars,
                     "sp500": row.ticker in members})
        if n % 500 == 0:
            print(f"  screened {n}/{len(index)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.sort_values(["passes", "hits"], ascending=False).to_parquet(OUT)

    print(f"\ntotal filings          {len(frame):,}")
    print(f"  pass prescreen       {frame.passes.sum():,} ({frame.passes.mean():.0%})")
    print(f"  >=1 resolvable name  {(frame.hits > 0).sum():,}")
    print(f"\nS&P 500 issuers        {frame.sp500.sum():,} filings, "
          f"{frame[frame.sp500].passes.sum():,} pass")
    for label, sub in [("whole corpus", frame[frame.passes]),
                       ("S&P 500 only", frame[frame.passes & frame.sp500])]:
        hours = len(sub) * SECONDS_PER_FILING / 3600
        print(f"  {label:14} {len(sub):>5,} to extract -> {hours:>5.0f}h ({hours/24:.1f} days)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
