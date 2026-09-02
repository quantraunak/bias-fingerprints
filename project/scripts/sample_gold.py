"""Draw a stratified sample of filings for annotation, and dump what the model sees.

The gold set is defined over `passages.select` output, not over the whole 10-K.
That is the fair scope: the extractor is only ever shown the selected passages,
so a link buried in a section the selector dropped is a selector failure, not an
extraction failure, and scoring it as a false negative would confuse the two.

Sampling is stratified across issuers and filing years and seeded, so the sample
cannot drift toward filings that happen to flatter one model. Filings whose
passages contain no resolvable candidate near relationship language are kept in
proportion: a benchmark made only of rich filings would overstate precision for
every model, since the empty ones are where spurious links get invented.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import Config, PROCESSED  # noqa: E402
from src.data import universe  # noqa: E402
from src.graph import filings, passages, resolve  # noqa: E402

OUT_DIR = PROCESSED / "gold_candidates"


def main(n: int, seed: int) -> None:
    config = Config.load("configs/default.yaml")
    index = filings.load_index()
    members = set(universe.load_spells().ticker)
    index = index[index.ticker.isin(members)].copy()
    index["year"] = pd.to_datetime(index.filed).dt.year

    # One filing per issuer at most, spread over years, so the sample is not
    # dominated by whichever issuers happen to have the longest histories.
    per_issuer = index.sample(frac=1.0, random_state=seed).drop_duplicates("ticker")
    bucket = (per_issuer["year"] // 4) * 4
    sample = (
        per_issuer.groupby(bucket, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, n // bucket.nunique())), random_state=seed))
        .head(n)
        .sort_values(["ticker", "filed"])
    )

    reference = resolve.load_reference(config.data.sec_user_agent)
    lookup = resolve.build_lookup(reference)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in sample.itertuples():
        text = Path(row.path).read_text(encoding="utf-8", errors="ignore")
        selected = passages.select(text)
        if not selected:
            continue
        _, hits = passages.prescreen(selected, lookup, resolve.resolve_one, row.ticker)
        body = "\n\n---\n\n".join(selected)
        (OUT_DIR / f"{row.accession}.txt").write_text(body, encoding="utf-8")
        manifest.append({
            "accession": row.accession, "ticker": row.ticker,
            "filed": str(pd.Timestamp(row.filed).date()), "year": int(row.year),
            "chars": len(body), "n_passages": len(selected), "prescreen_hits": hits,
        })

    (OUT_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    frame = pd.DataFrame(manifest)
    print(f"{len(frame)} filings from {frame.ticker.nunique()} issuers")
    print(f"  years        {frame.year.min()}-{frame.year.max()}")
    print(f"  chars each   median {frame.chars.median():,.0f}  total {frame.chars.sum():,}")
    print(f"  prescreen    {(frame.prescreen_hits > 0).sum()} with a resolvable candidate, "
          f"{(frame.prescreen_hits == 0).sum()} without")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    main(**vars(parser.parse_args()))
