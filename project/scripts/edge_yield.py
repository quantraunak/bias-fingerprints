"""How many usable counterparties does a filing actually yield?

This is the measurement `docs/PILOT_FINDINGS.md` used to decide the universe:
it counts, per filing, the distinct companies named near relationship language
that resolve to a listed ticker. A filing yielding nothing cannot contribute an
edge no matter how good the extractor is, so this bounds the whole study.

It is re-run here because that decision was made through a resolver that was
losing roughly a third of the names it should have matched. `--compare-ref`
loads the resolver from any git revision and measures both, so the effect of a
resolution change on edge yield is observable rather than argued.

    python scripts/edge_yield.py --sample 300 --compare-ref HEAD~1
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.data import universe  # noqa: E402
from src.graph import filings, passages  # noqa: E402
from src.graph import resolve as current_resolve  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = "project/src/graph/resolve.py"


def resolver_at(ref: str) -> types.ModuleType:
    """Load `resolve.py` as it stood at a git revision, as a live module."""
    source = subprocess.run(
        ["git", "show", f"{ref}:{MODULE_PATH}"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    spec = importlib.util.spec_from_loader(f"resolve_{ref}", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(REPO / MODULE_PATH)
    exec(compile(source, f"<{ref}:resolve.py>", "exec"), module.__dict__)
    return module


def counterparties(text: str, source_ticker: str, module, lookup) -> set[str]:
    """Distinct resolvable companies named near relationship language.

    The same condition the extractor's prescreen applies: a capitalised run that
    resolves to a ticker AND sits within `PROXIMITY_CHARS` of a customer or
    supplier phrase. A name resolving somewhere else in the document is not
    evidence of a relationship.
    """
    found: set[str] = set()
    for paragraph in passages.select(text):
        anchors = [m.start() for m in passages.WEAK.finditer(paragraph)]
        anchors += [m.start() for m in passages.STRONG.finditer(paragraph)]
        if not anchors:
            continue
        for position, name in passages.company_candidates(paragraph):
            if not any(abs(position - a) <= passages.PROXIMITY_CHARS for a in anchors):
                continue
            ticker, _ = module.resolve_one(name, lookup)
            if ticker and ticker != source_ticker:
                found.add(ticker)
    return found


def measure(sample: pd.DataFrame, module, label: str) -> pd.Series:
    reference = module.load_reference(CONFIG.data.sec_user_agent)
    lookup = module.build_lookup(reference)
    counts = []
    for n, row in enumerate(sample.itertuples(), 1):
        try:
            text = Path(row.path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        counts.append(len(counterparties(text, row.ticker, module, lookup)))
        if n % 50 == 0:
            print(f"    {label}: {n}/{len(sample)}", flush=True)
    return pd.Series(counts, name=label)


def report(counts: pd.Series) -> dict:
    return {
        "filings": len(counts),
        "share_with_any": float((counts > 0).mean()),
        "mean_per_filing": float(counts.mean()),
        "median_per_filing": float(counts.median()),
        "zero": int((counts == 0).sum()),
        "one": int((counts == 1).sum()),
        "two": int((counts == 2).sum()),
        "three_plus": int((counts >= 3).sum()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300, help="filings per universe")
    parser.add_argument("--compare-ref", default=None, help="git ref to A/B the resolver against")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    CONFIG = Config.load("configs/default.yaml")
    index = filings.load_index()
    members = set(universe.load_spells().ticker)

    groups = {
        "S&P 500": index[index.ticker.isin(members)],
        "concentrated suppliers": index[~index.ticker.isin(members)],
    }

    modules = {"fixed resolver": current_resolve}
    if args.compare_ref:
        modules[f"resolver at {args.compare_ref}"] = resolver_at(args.compare_ref)

    rows = []
    for universe_name, frame in groups.items():
        take = min(args.sample, len(frame))
        sample = frame.sample(take, random_state=args.seed)
        print(f"\n{universe_name}: {take} filings from {sample.ticker.nunique()} issuers")
        for label, module in modules.items():
            counts = measure(sample, module, label)
            rows.append({"universe": universe_name, "resolver": label, **report(counts)})

    out = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(f"{'universe':<24}{'resolver':<26}{'≥1 cp':>8}{'mean':>8}{'median':>8}"
          f"{'0':>6}{'1':>5}{'2':>5}{'3+':>5}")
    for r in out.itertuples():
        print(f"{r.universe:<24}{r.resolver:<26}{r.share_with_any:>7.1%}{r.mean_per_filing:>8.2f}"
              f"{r.median_per_filing:>8.0f}{r.zero:>6}{r.one:>5}{r.two:>5}{r.three_plus:>5}")
    out.to_csv("reports/edge_yield.csv", index=False)
    print("\nwrote reports/edge_yield.csv")
