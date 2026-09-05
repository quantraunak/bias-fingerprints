"""Map extracted counterparty names onto tickers with a price series.

A name in a filing is written however the filer felt like writing it: "Taiwan
Semiconductor Manufacturing Company Limited", "TSMC", "Taiwan Semiconductor".
Only names that resolve to a listed security can become a testable edge.

The bias here is deliberately toward **precision over recall**. An unresolved
counterparty costs one edge. A wrongly resolved one injects a return series that
has nothing to do with the relationship, and because the resulting signal is
still smooth and plausible it will not announce itself in any diagnostic. So
matching is token-based and tiered, single-token names are only accepted when
the token is distinctive, and the unmatched rate is reported rather than tuned
away.

Edit distance is deliberately not used. "Delta Air Lines" and "Delta Apparel"
are close in edit distance and unrelated as businesses; token overlap separates
them, and cheap string similarity is exactly how false edges get in.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW

REFERENCE_PATH = RAW / "fundamentals" / "_company_names.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "limited", "llc", "lp", "llp", "plc", "nv", "bv", "ag", "sa", "se",
    "gmbh", "ab", "as", "oyj", "spa", "pte", "pty", "kk", "holdings", "holding",
    "group", "the", "and", "of", "intl", "international", "worldwide", "global",
    "usa", "us", "america", "american", "na", "trust", "partners", "lc", "cos",
    "com",  # "AMAZON COM INC" -> "amazon com" never matched a filing's "Amazon"
}

# Single tokens too ambiguous to resolve on their own. Each is a real company
# name and also a common word or a shared prefix across many issuers.
AMBIGUOUS_SINGLE = {
    "delta", "apple", "target", "gap", "shell", "total", "orange", "sprint",
    "square", "block", "match", "unity", "arm", "asml", "sap", "abb", "bp",
    "first", "national", "general", "united", "standard", "premier", "pacific",
    "atlantic", "central", "western", "eastern", "northern", "southern",
    "advance", "advanced", "allied", "capital", "commercial", "consumer",
    "energy", "financial", "industrial", "insurance", "media", "micro",
    "nova", "omega", "sigma", "summit", "sun", "union", "vantage", "vision",
}

# EDGAR appends the state of incorporation to many registered names --
# "NORTHROP GRUMMAN CORP /DE/". Stripped before punctuation, because otherwise
# the slashes go first and "DE" survives as a token: "northrop grumman de" never
# matches "northrop grumman", and the failure is silent and systematic.
STATE_MARKER = re.compile(r"/[A-Z]{2}/?\s*$|\s/[A-Z]{2}/\s*")

PUNCTUATION = re.compile(r"[^\w\s]")
WHITESPACE = re.compile(r"\s+")

# Ticker suffixes marking something other than common stock.
NON_COMMON = r"-(?:P[A-Z]?|W[SI]?|U|R[T]?|CL)$"


def load_reference(user_agent: str, force: bool = False) -> pd.DataFrame:
    """Every ticker EDGAR knows about, with its registered company name."""
    if REFERENCE_PATH.exists() and not force:
        payload = json.loads(REFERENCE_PATH.read_text())
    else:
        time.sleep(0.12)
        response = requests.get(TICKERS_URL, headers={"User-Agent": user_agent}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.write_text(json.dumps(payload))

    rows = [
        {"ticker": entry["ticker"].upper(), "cik": int(entry["cik_str"]), "title": entry["title"]}
        for entry in payload.values()
    ]
    frame = pd.DataFrame(rows).drop_duplicates("ticker")
    frame["normalized"] = frame["title"].map(normalize)
    frame["tokens"] = frame["normalized"].map(lambda s: frozenset(s.split()))
    frame = frame[frame["normalized"].str.len() > 0]

    # Preferred shares, warrants, units and rights carry the same company name as
    # the common line, so they defeat the uniqueness test that every match tier
    # depends on -- "Boeing" resolved to nothing because BA and BA-PA both
    # normalise to "boeing". They are also the wrong return series for a graph
    # node regardless, so they are dropped rather than tie-broken.
    frame = frame[~frame["ticker"].str.contains(NON_COMMON, regex=True, na=False)]

    # Whatever duplicates survive are genuine multi-class listings (GOOG/GOOGL,
    # BRK-A/BRK-B). Keep the shortest ticker, which is conventionally the more
    # liquid line, so the name still resolves instead of being thrown away.
    frame = frame.assign(_len=frame["ticker"].str.len()).sort_values(["normalized", "_len"])
    frame = frame.drop_duplicates("normalized", keep="first").drop(columns="_len")
    return frame.reset_index(drop=True)


# Dotted acronyms: "N.V." and "S.p.A." are corporate suffixes, but stripping
# punctuation first shatters them into single letters that survive the suffix
# filter. STMicroelectronics N.V. normalised to "stmicroelectronics n v" and
# therefore never matched a filing's "STMicroelectronics, Inc." -- a listed
# counterparty lost to a tokenisation order.
DOTTED_ACRONYM = re.compile(r"\b(?:[A-Za-z]\.){2,}")

# Filings introduce a short form the first time they name a counterparty --
# `European Aeronautic Defence and Space Company (EADS)`, `Tianjin Haiguang
# Advanced Technology Investment Co., Ltd. ("THATIC")`. An extractor quoting the
# name as written carries the parenthetical; a second mention of the same
# company does not. Left in, the two normalise differently and one company
# becomes two nodes -- which also scores a correct extraction as simultaneously
# a false positive and a false negative against an annotation that omitted it.
SHORT_FORM = re.compile(r"[(\[][^)\]]{1,40}[)\]]|[\u201c\"][^\u201d\"]{1,40}[\u201d\"]")


def normalize(name: str) -> str:
    """Lowercase, strip punctuation, drop corporate suffixes and filler."""
    text = SHORT_FORM.sub(" ", str(name))
    text = STATE_MARKER.sub(" ", text)
    text = DOTTED_ACRONYM.sub(lambda m: m.group().replace(".", "") + " ", text)
    text = PUNCTUATION.sub(" ", text.lower())
    text = WHITESPACE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t not in SUFFIXES and not t.isdigit()]
    return " ".join(tokens)


# A token appearing in more than this many registered names cannot identify a
# company on its own. Measured against the registry rather than listed by hand,
# because no stoplist stays correct as the registry changes.
#
# Calibrated on the cases that motivated it. After suffix stripping most
# registered names are two or three tokens, so document frequencies are small
# and the boundary sits low: "deere" and "boeing" appear once, "taiwan" twice,
# "city" six times and "semiconductor" twelve. A ceiling of three admits the
# first group and refuses the second, which is exactly the split between a
# token that names one company and a token that names an industry.
GENERIC_MAX_DF = 3


def build_lookup(reference: pd.DataFrame) -> dict:
    """Indexes for the match tiers, built once and reused per name."""
    exact: dict[str, list[str]] = {}
    by_tokens: dict[frozenset, list[str]] = {}
    by_first: dict[str, list[int]] = {}
    token_df: dict[str, int] = {}

    for row in reference.itertuples():
        exact.setdefault(row.normalized, []).append(row.ticker)
        by_tokens.setdefault(row.tokens, []).append(row.ticker)
        if row.tokens:
            first = sorted(row.tokens)[0]
            by_first.setdefault(first, []).append(row.Index)
        for token in row.tokens:
            token_df[token] = token_df.get(token, 0) + 1

    return {
        "exact": exact,
        "by_tokens": by_tokens,
        "by_first": by_first,
        "reference": reference,
        "token_df": token_df,
        "generic_max_df": GENERIC_MAX_DF,
    }


def distinctive(tokens, lookup: dict) -> set:
    """Tokens rare enough in the registry to identify a company."""
    df, ceiling = lookup["token_df"], lookup["generic_max_df"]
    return {t for t in tokens if df.get(t, 0) <= ceiling}


# Counterparties whose filing name never matches their registered name. Kept
# small and explicit: each entry is a judgement that belongs in review, not a
# fuzzy threshold that silently admits its neighbours.
ALIASES = {
    "tsmc": "TSM",
    "taiwan semiconductor manufacturing": "TSM",
    "hon hai precision industry": None,      # Taipei-listed, no US line
    "foxconn": None,
    "samsung electronics": None,             # Korea-listed
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "facebook": "META",
    "meta platforms": "META",
    "international business machines": "IBM",
    "hewlett packard enterprise": "HPE",
    "hewlett packard": "HPQ",
    "wal mart stores": "WMT",
    "walmart": "WMT",
    "united parcel service": "UPS",
    "federal express": "FDX",
    "general motors": "GM",
    "ford motor": "F",
    "at t": "T",
    "verizon communications": "VZ",
    "nvidia": "NVDA",
    "advanced micro devices": "AMD",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "cisco systems": "CSCO",
    "dell technologies": "DELL",
    "lenovo": None,
    "huawei": None,
}


def resolve_one(name: str, lookup: dict) -> tuple[str | None, str]:
    """Return (ticker, tier). Tier is 'alias' | 'exact' | 'tokens' | 'subset' | 'none'."""
    normalized = normalize(name)
    if not normalized:
        return None, "none"

    if normalized in ALIASES:
        return ALIASES[normalized], "alias"

    tokens = frozenset(normalized.split())
    if not tokens:
        return None, "none"

    # A single common token is not enough to identify a company.
    if len(tokens) == 1 and next(iter(tokens)) in AMBIGUOUS_SINGLE:
        return None, "none"

    candidates = lookup["exact"].get(normalized)
    if candidates and len(candidates) == 1:
        return candidates[0], "exact"

    candidates = lookup["by_tokens"].get(tokens)
    if candidates and len(candidates) == 1:
        return candidates[0], "tokens"

    # Subset match: the filing name may be shorter than the registered name, or
    # longer. What makes one safe is not how many tokens it has but whether the
    # tokens the two sides share can identify a company at all.
    #
    # A token-count rule was the first attempt and failed in both directions.
    # Requiring two tokens on the query alone let "City Council" reach CHCO,
    # because "City Holding Co" reduces to the single generic token "city".
    # Requiring two on the reference side then blocked "John Deere" -> DE, since
    # "Deere & Co" also reduces to one token -- but "deere" names exactly one
    # registrant and "city" names many. Counting tokens cannot tell those apart.
    #
    # Sharing a *distinctive* token can. Semiconductor Manufacturing
    # International and Taiwan Semiconductor Manufacturing overlap on
    # {semiconductor, manufacturing}, both generic in this registry, so the
    # match is refused -- it previously resolved SMIC to TSM and put a Taiwanese
    # foundry's return series on a Chinese foundry's edge.
    if tokens:
        reference = lookup["reference"]
        hits = []
        for first in sorted(tokens):
            for index in lookup["by_first"].get(first, []):
                row = reference.iloc[index]
                other = row["tokens"]
                if not (tokens <= other or other <= tokens):
                    continue
                if not distinctive(tokens & other, lookup):
                    continue
                hits.append(row["ticker"])
        if len(set(hits)) == 1:
            return hits[0], "subset"

    return None, "none"


def resolve_frame(claims: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Attach `counterparty_ticker` and `match_tier` to a claims table."""
    lookup = build_lookup(reference)
    unique = claims["counterparty"].drop_duplicates()
    mapping = {name: resolve_one(name, lookup) for name in unique}

    out = claims.copy()
    out["counterparty_ticker"] = out["counterparty"].map(lambda n: mapping[n][0])
    out["match_tier"] = out["counterparty"].map(lambda n: mapping[n][1])
    return out


def coverage(resolved: pd.DataFrame) -> dict:
    """Match rates, reported rather than tuned away."""
    total = len(resolved)
    matched = resolved["counterparty_ticker"].notna().sum()
    names = resolved["counterparty"].nunique()
    names_matched = resolved.loc[resolved["counterparty_ticker"].notna(), "counterparty"].nunique()
    return {
        "claims": total,
        "claims_resolved": int(matched),
        "claim_match_rate": round(float(matched) / total, 3) if total else 0.0,
        "distinct_names": int(names),
        "distinct_resolved": int(names_matched),
        "name_match_rate": round(float(names_matched) / names, 3) if names else 0.0,
        "by_tier": resolved["match_tier"].value_counts().to_dict(),
    }
