from __future__ import annotations

import re
from io import StringIO
from typing import Iterable

import httpx
import pandas as pd

from ..models import Constituent
from .cache import CONSTITUENTS_CACHE, cache_get, cache_set

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_USER_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,19}$")


def normalize_yahoo_ticker(ticker: str) -> str:
    # Wikipedia uses dots for some share classes (e.g., BRK.B). Yahoo uses dashes.
    return ticker.strip().upper().replace(".", "-")


def normalize_user_ticker(ticker: str) -> str:
    """Return a clean display ticker, or an empty string for invalid input."""
    display = str(ticker or "").strip().upper()
    return display if _USER_TICKER_RE.fullmatch(display) else ""


def _find_constituents_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        cols = {str(c).strip() for c in table.columns}
        if {"Symbol", "Security"}.issubset(cols) and any("GICS" in c for c in cols):
            return table
    raise ValueError("Unable to locate S&P 500 constituents table on Wikipedia page.")


def fetch_sp500_constituents() -> list[Constituent]:
    resp = httpx.get(
        WIKIPEDIA_SP500_URL,
        timeout=httpx.Timeout(15.0),
        headers={"User-Agent": "sp500-movers-analyzer/1.0"},
        follow_redirects=True,
    )
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    df = _find_constituents_table(tables)

    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "companyName",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "subIndustry",
        }
    )
    required = ["ticker", "companyName", "sector"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' from constituents table.")

    constituents: list[Constituent] = []
    for row in df.to_dict(orient="records"):
        ticker = str(row["ticker"]).strip().upper()
        if not ticker:
            continue
        sub = row.get("subIndustry")
        constituents.append(
            Constituent(
                ticker=ticker,
                yahooTicker=normalize_yahoo_ticker(ticker),
                companyName=str(row.get("companyName", "")).strip(),
                sector=str(row.get("sector", "")).strip(),
                subIndustry=None if sub is None or pd.isna(sub) else str(sub).strip(),
            )
        )

    # De-duplicate by Wikipedia ticker while keeping stable order.
    seen: set[str] = set()
    out: list[Constituent] = []
    for c in constituents:
        if c.ticker in seen:
            continue
        seen.add(c.ticker)
        out.append(c)

    return out


def get_sp500_constituents_cached(*, refresh: bool = False) -> list[Constituent]:
    key = "sp500_constituents"
    if not refresh:
        cached = cache_get(CONSTITUENTS_CACHE, key)
        if cached is not None:
            return cached

    constituents = fetch_sp500_constituents()
    cache_set(CONSTITUENTS_CACHE, key, constituents)
    return constituents


def get_yahoo_tickers(constituents: Iterable[Constituent]) -> list[str]:
    return [c.yahooTicker for c in constituents]
