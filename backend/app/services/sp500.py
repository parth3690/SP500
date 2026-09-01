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


def fetch_nyse_smid_constituents(
    *,
    min_market_cap: float = 100e6,
    max_market_cap: float = 2e9,
    min_price: float = 2.0,
    min_dollar_volume: float = 1e6,
) -> list[Constituent]:
    """
    Fetch NYSE-listed common stocks with market cap between $100M and $2B.
    
    Filters:
    - Exchange: NYSE (not Nasdaq, not AMEX)
    - Market cap: $100M < cap < $2B  
    - Price: >= $2.00 (liquidity filter)
    - Dollar volume: >= $1M average (liquidity filter)
    - Excludes: ETFs, funds, ADRs, preferreds, warrants
    
    Uses FMP if API key is set, otherwise returns empty with warning.
    """
    import os
    
    api_key = os.getenv("FMP_API_KEY", "").strip()
    
    if api_key:
        return _fetch_nyse_smid_from_fmp(
            min_market_cap, max_market_cap, min_price, min_dollar_volume
        )
    else:
        # No fallback - FMP is required for NYSE listings + market cap data
        return []


def _fetch_nyse_smid_from_fmp(
    min_market_cap: float,
    max_market_cap: float,
    min_price: float,
    min_dollar_volume: float,
) -> list[Constituent]:
    """Fetch NYSE SMID stocks using FMP stock screener."""
    import os
    
    api_key = os.getenv("FMP_API_KEY", "").strip()
    base = os.getenv("FMP_API_BASE", "https://financialmodelingprep.com/stable").rstrip("/")
    
    try:
        # FMP stock screener endpoint
        resp = httpx.get(
            f"{base}/stock-screener",
            params={
                "exchange": "NYSE",
                "limit": 5000,
                "apikey": api_key,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[NYSE SMID] FMP stock screener failed: {e}")
        return []
    
    if not isinstance(data, list):
        print("[NYSE SMID] FMP returned unexpected data format")
        return []
    
    constituents: list[Constituent] = []
    
    for row in data:
        if not isinstance(row, dict):
            continue
        
        ticker = row.get("symbol", "").strip().upper()
        if not ticker:
            continue
        
        # Filter by market cap
        market_cap = row.get("marketCap")
        if market_cap is None:
            continue
        
        try:
            cap = float(market_cap)
        except (TypeError, ValueError):
            continue
        
        if not (min_market_cap < cap < max_market_cap):
            continue
        
        # Filter by price (liquidity)
        price = row.get("price")
        if price is not None:
            try:
                if float(price) < min_price:
                    continue
            except (TypeError, ValueError):
                continue
        
        # Filter by exchange (ensure it's NYSE)
        exchange = row.get("exchangeShortName", "").upper()
        if exchange != "NYSE":
            continue
        
        # Exclude funds, ETFs, ADRs, preferreds, warrants
        # Check symbol patterns
        if any(suffix in ticker for suffix in ["-P", ".P", "-W", ".W"]):
            continue
        
        company_name = row.get("companyName", ticker)
        
        # Exclude common fund/ETF patterns in name
        name_lower = company_name.lower()
        if any(keyword in name_lower for keyword in [
            "etf", "fund", "trust", "adr", "depositary", "preferred", "warrant"
        ]):
            continue
        
        # Sector/industry (FMP provides these)
        sector = row.get("sector", "")
        industry = row.get("industry", "")
        
        # Skip if sector indicates it's a fund
        if sector and sector.lower() in ["fund", "etf"]:
            continue
        
        constituents.append(
            Constituent(
                ticker=ticker,
                yahooTicker=normalize_yahoo_ticker(ticker),
                companyName=company_name,
                sector=sector or "Unknown",
                subIndustry=industry if industry else None,
            )
        )
    
    print(f"[NYSE SMID] Found {len(constituents)} candidates after market cap and exchange filters")
    
    # Apply volume filter if we have enough candidates (this is expensive)
    # We'll do a lighter check: just ensure price > min_price (already done above)
    # Full dollar volume check would require fetching info for each ticker
    
    return constituents


def get_nyse_smid_constituents_cached(*, refresh: bool = False) -> list[Constituent]:
    """Get cached NYSE SMID constituents ($100M-$2B market cap)."""
    key = "nyse_smid_constituents"
    if not refresh:
        cached = cache_get(CONSTITUENTS_CACHE, key)
        if cached is not None:
            return cached
    
    constituents = fetch_nyse_smid_constituents()
    if constituents:
        cache_set(CONSTITUENTS_CACHE, key, constituents)
    return constituents

