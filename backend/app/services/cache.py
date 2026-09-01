from __future__ import annotations

import os
from threading import RLock
from typing import Any, Hashable, Optional

from cachetools import TTLCache

CONSTITUENTS_TTL_SECONDS = int(os.getenv("CONSTITUENTS_TTL_SECONDS", "86400"))
MOVERS_TTL_SECONDS = int(os.getenv("MOVERS_TTL_SECONDS", "900"))
CROSSOVERS_TTL_SECONDS = int(os.getenv("CROSSOVERS_TTL_SECONDS", "900"))

CONSTITUENTS_CACHE: TTLCache = TTLCache(maxsize=2, ttl=CONSTITUENTS_TTL_SECONDS)
MOVERS_CACHE: TTLCache = TTLCache(maxsize=16, ttl=MOVERS_TTL_SECONDS)
CROSSOVERS_CACHE: TTLCache = TTLCache(maxsize=4, ttl=CROSSOVERS_TTL_SECONDS)

RESEARCH_TTL_SECONDS = int(os.getenv("RESEARCH_TTL_SECONDS", "900"))
RESEARCH_CACHE: TTLCache = TTLCache(maxsize=32, ttl=RESEARCH_TTL_SECONDS)

RSI_SCAN_TTL_SECONDS = int(os.getenv("RSI_SCAN_TTL_SECONDS", "900"))
RSI_SCAN_CACHE: TTLCache = TTLCache(maxsize=8, ttl=RSI_SCAN_TTL_SECONDS)
WEEKLY_MA_SCAN_CACHE: TTLCache = TTLCache(maxsize=4, ttl=RSI_SCAN_TTL_SECONDS)

# Fast-mover scan cache (FMP-based)
MOVE_FINDER_TTL_SECONDS = int(os.getenv("MOVE_FINDER_TTL_SECONDS", "120"))
MOVE_FINDER_CACHE: TTLCache = TTLCache(maxsize=8, ttl=MOVE_FINDER_TTL_SECONDS)

MULTIBAGGER_TTL_SECONDS = int(os.getenv("MULTIBAGGER_TTL_SECONDS", "86400"))
MULTIBAGGER_CACHE: TTLCache = TTLCache(maxsize=64, ttl=MULTIBAGGER_TTL_SECONDS)

MARKET_CONDITIONS_TTL_SECONDS = int(os.getenv("MARKET_CONDITIONS_TTL_SECONDS", "900"))
MARKET_CONDITIONS_CACHE: TTLCache = TTLCache(maxsize=4, ttl=MARKET_CONDITIONS_TTL_SECONDS)

ALPHA_TTL_SECONDS = int(os.getenv("ALPHA_TTL_SECONDS", "900"))
ALPHA_CACHE: TTLCache = TTLCache(maxsize=8, ttl=ALPHA_TTL_SECONDS)

INSTITUTIONAL_SCANNER_TTL_SECONDS = int(os.getenv("INSTITUTIONAL_SCANNER_TTL_SECONDS", "900"))
INSTITUTIONAL_SCANNER_CACHE: TTLCache = TTLCache(maxsize=8, ttl=INSTITUTIONAL_SCANNER_TTL_SECONDS)

VALUATION_TTL_SECONDS = int(os.getenv("VALUATION_TTL_SECONDS", "86400"))
VALUATION_CACHE: TTLCache = TTLCache(maxsize=1024, ttl=VALUATION_TTL_SECONDS)

# Shared S&P 500 price data cache — avoids redundant Yahoo downloads
# across movers, crossovers, and RSI endpoints
PRICE_DATA_TTL_SECONDS = int(os.getenv("PRICE_DATA_TTL_SECONDS", "900"))
PRICE_DATA_CACHE: TTLCache = TTLCache(maxsize=4, ttl=PRICE_DATA_TTL_SECONDS)

_LOCK = RLock()


def cache_get(cache: TTLCache, key: Hashable) -> Optional[Any]:
    with _LOCK:
        return cache.get(key)


def cache_set(cache: TTLCache, key: Hashable, value: Any) -> None:
    with _LOCK:
        cache[key] = value


def price_cache_get(start_iso: str, end_iso: str) -> Optional[Any]:
    """Return an exact or wider cached price frame covering the requested range."""
    exact_key = ("prices", start_iso, end_iso)
    with _LOCK:
        exact = PRICE_DATA_CACHE.get(exact_key)
        if exact is not None:
            return exact
        for key, value in PRICE_DATA_CACHE.items():
            if (
                isinstance(key, tuple)
                and len(key) == 3
                and key[0] == "prices"
                and key[1] <= start_iso
                and key[2] >= end_iso
            ):
                return value
    return None


def price_cache_set(start_iso: str, end_iso: str, value: Any) -> None:
    with _LOCK:
        PRICE_DATA_CACHE[("prices", start_iso, end_iso)] = value


def clear_research_and_price_caches() -> None:
    """Drop cached OHLCV/research responses and shared Yahoo price batches (stale quotes)."""
    with _LOCK:
        RESEARCH_CACHE.clear()
        PRICE_DATA_CACHE.clear()


def clear_all_caches() -> None:
    """Wipe all in-memory TTL caches (use sparingly; constituents will refetch)."""
    with _LOCK:
        RESEARCH_CACHE.clear()
        PRICE_DATA_CACHE.clear()
        MOVERS_CACHE.clear()
        CROSSOVERS_CACHE.clear()
        RSI_SCAN_CACHE.clear()
        WEEKLY_MA_SCAN_CACHE.clear()
        MOVE_FINDER_CACHE.clear()
        ALPHA_CACHE.clear()
        INSTITUTIONAL_SCANNER_CACHE.clear()
        VALUATION_CACHE.clear()
        CONSTITUENTS_CACHE.clear()
