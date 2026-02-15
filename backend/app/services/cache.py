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

_LOCK = RLock()


def cache_get(cache: TTLCache, key: Hashable) -> Optional[Any]:
    with _LOCK:
        return cache.get(key)


def cache_set(cache: TTLCache, key: Hashable, value: Any) -> None:
    with _LOCK:
        cache[key] = value

