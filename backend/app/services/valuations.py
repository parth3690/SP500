from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import pandas as pd

from .cache import VALUATION_CACHE, cache_get, cache_set
from .sp500 import normalize_yahoo_ticker


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        out = float(v)
        if out <= 0:
            return None
        return round(out, 2)
    except (TypeError, ValueError):
        return None


def _fetch_one(ticker: str) -> dict[str, Optional[float]]:
    cached = cache_get(VALUATION_CACHE, ticker)
    if cached is not None:
        return cached

    trailing_pe = None
    forward_pe = None
    fetched = False
    try:
        import yfinance as yf

        info = yf.Ticker(normalize_yahoo_ticker(ticker)).info or {}
        fetched = bool(info)
        trailing_pe = _safe_float(info.get("trailingPE"))
        forward_pe = _safe_float(info.get("forwardPE"))
    except Exception:
        pass

    payload = {"trailingPE": trailing_pe, "forwardPE": forward_pe}
    if fetched:
        cache_set(VALUATION_CACHE, ticker, payload)
    return payload


def fetch_pe_metrics(tickers: list[str], *, max_workers: int = 8) -> dict[str, dict[str, Optional[float]]]:
    unique = list(dict.fromkeys([t.strip().upper() for t in tickers if t and t.strip()]))
    if not unique:
        return {}

    out: dict[str, dict[str, Optional[float]]] = {}
    missing = [t for t in unique if cache_get(VALUATION_CACHE, t) is None]

    for ticker in unique:
        cached = cache_get(VALUATION_CACHE, ticker)
        if cached is not None:
            out[ticker] = cached

    if missing:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(missing))) as pool:
            futures = {pool.submit(_fetch_one, ticker): ticker for ticker in missing}
            for future in as_completed(futures):
                ticker = futures[future]
                out[ticker] = future.result()

    return out


def attach_pe_metrics(rows: list[dict[str, Any]], metrics: dict[str, dict[str, Optional[float]]]) -> None:
    for row in rows:
        ticker = str(row.get("ticker", "")).upper()
        vals = metrics.get(ticker) or {}
        row["trailingPE"] = vals.get("trailingPE")
        row["forwardPE"] = vals.get("forwardPE")
