from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _fetch_fmp_biggest_gainers() -> list[dict[str, Any]]:
    key = os.getenv("FMP_API_KEY", "").strip()
    if not key:
        return []
    base = os.getenv("FMP_API_BASE", "https://financialmodelingprep.com/stable").rstrip("/")
    url = f"{base}/biggest-gainers"
    try:
        r = httpx.get(url, params={"apikey": key}, timeout=15.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _fetch_fmp_quote(symbol: str) -> dict[str, Any] | None:
    key = os.getenv("FMP_API_KEY", "").strip()
    if not key:
        return None
    base = os.getenv("FMP_API_BASE", "https://financialmodelingprep.com/stable").rstrip("/")
    url = f"{base}/quote"
    try:
        r = httpx.get(url, params={"symbol": symbol, "apikey": key}, timeout=8.0)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def _score_move(row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    symbol = str(row.get("symbol", "")).upper()
    quote = _fetch_fmp_quote(symbol) or {}
    price = _f(quote.get("price", row.get("price")))
    pct = _f(quote.get("changePercentage", row.get("changePercentage", row.get("changesPercentage"))))
    volume = _f(quote.get("volume"))
    avg50 = _f(row.get("priceAvg50"), 0.0)
    avg200 = _f(row.get("priceAvg200"), 0.0)
    if avg50 == 0:
        avg50 = _f(quote.get("priceAvg50"), 0.0)
    if avg200 == 0:
        avg200 = _f(quote.get("priceAvg200"), 0.0)
    year_high = _f(quote.get("yearHigh", row.get("yearHigh")), 0.0)
    year_low = _f(quote.get("yearLow", row.get("yearLow")), 0.0)
    market_cap = _f(quote.get("marketCap", row.get("marketCap")), 0.0)
    day_low = _f(quote.get("dayLow", row.get("dayLow")), 0.0)
    day_high = _f(quote.get("dayHigh", row.get("dayHigh")), 0.0)

    trend50 = 1.0 if avg50 > 0 and price >= avg50 else 0.0
    trend200 = 1.0 if avg200 > 0 and price >= avg200 else 0.0
    if year_high > year_low:
        pos_52w = ((price - year_low) / (year_high - year_low)) * 100.0
    else:
        pos_52w = 50.0
    intraday_range_pct = ((day_high - day_low) / price) * 100.0 if price > 0 and day_high >= day_low else 0.0
    volume_score = min(max(volume, 1.0), 500_000_000.0)
    volume_score = max(0.0, (volume_score ** 0.25) - 1.0)  # smooth large caps

    score = (
        min(max(pct, 0.0), 100.0) * 0.45
        + volume_score * 3.0
        + min(max(intraday_range_pct, 0.0), 50.0) * 0.25
        + trend50 * 8.0
        + trend200 * 6.0
        + min(max(pos_52w, 0.0), 100.0) * 0.08
    )
    if price < 1:
        score -= 6.0
    if volume < 200_000:
        score -= 4.0
    if market_cap > 0 and market_cap < 200_000_000:
        score -= 2.0

    return score, {
        "price": round(price, 2),
        "changePct": round(pct, 2),
        "volume": int(volume) if volume > 0 else 0,
        "intradayRangePct": round(intraday_range_pct, 2),
        "priceAvg50": round(avg50, 2) if avg50 > 0 else None,
        "priceAvg200": round(avg200, 2) if avg200 > 0 else None,
        "marketCap": int(market_cap) if market_cap > 0 else None,
        "yearHigh": round(year_high, 2) if year_high > 0 else None,
        "yearLow": round(year_low, 2) if year_low > 0 else None,
        "position52wPct": round(pos_52w, 2),
        "score": round(score, 2),
    }


def find_runner_moves(
    *,
    min_change_pct: float = 15.0,
    min_volume: int = 200_000,
    min_price: float = 1.0,
    max_price: float = 150.0,
    limit: int = 25,
) -> dict[str, Any]:
    """
    Momentum runner finder for names like AXTI/VCX-style explosive moves.
    Uses FMP biggest gainers universe, then scores by:
    % move, relative volume, trend position vs 50/200DMA, and 52w position.
    """
    rows = _fetch_fmp_biggest_gainers()
    if not rows:
        return {
            "asOf": datetime.now(timezone.utc).isoformat(),
            "source": "Financial Modeling Prep",
            "candidates": [],
            "meta": {"totalFetched": 0, "qualified": 0},
        }

    out: list[dict[str, Any]] = []
    for r in rows:
        score, m = _score_move(r)
        if m["changePct"] < min_change_pct:
            continue
        if m["volume"] < min_volume:
            continue
        if m["price"] < min_price or m["price"] > max_price:
            continue

        out.append(
            {
                "ticker": str(r.get("symbol", "")).upper(),
                "name": r.get("name"),
                **m,
                "signal": "runner" if score >= 35 else "watch",
            }
        )

    out.sort(key=lambda x: x["score"], reverse=True)
    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "source": "Financial Modeling Prep",
        "algorithm": {
            "name": "Explosive Move Finder v1",
            "features": [
                "changePct",
                "volume",
                "intradayRangePct",
                "price_vs_50dma",
                "price_vs_200dma",
                "position_52w",
            ],
        },
        "params": {
            "minChangePct": min_change_pct,
            "minVolume": min_volume,
            "minPrice": min_price,
            "maxPrice": max_price,
            "limit": limit,
        },
        "candidates": out[: max(1, min(limit, 100))],
        "meta": {"totalFetched": len(rows), "qualified": len(out)},
    }
