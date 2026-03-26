"""
Financial Modeling Prep (FMP) API — live quote overlay for research.

Dashboard: https://site.financialmodelingprep.com/developer/docs/dashboard
Stable quote docs: https://site.financialmodelingprep.com/developer/docs/stable/quote

Set FMP_API_KEY in the environment. When unset, research uses Yahoo last bar only.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx


def fmp_symbol(yahoo_or_display: str) -> str:
    """FMP expects exchange-style symbols; map Yahoo BRK.B → BRK-B."""
    return yahoo_or_display.strip().upper().replace(".", "-")


def fetch_fmp_quote(symbol: str, *, timeout: float = 12.0) -> Optional[dict[str, Any]]:
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        return None

    base = os.getenv("FMP_API_BASE", "https://financialmodelingprep.com/stable").rstrip("/")
    url = f"{base}/quote"

    try:
        resp = httpx.get(url, params={"symbol": symbol, "apikey": api_key}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if isinstance(data, list) and data:
        row = data[0]
        return row if isinstance(row, dict) else None
    if isinstance(data, dict) and data.get("symbol") is not None:
        return data
    return None


def merge_fmp_live_into_research(payload: dict[str, Any], *, fmp_symbol_str: str) -> None:
    """
    If FMP returns a quote, overlay header fields with live price / change / volume.
    Preserves Yahoo last-bar close as chartLastClose for chart context.
    """
    q = fetch_fmp_quote(fmp_symbol_str)
    if not q:
        return

    price = q.get("price")
    if price is None:
        return
    try:
        price_f = float(price)
    except (TypeError, ValueError):
        return

    # Preserve pre-merge values (last daily close from Yahoo in selected range)
    payload["chartLastClose"] = round(float(payload.get("currentPrice", price_f)), 2)

    prev = q.get("previousClose")
    ch = q.get("change")
    chp = q.get("changePercentage")
    vol = q.get("volume")

    payload["currentPrice"] = round(price_f, 2)
    if prev is not None:
        try:
            payload["previousClose"] = round(float(prev), 2)
        except (TypeError, ValueError):
            pass
    if ch is not None:
        try:
            payload["change"] = round(float(ch), 2)
        except (TypeError, ValueError):
            pass
    if chp is not None:
        try:
            payload["changePct"] = round(float(chp), 2)
        except (TypeError, ValueError):
            pass
    elif prev is not None:
        try:
            p = float(prev)
            if p != 0:
                payload["changePct"] = round(((price_f - p) / p) * 100.0, 2)
        except (TypeError, ValueError):
            pass

    if vol is not None:
        try:
            payload["volume"] = int(float(vol))
        except (TypeError, ValueError):
            pass

    ts = q.get("timestamp")
    as_of: Optional[str] = None
    if ts is not None:
        try:
            as_of = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            pass

    payload["liveQuote"] = {
        "source": "Financial Modeling Prep",
        "providerUrl": "https://site.financialmodelingprep.com/developer/docs",
        "symbol": q.get("symbol"),
        "asOf": as_of,
    }
