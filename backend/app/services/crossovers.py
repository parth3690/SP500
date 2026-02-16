from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..models import Constituent


def compute_crossovers(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    threshold_pct: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Vectorised 50/200-DMA crossover detection across all S&P 500 tickers.

    Instead of computing rolling means one ticker at a time, compute both
    rolling windows across the entire DataFrame at once — orders of magnitude
    faster for 500 columns.
    """
    if close_prices is None or close_prices.empty:
        return [], {"computed": 0, "total": 0, "nearGoldenCross": 0, "nearDeathCross": 0}

    close_prices = close_prices.sort_index()

    # Batch-compute rolling means for every column in one call
    dma50_all = close_prices.rolling(window=50, min_periods=50).mean()
    dma200_all = close_prices.rolling(window=200, min_periods=200).mean()

    # Build a lookup {yahooTicker: Constituent} for O(1) access
    const_map: dict[str, Constituent] = {}
    total = 0
    for c in constituents:
        total += 1
        const_map[c.yahooTicker] = c

    rows: list[dict[str, Any]] = []
    skipped = 0

    # Iterate only over tickers present in both the constituents and the df
    available = set(close_prices.columns) & set(const_map.keys())

    for ticker in available:
        c = const_map[ticker]
        d50 = dma50_all[ticker].iloc[-1]
        d200 = dma200_all[ticker].iloc[-1]

        if pd.isna(d50) or pd.isna(d200) or d200 == 0:
            skipped += 1
            continue

        d50_f = float(d50)
        d200_f = float(d200)
        gap_pct = ((d50_f - d200_f) / d200_f) * 100.0

        if abs(gap_pct) > threshold_pct:
            continue

        latest_price = float(close_prices[ticker].dropna().iloc[-1])
        latest_date = close_prices[ticker].dropna().index[-1]

        signal = "near_golden_cross" if d50_f <= d200_f else "near_death_cross"

        rows.append({
            "ticker": c.ticker,
            "companyName": c.companyName,
            "sector": c.sector,
            "currentPrice": latest_price,
            "priceDate": latest_date.date() if hasattr(latest_date, "date") else latest_date,
            "dma50": round(d50_f, 2),
            "dma200": round(d200_f, 2),
            "gapPct": round(gap_pct, 4),
            "signal": signal,
        })

    skipped += total - len(available) - skipped

    rows.sort(key=lambda r: abs(r["gapPct"]))

    near_golden = sum(1 for r in rows if r["signal"] == "near_golden_cross")
    near_death = sum(1 for r in rows if r["signal"] == "near_death_cross")

    meta = {
        "total": total,
        "computed": len(available) - skipped,
        "skipped": total - len(available) + skipped,
        "nearGoldenCross": near_golden,
        "nearDeathCross": near_death,
        "thresholdPct": threshold_pct,
        "computedAt": datetime.utcnow().isoformat() + "Z",
    }

    return rows, meta
