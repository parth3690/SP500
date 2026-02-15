from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable

import pandas as pd

from ..models import Constituent


def compute_crossovers(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    threshold_pct: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compute 50-DMA and 200-DMA for each constituent and identify stocks
    near a golden cross or death cross.

    - Near Golden Cross:  50-DMA < 200-DMA and the gap is within `threshold_pct`%
      (50-DMA is about to cross above 200-DMA → bullish)
    - Near Death Cross:   50-DMA > 200-DMA and the gap is within `threshold_pct`%
      (50-DMA is about to cross below 200-DMA → bearish)
    """
    if close_prices is None or close_prices.empty:
        return [], {"computed": 0, "total": 0, "nearGoldenCross": 0, "nearDeathCross": 0}

    close_prices = close_prices.sort_index()

    rows: list[dict[str, Any]] = []
    total = 0
    skipped = 0

    for c in constituents:
        total += 1

        if c.yahooTicker not in close_prices.columns:
            skipped += 1
            continue

        series = close_prices[c.yahooTicker].dropna()

        # Need at least 200 data points to compute 200-DMA
        if len(series) < 200:
            skipped += 1
            continue

        dma_50 = series.rolling(window=50).mean()
        dma_200 = series.rolling(window=200).mean()

        # Use the latest available values
        latest_dma_50 = float(dma_50.iloc[-1])
        latest_dma_200 = float(dma_200.iloc[-1])
        latest_price = float(series.iloc[-1])
        latest_date = series.index[-1]

        if pd.isna(latest_dma_50) or pd.isna(latest_dma_200) or latest_dma_200 == 0:
            skipped += 1
            continue

        # Gap as a percentage of 200-DMA
        gap_pct = ((latest_dma_50 - latest_dma_200) / latest_dma_200) * 100.0

        if abs(gap_pct) > threshold_pct:
            continue  # Not near a crossover

        if latest_dma_50 < latest_dma_200:
            signal = "near_golden_cross"
        elif latest_dma_50 > latest_dma_200:
            signal = "near_death_cross"
        else:
            signal = "near_golden_cross"  # Exactly equal, could go either way

        rows.append(
            {
                "ticker": c.ticker,
                "companyName": c.companyName,
                "sector": c.sector,
                "currentPrice": latest_price,
                "priceDate": latest_date.date() if hasattr(latest_date, "date") else latest_date,
                "dma50": round(latest_dma_50, 2),
                "dma200": round(latest_dma_200, 2),
                "gapPct": round(gap_pct, 4),
                "signal": signal,
            }
        )

    # Sort by absolute gap (closest to crossover first)
    rows.sort(key=lambda r: abs(r["gapPct"]))

    near_golden = sum(1 for r in rows if r["signal"] == "near_golden_cross")
    near_death = sum(1 for r in rows if r["signal"] == "near_death_cross")

    meta = {
        "total": total,
        "computed": total - skipped,
        "skipped": skipped,
        "nearGoldenCross": near_golden,
        "nearDeathCross": near_death,
        "thresholdPct": threshold_pct,
        "computedAt": datetime.utcnow().isoformat() + "Z",
    }

    return rows, meta
