from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from ..models import Constituent


def _compute_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing method."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder's smoothing for subsequent values
    for i in range(period, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_rsi_scan(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    rsi_threshold: float = 30.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compute weekly RSI for each S&P 500 constituent and return stocks
    where the weekly RSI is at or below `rsi_threshold` (oversold).
    """
    if close_prices is None or close_prices.empty:
        return [], {"computed": 0, "total": 0, "oversoldCount": 0}

    close_prices = close_prices.sort_index()

    # Resample to weekly (Friday close) for weekly RSI
    weekly_prices = close_prices.resample("W-FRI").last().dropna(how="all")

    rows: list[dict[str, Any]] = []
    total = 0
    skipped = 0

    for c in constituents:
        total += 1

        if c.yahooTicker not in weekly_prices.columns:
            skipped += 1
            continue

        weekly_series = weekly_prices[c.yahooTicker].dropna()

        # Need at least 15 data points (14 for RSI period + 1 for diff)
        if len(weekly_series) < 15:
            skipped += 1
            continue

        rsi_series = _compute_rsi_series(weekly_series, period=14)
        latest_rsi = float(rsi_series.iloc[-1])

        if pd.isna(latest_rsi):
            skipped += 1
            continue

        # Also compute daily RSI for reference
        daily_series = close_prices[c.yahooTicker].dropna() if c.yahooTicker in close_prices.columns else None
        daily_rsi = None
        if daily_series is not None and len(daily_series) >= 15:
            daily_rsi_series = _compute_rsi_series(daily_series, period=14)
            daily_rsi_val = float(daily_rsi_series.iloc[-1])
            if not pd.isna(daily_rsi_val):
                daily_rsi = round(daily_rsi_val, 2)

        latest_price = float(close_prices[c.yahooTicker].dropna().iloc[-1])
        latest_date = close_prices[c.yahooTicker].dropna().index[-1]

        # Only include if weekly RSI is at or below the threshold
        if latest_rsi > rsi_threshold:
            continue

        rows.append(
            {
                "ticker": c.ticker,
                "companyName": c.companyName,
                "sector": c.sector,
                "currentPrice": round(latest_price, 2),
                "priceDate": latest_date.date() if hasattr(latest_date, "date") else latest_date,
                "weeklyRSI": round(latest_rsi, 2),
                "dailyRSI": daily_rsi,
            }
        )

    # Sort by weekly RSI (most oversold first)
    rows.sort(key=lambda r: r["weeklyRSI"])

    meta = {
        "total": total,
        "computed": total - skipped,
        "skipped": skipped,
        "oversoldCount": len(rows),
        "rsiThreshold": rsi_threshold,
        "computedAt": datetime.utcnow().isoformat() + "Z",
    }

    return rows, meta
