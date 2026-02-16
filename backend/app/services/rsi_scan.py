from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..models import Constituent


def _batch_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute RSI for ALL columns at once using vectorized EWM (Wilder's)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.iloc[:period] = np.nan
    return rsi


def compute_rsi_scan(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    rsi_threshold: float = 30.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Vectorised weekly RSI scan across all S&P 500 tickers.
    Computes weekly resample + RSI in one pass per DataFrame instead of
    looping per ticker.
    """
    if close_prices is None or close_prices.empty:
        return [], {"computed": 0, "total": 0, "oversoldCount": 0}

    close_prices = close_prices.sort_index()

    # Resample to weekly (Friday close) — one operation for ALL tickers
    weekly_prices = close_prices.resample("W-FRI").last().dropna(how="all")

    # Batch compute weekly RSI for ALL tickers at once
    weekly_rsi_all = _batch_rsi(weekly_prices, 14)
    # Batch compute daily RSI for ALL tickers at once
    daily_rsi_all = _batch_rsi(close_prices, 14)

    # Build lookup
    const_map: dict[str, Constituent] = {}
    total = 0
    for c in constituents:
        total += 1
        const_map[c.yahooTicker] = c

    rows: list[dict[str, Any]] = []
    skipped = 0
    available = set(weekly_rsi_all.columns) & set(const_map.keys())

    for ticker in available:
        weekly_col = weekly_rsi_all[ticker].dropna()
        if len(weekly_col) < 1:
            skipped += 1
            continue

        latest_rsi = float(weekly_col.iloc[-1])
        if pd.isna(latest_rsi) or latest_rsi > rsi_threshold:
            continue

        c = const_map[ticker]

        # Daily RSI for reference
        daily_rsi = None
        if ticker in daily_rsi_all.columns:
            daily_col = daily_rsi_all[ticker].dropna()
            if len(daily_col) > 0:
                daily_rsi = round(float(daily_col.iloc[-1]), 2)

        price_col = close_prices[ticker].dropna()
        latest_price = float(price_col.iloc[-1])
        latest_date = price_col.index[-1]

        rows.append({
            "ticker": c.ticker,
            "companyName": c.companyName,
            "sector": c.sector,
            "currentPrice": round(latest_price, 2),
            "priceDate": latest_date.date() if hasattr(latest_date, "date") else latest_date,
            "weeklyRSI": round(latest_rsi, 2),
            "dailyRSI": daily_rsi,
        })

    skipped += total - len(available)

    rows.sort(key=lambda r: r["weeklyRSI"])

    meta = {
        "total": total,
        "computed": len(available) - skipped,
        "skipped": skipped,
        "oversoldCount": len(rows),
        "rsiThreshold": rsi_threshold,
        "computedAt": datetime.utcnow().isoformat() + "Z",
    }

    return rows, meta
