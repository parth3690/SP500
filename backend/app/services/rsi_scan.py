from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..models import Constituent


def _batch_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute RSI for ALL columns at once using vectorized EWM (Wilder's).
    Same formula as indicators.compute_rsi_series (alpha=1/period, RS = avg_gain/avg_loss).
    """
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


def _build_constituent_map(
    constituents: Iterable[Constituent],
) -> tuple[dict[str, Constituent], int]:
    const_map: dict[str, Constituent] = {}
    total = 0
    for c in constituents:
        total += 1
        const_map[c.yahooTicker] = c
    return const_map, total


def _compute_rsi_scan(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    rsi_threshold: float,
    timeframe: str,
    direction: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if close_prices is None or close_prices.empty:
        count_key = "oversoldCount" if direction == "oversold" else "overboughtCount"
        return [], {"computed": 0, "total": 0, "skipped": 0, count_key: 0, "rsiThreshold": rsi_threshold}

    close_prices = close_prices.sort_index()
    weekly_prices = close_prices.resample("W-FRI").last().dropna(how="all")
    weekly_rsi_all = _batch_rsi(weekly_prices, 14)
    daily_rsi_all = _batch_rsi(close_prices, 14)

    primary = weekly_rsi_all if timeframe == "weekly" else daily_rsi_all
    primary_key = "weeklyRSI" if timeframe == "weekly" else "dailyRSI"
    count_key = "oversoldCount" if direction == "oversold" else "overboughtCount"
    descending = direction == "overbought"
    const_map, total = _build_constituent_map(constituents)
    available = sorted(set(primary.columns) & set(const_map.keys()))
    rows: list[dict[str, Any]] = []
    skipped = 0

    for ticker in available:
        rsi_col = primary[ticker].dropna()
        if len(rsi_col) < 1:
            skipped += 1
            continue

        latest_rsi = float(rsi_col.iloc[-1])
        if pd.isna(latest_rsi):
            continue
        if direction == "oversold" and latest_rsi > rsi_threshold:
            continue
        if direction == "overbought" and latest_rsi < rsi_threshold:
            continue

        c = const_map[ticker]
        weekly_rsi = round(latest_rsi, 2) if timeframe == "weekly" else None
        daily_rsi = round(latest_rsi, 2) if timeframe == "daily" else None

        if ticker in weekly_rsi_all.columns:
            weekly_col = weekly_rsi_all[ticker].dropna()
            if len(weekly_col) > 0 and timeframe != "weekly":
                weekly_rsi = round(float(weekly_col.iloc[-1]), 2)
        if ticker in daily_rsi_all.columns:
            daily_col = daily_rsi_all[ticker].dropna()
            if len(daily_col) > 0 and timeframe != "daily":
                daily_rsi = round(float(daily_col.iloc[-1]), 2)

        price_col = close_prices[ticker].dropna()
        if len(price_col) < 1:
            skipped += 1
            continue
        latest_price = float(price_col.iloc[-1])
        latest_date = price_col.index[-1]

        rows.append({
            "ticker": c.ticker,
            "companyName": c.companyName,
            "sector": c.sector,
            "currentPrice": round(latest_price, 2),
            "priceDate": latest_date.date() if hasattr(latest_date, "date") else latest_date,
            "weeklyRSI": weekly_rsi,
            "dailyRSI": daily_rsi,
        })

    unavailable = total - len(available)
    skipped += unavailable
    rows.sort(key=lambda r: r[primary_key], reverse=descending)

    meta = {
        "total": total,
        "computed": max(len(available) - (skipped - unavailable), 0),
        "skipped": skipped,
        count_key: len(rows),
        "rsiThreshold": rsi_threshold,
        "computedAt": datetime.utcnow().isoformat() + "Z",
    }
    return rows, meta


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
    return _compute_rsi_scan(
        constituents,
        close_prices,
        rsi_threshold=rsi_threshold,
        timeframe="weekly",
        direction="oversold",
    )


def compute_rsi_scan_overbought(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    rsi_threshold: float = 70.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Weekly RSI scan for stocks where weekly RSI is at or above threshold."""
    return _compute_rsi_scan(
        constituents,
        close_prices,
        rsi_threshold=rsi_threshold,
        timeframe="weekly",
        direction="overbought",
    )


def compute_rsi_scan_daily_oversold(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    rsi_threshold: float = 30.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Daily RSI oversold scan; weekly RSI is included for reference."""
    return _compute_rsi_scan(
        constituents,
        close_prices,
        rsi_threshold=rsi_threshold,
        timeframe="daily",
        direction="oversold",
    )


def compute_rsi_scan_daily_overbought(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    rsi_threshold: float = 70.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Daily RSI overbought scan; weekly RSI is included for reference."""
    return _compute_rsi_scan(
        constituents,
        close_prices,
        rsi_threshold=rsi_threshold,
        timeframe="daily",
        direction="overbought",
    )
