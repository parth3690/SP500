from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from ..models import Constituent


def compute_weekly_ma_watch(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    ma_length: int = 200,
    near_pct: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproduce the uploaded Pine script's default 200-week SMA watch."""
    constituent_list = list(constituents)
    total = len(constituent_list)
    empty_meta = {
        "total": total,
        "computed": 0,
        "skipped": total,
        "candidateCount": 0,
        "crossedBelowCount": 0,
        "belowCount": 0,
        "nearCount": 0,
        "reclaimedCount": 0,
        "maLength": ma_length,
        "nearPct": near_pct,
        "computedAt": datetime.utcnow().isoformat() + "Z",
    }
    if close_prices is None or close_prices.empty:
        return [], empty_meta

    prices = close_prices.sort_index()
    daily_sma = prices.rolling(window=ma_length, min_periods=ma_length).mean()
    weekly_prices = prices.resample("W-FRI").last()
    weekly_sma = weekly_prices.rolling(window=ma_length, min_periods=ma_length).mean()
    const_map = {constituent.yahooTicker: constituent for constituent in constituent_list}
    available = sorted(set(prices.columns) & set(const_map))

    rows: list[dict[str, Any]] = []
    computed = 0
    for yahoo_ticker in available:
        daily = prices[yahoo_ticker].dropna()
        daily_ma = daily_sma[yahoo_ticker].dropna()
        sma = weekly_sma[yahoo_ticker].dropna()
        weekly = weekly_prices[yahoo_ticker].dropna()
        if len(daily) < 2 or daily_ma.empty or len(weekly) < ma_length or sma.empty:
            continue

        current_price = float(daily.iloc[-1])
        previous_price = float(daily.iloc[-2])
        current_sma = float(sma.iloc[-1])
        current_daily_sma = float(daily_ma.iloc[-1])
        if current_sma <= 0 or current_daily_sma <= 0:
            continue

        current_date = daily.index[-1]
        previous_date = daily.index[-2]
        same_week = current_date.to_period("W-FRI") == previous_date.to_period("W-FRI")
        if same_week:
            previous_sma = current_sma + (previous_price - current_price) / ma_length
        elif len(sma) >= 2:
            previous_sma = float(sma.iloc[-2])
        else:
            continue

        computed += 1
        crossed_below = current_price < current_sma and previous_price >= previous_sma
        reclaimed = current_price > current_sma and previous_price <= previous_sma
        below = current_price <= current_sma
        distance_pct = (current_price - current_sma) / current_sma * 100.0
        near = current_price > current_sma and distance_pct <= near_pct

        if crossed_below:
            signal = "crossed_below"
        elif reclaimed:
            signal = "reclaimed"
        elif below:
            signal = "below"
        elif near:
            signal = "near"
        else:
            continue

        constituent = const_map[yahoo_ticker]
        rows.append(
            {
                "ticker": constituent.ticker,
                "companyName": constituent.companyName,
                "sector": constituent.sector,
                "currentPrice": round(current_price, 2),
                "priceDate": current_date.date() if hasattr(current_date, "date") else current_date,
                "dailySma": round(current_daily_sma, 2),
                "dailyDistancePct": round((current_price - current_daily_sma) / current_daily_sma * 100.0, 2),
                "weeklySma": round(current_sma, 2),
                "distancePct": round(distance_pct, 2),
                "signal": signal,
                "weeklyObservations": int(len(weekly)),
            }
        )

    priority = {"crossed_below": 0, "below": 1, "near": 2, "reclaimed": 3}
    rows.sort(key=lambda row: (priority[row["signal"]], abs(row["distancePct"]), row["ticker"]))
    counts = {
        signal: sum(1 for row in rows if row["signal"] == signal)
        for signal in priority
    }
    meta = {
        "total": total,
        "computed": computed,
        "skipped": total - computed,
        "candidateCount": len(rows),
        "crossedBelowCount": counts["crossed_below"],
        "belowCount": counts["below"],
        "nearCount": counts["near"],
        "reclaimedCount": counts["reclaimed"],
        "maLength": ma_length,
        "nearPct": near_pct,
        "computedAt": datetime.utcnow().isoformat() + "Z",
    }
    return rows, meta
