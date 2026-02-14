from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

from ..models import Constituent


def _cutoff_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d.year, d.month, d.day, 23, 59, 59)


def compute_movers(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    start: date,
    end: date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Compute % change between the last close on/before `start` and on/before `end`.
    """

    if close_prices is None or close_prices.empty:
        return [], [], {"missingTickers": [], "computed": 0, "total": 0}

    close_prices = close_prices.sort_index()

    start_cutoff = _cutoff_ts(start)
    end_cutoff = _cutoff_ts(end)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    total = 0

    for c in constituents:
        total += 1
        if c.yahooTicker not in close_prices.columns:
            missing.append(c.ticker)
            continue

        series = close_prices[c.yahooTicker].dropna()
        if series.empty:
            missing.append(c.ticker)
            continue

        past = series.loc[:start_cutoff]
        curr = series.loc[:end_cutoff]
        if past.empty or curr.empty:
            missing.append(c.ticker)
            continue

        past_price = float(past.iloc[-1])
        curr_price = float(curr.iloc[-1])
        if past_price == 0:
            missing.append(c.ticker)
            continue

        pct_change = (curr_price / past_price - 1.0) * 100.0

        rows.append(
            {
                "ticker": c.ticker,
                "companyName": c.companyName,
                "sector": c.sector,
                "currentPrice": curr_price,
                "currentPriceDate": curr.index[-1].date(),
                "pastPrice": past_price,
                "pastPriceDate": past.index[-1].date(),
                "pctChange": pct_change,
            }
        )

    movers_df = pd.DataFrame(rows)
    if movers_df.empty:
        return [], [], {"missingTickers": missing, "computed": 0, "total": total}

    # Sector summary
    sector_groups = movers_df.groupby("sector", dropna=False)
    sector_summary = (
        sector_groups["pctChange"]
        .agg(["count", "mean", "median"])
        .rename(columns={"count": "count", "mean": "avgPctChange", "median": "medianPctChange"})
    )
    pos = movers_df[movers_df["pctChange"] > 0].groupby("sector")["pctChange"].size().rename("positiveCount")
    neg = movers_df[movers_df["pctChange"] < 0].groupby("sector")["pctChange"].size().rename("negativeCount")
    sector_summary = sector_summary.join(pos, how="left").join(neg, how="left").fillna(0)
    sector_summary["positiveCount"] = sector_summary["positiveCount"].astype(int)
    sector_summary["negativeCount"] = sector_summary["negativeCount"].astype(int)
    sector_summary = sector_summary.reset_index().sort_values("avgPctChange", ascending=False)

    meta: dict[str, Any] = {
        "total": total,
        "computed": int(movers_df.shape[0]),
        "missingCount": len(missing),
        "missingTickers": missing,
        "computedAt": datetime.utcnow().isoformat() + "Z",
    }

    return (
        movers_df.to_dict(orient="records"),
        sector_summary.to_dict(orient="records"),
        meta,
    )

