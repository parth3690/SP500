from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator

import pandas as pd


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _extract_close_prices(download_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if download_df is None or download_df.empty:
        return pd.DataFrame()

    if isinstance(download_df.columns, pd.MultiIndex):
        level0 = set(download_df.columns.get_level_values(0))
        level1 = set(download_df.columns.get_level_values(1))

        if "Close" in level0:
            close = download_df["Close"]
        elif "Close" in level1:
            close = download_df.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Unexpected yfinance columns; unable to locate Close prices.")
        close.columns = [str(c) for c in close.columns]
        return close

    # Single ticker case: OHLC columns in a flat index.
    if "Close" not in download_df.columns:
        return pd.DataFrame()
    ticker = tickers[0] if tickers else "TICKER"
    return download_df[["Close"]].rename(columns={"Close": ticker})


def fetch_close_prices(
    yahoo_tickers: list[str],
    start: date,
    end: date,
    *,
    chunk_size: int = 400,
) -> pd.DataFrame:
    """
    Fetch daily close prices for many tickers via yfinance.

    Notes:
    - Uses chunking to reduce the chance of Yahoo throttling.
    - Adds a small date buffer so "past price" can resolve on/before start date.
    """

    import yfinance as yf  # local import keeps module import cost off cold paths

    if not yahoo_tickers:
        return pd.DataFrame()

    buffered_start = start - timedelta(days=7)
    buffered_end = end + timedelta(days=1)

    frames: list[pd.DataFrame] = []
    unique = list(dict.fromkeys([t.strip().upper() for t in yahoo_tickers if t and t.strip()]))

    for chunk in _chunks(unique, chunk_size):
        try:
            df = yf.download(
                tickers=chunk,
                start=buffered_start.isoformat(),
                end=buffered_end.isoformat(),
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            continue
        close = _extract_close_prices(df, chunk)
        if not close.empty:
            frames.append(close)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].sort_index()
    return out
