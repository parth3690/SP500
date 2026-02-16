from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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

    if "Close" not in download_df.columns:
        return pd.DataFrame()
    ticker = tickers[0] if tickers else "TICKER"
    return download_df[["Close"]].rename(columns={"Close": ticker})


def _download_chunk(chunk: list[str], start_iso: str, end_iso: str) -> pd.DataFrame:
    """Download a single chunk — designed to run in a thread."""
    import yfinance as yf

    try:
        df = yf.download(
            tickers=chunk,
            start=start_iso,
            end=end_iso,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()
    return _extract_close_prices(df, chunk)


def fetch_close_prices(
    yahoo_tickers: list[str],
    start: date,
    end: date,
    *,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """
    Fetch daily close prices for many tickers via yfinance.

    Chunks are downloaded in parallel threads for maximum throughput.
    """
    if not yahoo_tickers:
        return pd.DataFrame()

    buffered_start = start - timedelta(days=7)
    buffered_end = end + timedelta(days=1)
    start_iso = buffered_start.isoformat()
    end_iso = buffered_end.isoformat()

    unique = list(dict.fromkeys([t.strip().upper() for t in yahoo_tickers if t and t.strip()]))

    chunks = list(_chunks(unique, chunk_size))

    # If only one chunk, download directly (no thread overhead)
    if len(chunks) == 1:
        result = _download_chunk(chunks[0], start_iso, end_iso)
        return result if not result.empty else pd.DataFrame()

    # Parallel download for multiple chunks
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
        futures = {pool.submit(_download_chunk, c, start_iso, end_iso): c for c in chunks}
        for future in as_completed(futures):
            close = future.result()
            if not close.empty:
                frames.append(close)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].sort_index()
    return out
