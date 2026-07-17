from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Iterator

import httpx
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


def fetch_fmp_price_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch one ticker's daily OHLCV history from FMP, returning Yahoo-style columns."""
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame()
    base = os.getenv("FMP_API_BASE", "https://financialmodelingprep.com/stable").rstrip("/")
    try:
        response = httpx.get(
            f"{base}/historical-price-eod/full",
            params={
                "symbol": symbol.strip().upper().replace(".", "-"),
                "from": start.isoformat(),
                "to": end.isoformat(),
                "apikey": api_key,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return pd.DataFrame()
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("date"):
            continue
        rows.append(
            {
                "Date": item.get("date"),
                "Open": item.get("open"),
                "High": item.get("high"),
                "Low": item.get("low"),
                "Close": item.get("close"),
                "Volume": item.get("volume"),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).set_index("Date").sort_index()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.attrs["source"] = "Financial Modeling Prep"
    return frame


def _fetch_fmp_close_frame(symbol: str, start: date, end: date) -> pd.DataFrame:
    history = fetch_fmp_price_history(symbol, start, end)
    if history.empty:
        return pd.DataFrame()
    return history[["Close"]].rename(columns={"Close": symbol})


def fetch_close_prices(
    yahoo_tickers: list[str],
    start: date,
    end: date,
    *,
    chunk_size: int = 200,
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

    frames: list[pd.DataFrame] = []
    if len(chunks) == 1:
        result = _download_chunk(chunks[0], start_iso, end_iso)
        if not result.empty:
            frames.append(result)
    else:
        with ThreadPoolExecutor(max_workers=min(2, len(chunks))) as pool:
            futures = {pool.submit(_download_chunk, c, start_iso, end_iso): c for c in chunks}
            for future in as_completed(futures):
                close = future.result()
                if not close.empty:
                    frames.append(close)

    out = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if not out.empty:
        out = out.loc[:, ~out.columns.duplicated()]
        out = out.dropna(axis=1, how="all")
    available = set(str(column).upper() for column in out.columns)
    missing = [symbol for symbol in unique if symbol not in available]

    # Broad Yahoo requests can return a partial frame without raising. Retry only
    # the gaps in small batches before asking the secondary provider.
    retry_size = max(0, int(os.getenv("YAHOO_RETRY_CHUNK_SIZE", "25")))
    retry_passes = max(0, int(os.getenv("YAHOO_RETRY_PASSES", "2")))
    for _ in range(retry_passes):
        if not missing or retry_size <= 0:
            break
        missing_before = len(missing)
        retry_chunks = list(_chunks(missing, retry_size))
        retry_frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=min(2, len(retry_chunks))) as pool:
            futures = {
                pool.submit(_download_chunk, chunk, start_iso, end_iso): chunk
                for chunk in retry_chunks
            }
            for future in as_completed(futures):
                frame = future.result()
                if not frame.empty:
                    retry_frames.append(frame)
        if not retry_frames:
            break
        out = pd.concat([out, *retry_frames], axis=1)
        out = out.loc[:, ~out.columns.duplicated()].dropna(axis=1, how="all")
        available = set(str(column).upper() for column in out.columns)
        missing = [symbol for symbol in unique if symbol not in available]
        if len(missing) >= missing_before:
            break

    fallback_max = int(os.getenv("FMP_PRICE_FALLBACK_MAX_TICKERS", "100"))
    if missing and len(missing) <= fallback_max and os.getenv("FMP_API_KEY", "").strip():
        fallback_frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=min(6, len(missing))) as pool:
            futures = {
                pool.submit(_fetch_fmp_close_frame, symbol, buffered_start, buffered_end): symbol
                for symbol in missing
            }
            for future in as_completed(futures):
                frame = future.result()
                if not frame.empty:
                    fallback_frames.append(frame)
        if fallback_frames:
            out = pd.concat([out, *fallback_frames], axis=1)

    if out.empty:
        return pd.DataFrame()
    out = out.loc[:, ~out.columns.duplicated()].sort_index()
    available = {
        str(column).upper()
        for column in out.columns
        if not out[column].dropna().empty
    }
    out.attrs["requestedTickers"] = len(unique)
    out.attrs["availableTickers"] = len(available)
    out.attrs["coveragePct"] = round(len(available) / len(unique) * 100, 1)
    out.attrs["missingTickers"] = [symbol for symbol in unique if symbol not in available]
    return out
