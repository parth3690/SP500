from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool

from dotenv import load_dotenv

from .models import (
    Constituent, MoversResponse, MoverRow, SectorSummaryRow,
    CrossoverRow, CrossoversResponse,
    OversoldRow, OversoldResponse,
)
from .services.cache import (
    MOVERS_CACHE, CROSSOVERS_CACHE, RESEARCH_CACHE, RSI_SCAN_CACHE,
    PRICE_DATA_CACHE,
    cache_get, cache_set,
)
from .services.movers import compute_movers
from .services.crossovers import compute_crossovers
from .services.research import compute_research
from .services.rsi_scan import compute_rsi_scan
from .services.prices import fetch_close_prices
from .services.sp500 import get_sp500_constituents_cached, get_yahoo_tickers, normalize_yahoo_ticker

load_dotenv()

DEFAULT_RANGE_DAYS = int(os.getenv("DEFAULT_RANGE_DAYS", "30"))
MAX_RANGE_DAYS = int(os.getenv("MAX_RANGE_DAYS", "366"))


def _parse_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="S&P 500 Monthly Movers Analyzer API", version="0.1.0")

origins = _parse_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False if origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Shared price data helper ─────────────────────────────────────────────


def _get_shared_price_data(
    yahoo_tickers: list[str],
    start: date,
    end: date,
    *,
    refresh: bool = False,
) -> Any:
    """
    Fetch (or reuse cached) close prices for all S&P 500 tickers.
    This avoids redundant Yahoo Finance downloads across endpoints.
    """
    cache_key = f"prices_{start.isoformat()}_{end.isoformat()}"
    if not refresh:
        cached = cache_get(PRICE_DATA_CACHE, cache_key)
        if cached is not None:
            return cached

    prices = fetch_close_prices(yahoo_tickers, start, end)
    cache_set(PRICE_DATA_CACHE, cache_key, prices)
    return prices


# ── Background preload on startup ─────────────────────────────────────────


async def _preload_dashboard_data() -> None:
    """Preload constituents and price data in background so first request is fast."""
    try:
        constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
        yahoo_tickers = get_yahoo_tickers(constituents_list)

        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        # Single download for all dashboard endpoints
        close_prices = await run_in_threadpool(
            _get_shared_price_data, yahoo_tickers, start_date, end_date
        )

        # Pre-compute movers (default 30-day range)
        movers_start = end_date - timedelta(days=DEFAULT_RANGE_DAYS)
        movers_key = (movers_start.isoformat(), end_date.isoformat())
        if cache_get(MOVERS_CACHE, movers_key) is None:
            rows, sector_summary, meta = await run_in_threadpool(
                compute_movers, constituents_list, close_prices, movers_start, end_date
            )
            cache_set(MOVERS_CACHE, movers_key, {
                "rows": rows,
                "sectorSummary": sector_summary,
                "meta": meta,
                "asOf": datetime.now(timezone.utc),
            })

        # Pre-compute crossovers
        crossover_key = "crossovers_2.0"
        if cache_get(CROSSOVERS_CACHE, crossover_key) is None:
            c_rows, c_meta = await run_in_threadpool(
                compute_crossovers, constituents_list, close_prices, threshold_pct=2.0
            )
            cache_set(CROSSOVERS_CACHE, crossover_key, {
                "rows": c_rows, "meta": c_meta,
                "asOf": datetime.now(timezone.utc),
            })

        # Pre-compute RSI oversold
        rsi_key = "rsi_oversold_30.0"
        if cache_get(RSI_SCAN_CACHE, rsi_key) is None:
            r_rows, r_meta = await run_in_threadpool(
                compute_rsi_scan, constituents_list, close_prices, rsi_threshold=30.0
            )
            cache_set(RSI_SCAN_CACHE, rsi_key, {
                "rows": r_rows, "meta": r_meta,
                "asOf": datetime.now(timezone.utc),
            })

    except Exception as e:
        # Non-fatal: first request will just compute on demand
        print(f"[preload] Background preload failed (non-fatal): {e}")


@app.on_event("startup")
async def startup_event() -> None:
    """Kick off background preload so first page load is fast."""
    asyncio.create_task(_preload_dashboard_data())


# ── Endpoints ────────────────────────────────────────────────────────────


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


def _resolve_dates(start: Optional[date], end: Optional[date]) -> tuple[date, date]:
    end_date = end or date.today()
    start_date = start or (end_date - timedelta(days=DEFAULT_RANGE_DAYS))
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start must be on/before end")
    if (end_date - start_date).days > MAX_RANGE_DAYS:
        raise HTTPException(status_code=400, detail=f"date range exceeds {MAX_RANGE_DAYS} days")
    return start_date, end_date


@app.get("/api/constituents", response_model=list[Constituent])
async def constituents(refresh: bool = Query(False)) -> list[Constituent]:
    return await run_in_threadpool(get_sp500_constituents_cached, refresh=refresh)


def _ranked(rows: list[dict[str, Any]], *, descending: bool, limit: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: r["pctChange"], reverse=descending)
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(sorted_rows[:limit], start=1):
        out.append({"rank": idx, **row})
    return out


@app.get("/api/movers", response_model=MoversResponse)
async def movers(
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    include_all: bool = Query(False, alias="includeAll"),
    refresh: bool = Query(False),
) -> MoversResponse:
    start_date, end_date = _resolve_dates(start, end)
    cache_key = (start_date.isoformat(), end_date.isoformat())

    cached = None if refresh else cache_get(MOVERS_CACHE, cache_key)
    if cached is None:
        constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
        yahoo_tickers = get_yahoo_tickers(constituents_list)

        close_prices = await run_in_threadpool(
            _get_shared_price_data, yahoo_tickers, start_date, end_date, refresh=refresh
        )
        rows, sector_summary, meta = await run_in_threadpool(
            compute_movers, constituents_list, close_prices, start_date, end_date
        )

        cached = {
            "rows": rows,
            "sectorSummary": sector_summary,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        cache_set(MOVERS_CACHE, cache_key, cached)

    rows = cached["rows"]
    gainers = _ranked(rows, descending=True, limit=limit)
    losers = _ranked(rows, descending=False, limit=limit)

    all_rows = None
    if include_all:
        all_rows = _ranked(rows, descending=True, limit=len(rows))

    return MoversResponse(
        start=start_date,
        end=end_date,
        asOf=cached["asOf"],
        gainers=[MoverRow(**r) for r in gainers],
        losers=[MoverRow(**r) for r in losers],
        sectorSummary=[SectorSummaryRow(**r) for r in cached["sectorSummary"]],
        meta=cached["meta"],
        all=[MoverRow(**r) for r in all_rows] if all_rows is not None else None,
    )


@app.get("/api/crossovers", response_model=CrossoversResponse)
async def crossovers(
    threshold: float = Query(2.0, ge=0.1, le=10.0, description="Max gap (%) between 50-DMA and 200-DMA"),
    refresh: bool = Query(False),
) -> CrossoversResponse:
    """
    Returns stocks where the 50-DMA and 200-DMA are within `threshold`%
    of each other, signalling a potential golden cross or death cross.
    """
    cache_key = f"crossovers_{threshold}"

    cached = None if refresh else cache_get(CROSSOVERS_CACHE, cache_key)
    if cached is None:
        constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
        yahoo_tickers = get_yahoo_tickers(constituents_list)

        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        close_prices = await run_in_threadpool(
            _get_shared_price_data, yahoo_tickers, start_date, end_date, refresh=refresh
        )
        rows, meta = await run_in_threadpool(
            compute_crossovers, constituents_list, close_prices, threshold_pct=threshold
        )

        cached = {
            "rows": rows,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        cache_set(CROSSOVERS_CACHE, cache_key, cached)

    rows = cached["rows"]
    near_golden = [CrossoverRow(**r) for r in rows if r["signal"] == "near_golden_cross"]
    near_death = [CrossoverRow(**r) for r in rows if r["signal"] == "near_death_cross"]

    return CrossoversResponse(
        asOf=cached["asOf"],
        thresholdPct=threshold,
        nearGoldenCross=near_golden,
        nearDeathCross=near_death,
        meta=cached["meta"],
    )


@app.get("/api/rsi-oversold", response_model=OversoldResponse)
async def rsi_oversold(
    threshold: float = Query(30.0, ge=1.0, le=50.0, description="Weekly RSI threshold (stocks at or below this are returned)"),
    refresh: bool = Query(False),
) -> OversoldResponse:
    """
    Returns S&P 500 stocks where the weekly (14-period) RSI is at or below
    the given threshold, highlighting oversold conditions.
    """
    cache_key = f"rsi_oversold_{threshold}"

    cached = None if refresh else cache_get(RSI_SCAN_CACHE, cache_key)
    if cached is None:
        constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
        yahoo_tickers = get_yahoo_tickers(constituents_list)

        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        close_prices = await run_in_threadpool(
            _get_shared_price_data, yahoo_tickers, start_date, end_date, refresh=refresh
        )
        rows, meta = await run_in_threadpool(
            compute_rsi_scan, constituents_list, close_prices, rsi_threshold=threshold
        )

        cached = {
            "rows": rows,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        cache_set(RSI_SCAN_CACHE, cache_key, cached)

    return OversoldResponse(
        asOf=cached["asOf"],
        rsiThreshold=threshold,
        stocks=[OversoldRow(**r) for r in cached["rows"]],
        meta=cached["meta"],
    )


@app.get("/api/research/{ticker}")
async def research(
    ticker: str,
    start: Optional[date] = Query(None, description="Start date (YYYY-MM-DD). Defaults to 365 days ago."),
    end: Optional[date] = Query(None, description="End date (YYYY-MM-DD). Defaults to today."),
    refresh: bool = Query(False),
) -> dict:
    """
    Deep research for a single ticker: OHLCV, indicators, strategies.
    Accepts optional start/end date range for custom analysis periods.
    """
    ticker_upper = ticker.strip().upper()

    end_date = end or date.today()
    start_date = start or (end_date - timedelta(days=365))

    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start must be on or before end")
    if (end_date - start_date).days > 3650:
        raise HTTPException(status_code=400, detail="date range cannot exceed 10 years")
    if (end_date - start_date).days < 30:
        raise HTTPException(status_code=400, detail="date range must be at least 30 days")

    cache_key = f"research_{ticker_upper}_{start_date.isoformat()}_{end_date.isoformat()}"

    cached = None if refresh else cache_get(RESEARCH_CACHE, cache_key)
    if cached is not None:
        return cached

    # Look up constituent for company name / sector
    constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)

    company_name = ticker_upper
    sector = ""
    yahoo_ticker = normalize_yahoo_ticker(ticker_upper)

    for c in constituents_list:
        if c.ticker.upper() == ticker_upper or c.yahooTicker.upper() == ticker_upper:
            company_name = c.companyName
            sector = c.sector
            yahoo_ticker = c.yahooTicker
            break

    try:
        result = await run_in_threadpool(
            compute_research, yahoo_ticker, company_name, sector,
            start_date=start_date, end_date=end_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    cache_set(RESEARCH_CACHE, cache_key, result)
    return result


@app.get("/api/movers.csv")
async def movers_csv(
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    refresh: bool = Query(False),
) -> Response:
    import csv
    import io

    start_date, end_date = _resolve_dates(start, end)

    payload = await movers(start=start_date, end=end_date, limit=5000, include_all=True, refresh=refresh)
    rows = payload.all or []

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Rank",
            "Ticker",
            "Company Name",
            "Sector",
            "Current Price",
            "Current Price Date",
            "Past Price",
            "Past Price Date",
            "% Change",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r.rank,
                r.ticker,
                r.companyName,
                r.sector,
                f"{r.currentPrice:.4f}",
                r.currentPriceDate.isoformat(),
                f"{r.pastPrice:.4f}",
                r.pastPriceDate.isoformat(),
                f"{r.pctChange:.4f}",
            ]
        )

    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="sp500-movers_{start_date.isoformat()}_{end_date.isoformat()}.csv"'
        },
    )
