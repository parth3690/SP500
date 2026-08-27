from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from dotenv import load_dotenv

# Load backend/.env before any service reads os.environ (keys stay server-side only).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool

from .models import (
    Constituent, MoversResponse, MoverRow, SectorSummaryRow,
    CrossoverRow, CrossoversResponse,
    OversoldRow, OversoldResponse,
    OverboughtRow, OverboughtResponse,
    WeeklyMaWatchRow, WeeklyMaWatchResponse,
    MultibaggerResponse,
    MarketConditionsFetchResponse,
    AlphaCandidatesResponse,
    AlphaWatchlistRequest,
    AgentBotRunRequest,
    AgentBotRunResponse,
)
from .services.cache import (
    MOVERS_CACHE, CROSSOVERS_CACHE, RESEARCH_CACHE, RSI_SCAN_CACHE, WEEKLY_MA_SCAN_CACHE,
    MOVE_FINDER_CACHE, MULTIBAGGER_CACHE, MARKET_CONDITIONS_CACHE,
    ALPHA_CACHE,
    cache_get, cache_set, price_cache_get, price_cache_set,
    clear_research_and_price_caches,
)
from .services.movers import compute_movers
from .services.crossovers import compute_crossovers
from .services.move_finder import find_runner_moves
from .services.research import compute_research
from .services.rsi_scan import (
    compute_rsi_scan,
    compute_rsi_scan_overbought,
    compute_rsi_scan_daily_oversold,
    compute_rsi_scan_daily_overbought,
)
from .services.weekly_ma_scan import compute_weekly_ma_watch
from .services.prices import fetch_close_prices
from .services.valuations import attach_pe_metrics, fetch_pe_metrics
from .services.sp500 import (
    get_sp500_constituents_cached,
    get_yahoo_tickers,
    normalize_user_ticker,
    normalize_yahoo_ticker,
)
from .services.multibagger import scan_ticker
from .services.market_conditions import fetch_all_market_conditions
from .services.alpha import ALPHA_CACHE_VERSION, alpha_universe_tickers, compute_alpha_candidates
from .services.agent_bot import run_agent_bot

DEFAULT_RANGE_DAYS = int(os.getenv("DEFAULT_RANGE_DAYS", "30"))
MAX_RANGE_DAYS = int(os.getenv("MAX_RANGE_DAYS", "366"))
PRELOAD_DASHBOARD = os.getenv("PRELOAD_DASHBOARD", "false").strip().lower() in ("1", "true", "yes")
_PRICE_FETCH_LOCK = Lock()


def _parse_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="S&P 500 Monthly Movers Analyzer API", version="0.1.0")

origins = _parse_origins()
app.add_middleware(GZipMiddleware, minimum_size=500)
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
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    if not refresh:
        cached = price_cache_get(start_iso, end_iso)
        if cached is not None:
            return cached

    with _PRICE_FETCH_LOCK:
        if not refresh:
            cached = price_cache_get(start_iso, end_iso)
            if cached is not None:
                return cached
        prices = fetch_close_prices(yahoo_tickers, start, end)
        coverage = _price_coverage(prices, yahoo_tickers, min_rows=2)
        if coverage["coveragePct"] >= 90.0:
            price_cache_set(start_iso, end_iso, prices)
        return prices


def _price_coverage(close_prices: Any, tickers: list[str], *, min_rows: int = 2) -> dict[str, Any]:
    """Describe usable price coverage without treating all-NaN columns as loaded."""
    requested = list(dict.fromkeys(tickers))
    columns = set(getattr(close_prices, "columns", []))
    available = [
        ticker
        for ticker in requested
        if ticker in columns and len(close_prices[ticker].dropna()) >= min_rows
    ]
    missing = [ticker for ticker in requested if ticker not in available]
    return {
        "requested": len(requested),
        "available": len(available),
        "coveragePct": round(len(available) / len(requested) * 100, 1) if requested else 0.0,
        "missingTickers": missing,
    }


def _require_price_coverage(
    close_prices: Any,
    tickers: list[str],
    *,
    minimum_pct: float = 90.0,
    min_rows: int = 2,
) -> dict[str, Any]:
    coverage = _price_coverage(close_prices, tickers, min_rows=min_rows)
    if coverage["coveragePct"] < minimum_pct:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Price data coverage is {coverage['coveragePct']}% "
                f"({coverage['available']}/{coverage['requested']}). "
                "The scan was withheld because the result would be incomplete. "
                "Please refresh after the market data provider recovers."
            ),
        )
    return coverage


def _research_payload_has_price_mismatch(payload: dict[str, Any]) -> bool:
    chart_close = payload.get("chartLastClose")
    if chart_close is None:
        return False
    candidates = [payload.get("currentPrice"), payload.get("previousClose")]
    try:
        chart = float(chart_close)
        prices = [float(value) for value in candidates if value is not None and float(value) > 0]
    except (TypeError, ValueError):
        return False
    if chart <= 0 or not prices:
        return False
    return min(abs(chart - price) / price for price in prices) > 0.35


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
        _require_price_coverage(close_prices, yahoo_tickers, minimum_pct=90.0, min_rows=2)

        # Pre-compute movers (default 30-day range)
        movers_start = end_date - timedelta(days=DEFAULT_RANGE_DAYS)
        movers_key = (movers_start.isoformat(), end_date.isoformat())
        if cache_get(MOVERS_CACHE, movers_key) is None:
            rows, sector_summary, meta = await run_in_threadpool(
                compute_movers, constituents_list, close_prices, movers_start, end_date
            )
            if meta.get("computed", 0) > 0:
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
            if c_meta.get("computed", 0) > 0:
                cache_set(CROSSOVERS_CACHE, crossover_key, {
                    "rows": c_rows, "meta": c_meta,
                    "asOf": datetime.now(timezone.utc),
                })

        # Pre-compute RSI oversold (below 30) and overbought (above 70)
        rsi_oversold_key = "rsi_oversold_30.0"
        if cache_get(RSI_SCAN_CACHE, rsi_oversold_key) is None:
            r_rows, r_meta = await run_in_threadpool(
                compute_rsi_scan, constituents_list, close_prices, rsi_threshold=30.0
            )
            if r_meta.get("computed", 0) > 0:
                cache_set(RSI_SCAN_CACHE, rsi_oversold_key, {
                    "rows": r_rows, "meta": r_meta,
                    "asOf": datetime.now(timezone.utc),
                })
        rsi_overbought_key = "rsi_overbought_70.0"
        if cache_get(RSI_SCAN_CACHE, rsi_overbought_key) is None:
            ob_rows, ob_meta = await run_in_threadpool(
                compute_rsi_scan_overbought, constituents_list, close_prices, rsi_threshold=70.0
            )
            if ob_meta.get("computed", 0) > 0:
                cache_set(RSI_SCAN_CACHE, rsi_overbought_key, {
                    "rows": ob_rows, "meta": ob_meta,
                    "asOf": datetime.now(timezone.utc),
                })

        # Pre-compute Daily RSI oversold (below 30) and overbought (above 70)
        daily_oversold_key = "rsi_daily_oversold_30.0"
        if cache_get(RSI_SCAN_CACHE, daily_oversold_key) is None:
            do_rows, do_meta = await run_in_threadpool(
                compute_rsi_scan_daily_oversold, constituents_list, close_prices, rsi_threshold=30.0
            )
            if do_meta.get("computed", 0) > 0:
                cache_set(RSI_SCAN_CACHE, daily_oversold_key, {
                    "rows": do_rows, "meta": do_meta,
                    "asOf": datetime.now(timezone.utc),
                })
        daily_overbought_key = "rsi_daily_overbought_70.0"
        if cache_get(RSI_SCAN_CACHE, daily_overbought_key) is None:
            dob_rows, dob_meta = await run_in_threadpool(
                compute_rsi_scan_daily_overbought, constituents_list, close_prices, rsi_threshold=70.0
            )
            if dob_meta.get("computed", 0) > 0:
                cache_set(RSI_SCAN_CACHE, daily_overbought_key, {
                    "rows": dob_rows, "meta": dob_meta,
                    "asOf": datetime.now(timezone.utc),
                })

    except Exception as e:
        # Non-fatal: first request will just compute on demand
        print(f"[preload] Background preload failed (non-fatal): {e}")


@app.on_event("startup")
async def startup_event() -> None:
    """Fresh Yahoo-backed research/prices after each process start (avoids stale cached quotes)."""
    clear_research_and_price_caches()
    if PRELOAD_DASHBOARD:
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


def _custom_alpha_constituents(
    tickers: list[str],
    reference: list[Constituent],
) -> list[Constituent]:
    by_display = {c.ticker.upper(): c for c in reference}
    by_yahoo = {c.yahooTicker.upper(): c for c in reference}
    out: list[Constituent] = []
    seen: set[str] = set()

    for raw in tickers:
        display = normalize_user_ticker(raw)
        if not display:
            continue
        yahoo = normalize_yahoo_ticker(display)
        key = yahoo.upper()
        if not key or key in seen:
            continue
        seen.add(key)

        existing = by_display.get(display) or by_yahoo.get(key)
        if existing is not None:
            out.append(existing)
            continue

        out.append(
            Constituent(
                ticker=display,
                yahooTicker=yahoo,
                companyName=display,
                sector="Custom Watchlist",
                subIndustry=None,
            )
        )

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
        _require_price_coverage(close_prices, yahoo_tickers, minimum_pct=90.0, min_rows=2)
        rows, sector_summary, meta = await run_in_threadpool(
            compute_movers, constituents_list, close_prices, start_date, end_date
        )

        cached = {
            "rows": rows,
            "sectorSummary": sector_summary,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        if meta.get("computed", 0) > 0:
            cache_set(MOVERS_CACHE, cache_key, cached)

    rows = cached["rows"]
    gainers = _ranked(rows, descending=True, limit=limit)
    losers = _ranked(rows, descending=False, limit=limit)

    all_rows = None
    if include_all:
        all_rows = _ranked(rows, descending=True, limit=len(rows))

    rows_for_pe = all_rows if all_rows is not None else [*gainers, *losers]
    pe_metrics = await run_in_threadpool(
        fetch_pe_metrics,
        list({str(r["ticker"]).upper() for r in rows_for_pe}),
    )
    attach_pe_metrics(gainers, pe_metrics)
    attach_pe_metrics(losers, pe_metrics)
    if all_rows is not None:
        attach_pe_metrics(all_rows, pe_metrics)

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
        _require_price_coverage(close_prices, yahoo_tickers, minimum_pct=90.0, min_rows=200)
        rows, meta = await run_in_threadpool(
            compute_crossovers, constituents_list, close_prices, threshold_pct=threshold
        )

        cached = {
            "rows": rows,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        if meta.get("computed", 0) > 0:
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


@app.get("/api/weekly-ma-watch", response_model=WeeklyMaWatchResponse)
async def weekly_ma_watch(
    length: int = Query(200, ge=20, le=300, description="Weekly SMA lookback length"),
    near_pct: float = Query(2.0, alias="nearPct", ge=0.0, le=10.0, description="Early-warning distance above the SMA"),
    refresh: bool = Query(False),
) -> WeeklyMaWatchResponse:
    """Scan the S&P 500 using the uploaded Pine watch's default SMA rules."""
    cache_key = f"weekly_ma_watch_sma_{length}_{near_pct}"
    cached = None if refresh else cache_get(WEEKLY_MA_SCAN_CACHE, cache_key)
    if cached is None:
        constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
        yahoo_tickers = get_yahoo_tickers(constituents_list)
        end_date = date.today()
        # Six years provides enough buffer for 200 valid weekly closes and holidays.
        start_date = end_date - timedelta(days=6 * 366)
        close_prices = await run_in_threadpool(
            _get_shared_price_data, yahoo_tickers, start_date, end_date, refresh=refresh
        )
        _require_price_coverage(close_prices, yahoo_tickers, minimum_pct=90.0, min_rows=2)
        rows, meta = await run_in_threadpool(
            compute_weekly_ma_watch,
            constituents_list,
            close_prices,
            ma_length=length,
            near_pct=near_pct,
        )
        cached = {
            "rows": rows,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        if meta.get("computed", 0) > 0:
            cache_set(WEEKLY_MA_SCAN_CACHE, cache_key, cached)

    return WeeklyMaWatchResponse(
        asOf=cached["asOf"],
        maLength=length,
        maType="SMA",
        nearPct=near_pct,
        stocks=[WeeklyMaWatchRow(**row) for row in cached["rows"]],
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
        _require_price_coverage(close_prices, yahoo_tickers, minimum_pct=90.0, min_rows=40)
        rows, meta = await run_in_threadpool(
            compute_rsi_scan, constituents_list, close_prices, rsi_threshold=threshold
        )

        cached = {
            "rows": rows,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        if meta.get("computed", 0) > 0:
            cache_set(RSI_SCAN_CACHE, cache_key, cached)

    return OversoldResponse(
        asOf=cached["asOf"],
        rsiThreshold=threshold,
        stocks=[OversoldRow(**r) for r in cached["rows"]],
        meta=cached["meta"],
    )


@app.get("/api/rsi-overbought", response_model=OverboughtResponse)
async def rsi_overbought(
    threshold: float = Query(70.0, ge=50.0, le=99.0, description="Weekly RSI threshold (stocks at or above this are returned)"),
    refresh: bool = Query(False),
) -> OverboughtResponse:
    """
    Returns S&P 500 stocks where the weekly (14-period) RSI is at or above
    the given threshold, highlighting overbought conditions.
    """
    cache_key = f"rsi_overbought_{threshold}"

    cached = None if refresh else cache_get(RSI_SCAN_CACHE, cache_key)
    if cached is None:
        constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
        yahoo_tickers = get_yahoo_tickers(constituents_list)

        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        close_prices = await run_in_threadpool(
            _get_shared_price_data, yahoo_tickers, start_date, end_date, refresh=refresh
        )
        _require_price_coverage(close_prices, yahoo_tickers, minimum_pct=90.0, min_rows=40)
        rows, meta = await run_in_threadpool(
            compute_rsi_scan_overbought, constituents_list, close_prices, rsi_threshold=threshold
        )

        cached = {
            "rows": rows,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        if meta.get("computed", 0) > 0:
            cache_set(RSI_SCAN_CACHE, cache_key, cached)

    return OverboughtResponse(
        asOf=cached["asOf"],
        rsiThreshold=threshold,
        stocks=[OverboughtRow(**r) for r in cached["rows"]],
        meta=cached["meta"],
    )


@app.get("/api/rsi-daily-oversold", response_model=OversoldResponse)
async def rsi_daily_oversold(
    threshold: float = Query(30.0, ge=1.0, le=50.0, description="Daily RSI threshold (stocks at or below this are returned)"),
    refresh: bool = Query(False),
) -> OversoldResponse:
    """
    Returns S&P 500 stocks where the daily (14-period) RSI is at or below
    the given threshold, highlighting oversold conditions.
    """
    cache_key = f"rsi_daily_oversold_{threshold}"

    cached = None if refresh else cache_get(RSI_SCAN_CACHE, cache_key)
    if cached is None:
        constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
        yahoo_tickers = get_yahoo_tickers(constituents_list)
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        close_prices = await run_in_threadpool(
            _get_shared_price_data, yahoo_tickers, start_date, end_date, refresh=refresh
        )
        _require_price_coverage(close_prices, yahoo_tickers, minimum_pct=90.0, min_rows=20)
        rows, meta = await run_in_threadpool(
            compute_rsi_scan_daily_oversold, constituents_list, close_prices, rsi_threshold=threshold
        )
        cached = {
            "rows": rows,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        if meta.get("computed", 0) > 0:
            cache_set(RSI_SCAN_CACHE, cache_key, cached)

    return OversoldResponse(
        asOf=cached["asOf"],
        rsiThreshold=threshold,
        stocks=[OversoldRow(**r) for r in cached["rows"]],
        meta=cached["meta"],
    )


@app.get("/api/rsi-daily-overbought", response_model=OverboughtResponse)
async def rsi_daily_overbought(
    threshold: float = Query(70.0, ge=50.0, le=99.0, description="Daily RSI threshold (stocks at or above this are returned)"),
    refresh: bool = Query(False),
) -> OverboughtResponse:
    """
    Returns S&P 500 stocks where the daily (14-period) RSI is at or above
    the given threshold, highlighting overbought conditions.
    """
    cache_key = f"rsi_daily_overbought_{threshold}"

    cached = None if refresh else cache_get(RSI_SCAN_CACHE, cache_key)
    if cached is None:
        constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
        yahoo_tickers = get_yahoo_tickers(constituents_list)
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        close_prices = await run_in_threadpool(
            _get_shared_price_data, yahoo_tickers, start_date, end_date, refresh=refresh
        )
        _require_price_coverage(close_prices, yahoo_tickers, minimum_pct=90.0, min_rows=20)
        rows, meta = await run_in_threadpool(
            compute_rsi_scan_daily_overbought, constituents_list, close_prices, rsi_threshold=threshold
        )
        cached = {
            "rows": rows,
            "meta": meta,
            "asOf": datetime.now(timezone.utc),
        }
        if meta.get("computed", 0) > 0:
            cache_set(RSI_SCAN_CACHE, cache_key, cached)

    return OverboughtResponse(
        asOf=cached["asOf"],
        rsiThreshold=threshold,
        stocks=[OverboughtRow(**r) for r in cached["rows"]],
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
    If FMP_API_KEY is set, merges a live quote from Financial Modeling Prep into header fields.
    """
    ticker_upper = normalize_user_ticker(ticker)
    if not ticker_upper:
        raise HTTPException(status_code=400, detail="A valid US ticker is required")

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
    if cached is not None and not _research_payload_has_price_mismatch(cached):
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
            "Trailing P/E",
            "Forward P/E",
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
                "" if r.trailingPE is None else f"{r.trailingPE:.4f}",
                "" if r.forwardPE is None else f"{r.forwardPE:.4f}",
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


@app.get("/api/move-finder")
async def move_finder(
    min_change_pct: float = Query(15.0, ge=0.1, le=200.0),
    min_volume: int = Query(200_000, ge=1, le=500_000_000),
    min_price: float = Query(1.0, ge=0.01, le=1000.0),
    max_price: float = Query(150.0, ge=0.01, le=10000.0),
    limit: int = Query(25, ge=1, le=100),
    refresh: bool = Query(False),
) -> dict[str, Any]:
    """
    Find explosive runner-type moves (AXTI/VCX-style) from live FMP gainers.
    """
    if min_price > max_price:
        raise HTTPException(status_code=400, detail="min_price must be <= max_price")

    cache_key = (
        "move_finder",
        round(min_change_pct, 2),
        min_volume,
        round(min_price, 2),
        round(max_price, 2),
        limit,
    )
    if not refresh:
        cached = cache_get(MOVE_FINDER_CACHE, cache_key)
        if cached is not None:
            return cached

    payload = await run_in_threadpool(
        find_runner_moves,
        min_change_pct=min_change_pct,
        min_volume=min_volume,
        min_price=min_price,
        max_price=max_price,
        limit=limit,
    )
    cache_set(MOVE_FINDER_CACHE, cache_key, payload)
    return payload


@app.get("/api/alpha-candidates", response_model=AlphaCandidatesResponse)
async def alpha_candidates(
    limit: int = Query(50, ge=5, le=150),
    min_score: float = Query(55.0, ge=0.0, le=100.0, alias="minScore"),
    sector: Optional[str] = Query(None),
    max_beta: Optional[float] = Query(None, ge=0.1, le=5.0, alias="maxBeta"),
    risk_mode: str = Query("balanced", pattern="^(balanced|aggressive|defensive)$", alias="riskMode"),
    regime: str = Query("auto", pattern="^(auto|risk_on|neutral|risk_off)$"),
    enrich_top: int = Query(20, ge=0, le=50, alias="enrichTop"),
    refresh: bool = Query(False),
) -> AlphaCandidatesResponse:
    """Rank S&P 500 candidates by lightweight alpha score, relative strength, risk, regime fit, and signal backtests."""
    sector_value = sector.strip() if sector and sector.strip() else None
    cache_key = (
        "alpha_candidates",
        ALPHA_CACHE_VERSION,
        limit,
        round(min_score, 2),
        sector_value,
        round(max_beta, 2) if max_beta is not None else None,
        risk_mode,
        regime,
        enrich_top,
    )
    if not refresh:
        cached = cache_get(ALPHA_CACHE, cache_key)
        if cached is not None:
            return AlphaCandidatesResponse(**cached)

    constituents_list = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
    end_date = date.today()
    start_date = end_date - timedelta(days=760)
    price_key = ("alpha_prices", start_date.isoformat(), end_date.isoformat())
    close_prices = None if refresh else cache_get(ALPHA_CACHE, price_key)
    fetched_prices = close_prices is None
    if close_prices is None:
        tickers = alpha_universe_tickers(constituents_list)
        close_prices = await run_in_threadpool(fetch_close_prices, tickers, start_date, end_date)

    coverage = _require_price_coverage(
        close_prices,
        [c.yahooTicker for c in constituents_list],
        minimum_pct=90.0,
        min_rows=220,
    )
    if "SPY" not in getattr(close_prices, "columns", []):
        raise HTTPException(
            status_code=503,
            detail="SPY benchmark history is unavailable, so alpha ranking was withheld.",
        )
    if fetched_prices:
        cache_set(ALPHA_CACHE, price_key, close_prices)

    payload = await run_in_threadpool(
        compute_alpha_candidates,
        constituents_list,
        close_prices,
        limit=limit,
        min_score=min_score,
        sector=sector_value,
        max_beta=max_beta,
        risk_mode=risk_mode,
        regime_override=regime,
        enrich_top=enrich_top,
    )
    payload["meta"] = {**payload["meta"], **coverage, "status": "complete"}
    if payload.get("meta", {}).get("status") == "complete":
        cache_set(ALPHA_CACHE, cache_key, payload)
    return AlphaCandidatesResponse(**payload)


@app.post("/api/alpha-watchlist", response_model=AlphaCandidatesResponse)
async def alpha_watchlist(request: AlphaWatchlistRequest) -> AlphaCandidatesResponse:
    """Rank a user-supplied watchlist of up to 100 tickers with the same alpha model."""
    constituents_reference = await run_in_threadpool(get_sp500_constituents_cached, refresh=False)
    custom_constituents = _custom_alpha_constituents(request.tickers, constituents_reference)
    if not custom_constituents:
        raise HTTPException(status_code=400, detail="At least one valid ticker is required")

    yahoo_tickers = [c.yahooTicker for c in custom_constituents]
    cache_key = (
        "alpha_watchlist",
        ALPHA_CACHE_VERSION,
        tuple(yahoo_tickers),
        request.limit,
        round(request.minScore, 2),
        round(request.maxBeta, 2) if request.maxBeta is not None else None,
        request.riskMode,
        request.regime,
        request.enrichTop,
    )
    if not request.refresh:
        cached = cache_get(ALPHA_CACHE, cache_key)
        if cached is not None:
            return AlphaCandidatesResponse(**cached)

    end_date = date.today()
    start_date = end_date - timedelta(days=760)
    price_tickers = alpha_universe_tickers(custom_constituents)
    price_key = ("alpha_watchlist_prices", tuple(price_tickers), start_date.isoformat(), end_date.isoformat())
    close_prices = None if request.refresh else cache_get(ALPHA_CACHE, price_key)
    fetched_prices = close_prices is None
    if close_prices is None:
        close_prices = await run_in_threadpool(fetch_close_prices, price_tickers, start_date, end_date)

    coverage = _price_coverage(
        close_prices,
        [c.yahooTicker for c in custom_constituents],
        min_rows=220,
    )
    if coverage["available"] == 0 or "SPY" not in getattr(close_prices, "columns", []):
        raise HTTPException(
            status_code=503,
            detail=(
                "Reliable stock and SPY price history is unavailable, so the watchlist "
                "ranking was withheld. Please refresh after the data provider recovers."
            ),
        )
    if fetched_prices and coverage["coveragePct"] >= 80.0:
        cache_set(ALPHA_CACHE, price_key, close_prices)

    payload = await run_in_threadpool(
        compute_alpha_candidates,
        custom_constituents,
        close_prices,
        limit=min(request.limit, len(custom_constituents)),
        min_score=request.minScore,
        sector=None,
        max_beta=request.maxBeta,
        risk_mode=request.riskMode,
        regime_override=request.regime,
        enrich_top=min(request.enrichTop, len(custom_constituents)),
    )
    payload["meta"] = {
        **payload["meta"],
        **coverage,
        "universe": "watchlist",
        "requestedTickers": len(request.tickers),
        "validTickers": len(custom_constituents),
        "status": "complete" if coverage["coveragePct"] == 100.0 else "partial",
        "warnings": [
            *payload["meta"].get("warnings", []),
            *(
                [
                    f"{coverage['available']} of {coverage['requested']} watchlist tickers have "
                    "enough history; missing tickers were excluded."
                ]
                if coverage["coveragePct"] < 100.0
                else []
            ),
        ],
    }
    if payload.get("meta", {}).get("status") == "complete":
        cache_set(ALPHA_CACHE, cache_key, payload)
    return AlphaCandidatesResponse(**payload)


@app.post("/api/agent-bot/run", response_model=AgentBotRunResponse)
async def agent_bot_run(request: AgentBotRunRequest) -> AgentBotRunResponse:
    """Run the autonomous agent bot over a watchlist or the full S&P 500."""
    payload = await run_in_threadpool(
        run_agent_bot,
        request.tickers,
        mode=request.mode,
        risk_mode=request.riskMode,
        regime=request.regime,
        top_n=request.topN,
        min_score=request.minScore,
        history=[h.model_dump() for h in request.history],
        refresh=request.refresh,
    )
    return AgentBotRunResponse(**payload)


@app.get("/api/multibagger/{ticker}", response_model=MultibaggerResponse)
async def multibagger_scan(
    ticker: str,
    deep: bool = Query(False, description="Use multi-year ROE / CAGR from financial statements"),
    refresh: bool = Query(False),
) -> MultibaggerResponse:
    """Evaluate one US ticker against the multibagger-style fundamental checklist."""
    display_ticker = normalize_user_ticker(ticker)
    if not display_ticker:
        raise HTTPException(status_code=400, detail="A valid US ticker is required")
    sym = normalize_yahoo_ticker(display_ticker)

    cache_key = ("multibagger", sym, deep)
    if not refresh:
        cached = cache_get(MULTIBAGGER_CACHE, cache_key)
        if cached is not None:
            return MultibaggerResponse(**cached)

    try:
        payload = await run_in_threadpool(scan_ticker, sym, deep=deep)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Scan failed: {e}") from e

    cache_set(MULTIBAGGER_CACHE, cache_key, payload)
    return MultibaggerResponse(**payload)


@app.get("/api/market-conditions/fetch", response_model=MarketConditionsFetchResponse)
async def market_conditions_fetch(refresh: bool = Query(False)) -> MarketConditionsFetchResponse:
    """Fetch live readings for all market peak signpost criteria (best effort)."""
    cache_key = "market_conditions_fetch"
    previous = cache_get(MARKET_CONDITIONS_CACHE, cache_key)
    if not refresh:
        cached = previous
        if cached is not None:
            return MarketConditionsFetchResponse(**cached)

    try:
        payload = await run_in_threadpool(fetch_all_market_conditions)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market conditions fetch failed: {e}") from e

    fetched_count = payload.get("meta", {}).get("fetchedCount", 0)
    if fetched_count > 0:
        cache_set(MARKET_CONDITIONS_CACHE, cache_key, payload)
    elif previous is not None:
        payload = {
            **previous,
            "meta": {
                **previous.get("meta", {}),
                "stale": True,
                "warnings": [
                    *previous.get("meta", {}).get("warnings", []),
                    "Refresh returned no usable data; showing the last successful snapshot.",
                ],
            },
        }
    return MarketConditionsFetchResponse(**payload)
