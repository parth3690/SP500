"""
NYSE SMID Agent - Institutional scanner for small-mid cap NYSE stocks.

Reuses the existing S&P 500 data pipeline (constituents, prices, FMP, alpha engine)
to scan individual stocks in the NYSE $100M-$2B universe.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ..models import Constituent
from .alpha import alpha_universe_tickers, compute_alpha_candidates
from .cache import INSTITUTIONAL_SCANNER_CACHE, cache_get, cache_set
from .institutional_scanner import (
    INSTITUTIONAL_SCANNER_VERSION,
    MIN_ALPHA_VS_BENCHMARK,
    MIN_BACKTEST_SAMPLE_SIZE,
    MIN_BACKTEST_WIN_RATE,
    MIN_CONFIDENCE_FOR_TAKE,
    _apply_trade_gate,
    _compute_confidence,
    _detect_convexity_alert,
    _run_simulation_validation,
    _run_walk_forward_backtest,
)
from .prices import fetch_close_prices
from .sp500 import SECTOR_ETFS, get_nyse_smid_constituents_cached, normalize_user_ticker


def run_nyse_smid_agent(
    tickers: list[str] | None = None,
    *,
    limit: int = 20,
    min_score: float = 65.0,
    risk_mode: str = "balanced",
    regime: str = "auto",
    refresh: bool = False,
) -> dict[str, Any]:
    """
    Run institutional scanner on NYSE $100M-$2B universe.
    
    Reuses the existing S&P 500 data stack:
    - Constituents from FMP (NYSE listings + market cap filter)
    - Prices from fetch_close_prices (shared Yahoo/FMP pipeline)
    - Alpha engine (compute_alpha_candidates)
    - Institutional gate (backtest + simulation + confidence)
    
    Parameters
    ----------
    tickers: Optional list of specific tickers to scan within the universe.
             If None, scans the entire NYSE SMID universe.
    limit: Maximum number of candidates to return
    min_score: Minimum alpha score threshold
    risk_mode: "balanced" | "aggressive" | "defensive"
    regime: "auto" | "risk_on" | "neutral" | "risk_off"
    refresh: Force refresh of cached data
    
    Returns
    -------
    dict with candidates, convexityAlerts, marketRegime, and meta
    """
    warnings: list[str] = []
    end_date = date.today()
    start_date = end_date - timedelta(days=760)

    # Load NYSE SMID constituents (uses FMP via existing helpers)
    try:
        all_constituents = get_nyse_smid_constituents_cached(refresh=refresh)
    except Exception as exc:
        return _empty_response(
            error="NYSE SMID universe could not be loaded. FMP_API_KEY may be required.",
            warnings=[str(exc)],
        )

    if not all_constituents:
        return _empty_response(
            error="NYSE SMID universe is empty. FMP_API_KEY required for NYSE listings.",
            warnings=["Set FMP_API_KEY in environment to enable NYSE SMID scanning."],
        )

    # Filter to specific tickers if provided
    if tickers:
        clean_tickers = [ticker for raw in tickers if (ticker := normalize_user_ticker(raw))]
        clean_tickers = list(dict.fromkeys(clean_tickers))
        
        if not clean_tickers:
            return _empty_response(
                error="No valid tickers provided.",
                warnings=["Provide valid NYSE ticker symbols."],
            )
        
        # Filter constituents to requested tickers
        ticker_set = set(clean_tickers)
        constituents_list = [c for c in all_constituents if c.ticker in ticker_set]
        
        if not constituents_list:
            return _empty_response(
                error="None of the provided tickers are in the NYSE SMID universe ($100M-$2B).",
                warnings=[f"Requested: {', '.join(clean_tickers)}"],
            )
    else:
        constituents_list = all_constituents

    # Fetch prices (reuses existing fetch_close_prices from S&P 500 stack)
    price_cache_key = ("nyse_smid_prices", start_date.isoformat(), end_date.isoformat())
    close_prices = None if refresh else cache_get(INSTITUTIONAL_SCANNER_CACHE, price_cache_key)
    
    if close_prices is None:
        price_tickers = alpha_universe_tickers(constituents_list)
        try:
            close_prices = fetch_close_prices(price_tickers, start_date, end_date)
        except Exception as exc:
            return _empty_response(
                error="Price data fetch failed.",
                warnings=[str(exc)],
            )
        
        if not close_prices.empty:
            cache_set(INSTITUTIONAL_SCANNER_CACHE, price_cache_key, close_prices)

    if close_prices.empty or "SPY" not in close_prices.columns:
        return _empty_response(
            error="Price data unavailable or SPY benchmark missing.",
            warnings=["Cannot run institutional scanner without price history."],
        )

    # Run alpha candidates (reuses existing alpha engine)
    try:
        alpha_result = compute_alpha_candidates(
            constituents_list,
            close_prices,
            limit=limit * 3,  # Get more candidates for institutional gate filtering
            min_score=min_score,
            sector=None,
            max_beta=None,
            risk_mode=risk_mode,
            regime_override=regime,
            enrich_top=0,
            include_lowest=0,
        )
    except Exception as exc:
        return _empty_response(
            error="Alpha candidate computation failed.",
            warnings=[str(exc)],
        )

    if not alpha_result.get("candidates"):
        return {
            "asOf": datetime.now(timezone.utc).isoformat(),
            "marketRegime": alpha_result.get("marketRegime", {}),
            "candidates": [],
            "convexityAlerts": [],
            "meta": {
                **alpha_result.get("meta", {}),
                "universe": "NYSE SMID ($100M-$2B)",
                "universeType": "nyse_smid",
                "status": "no_candidates",
                "scannerVersion": INSTITUTIONAL_SCANNER_VERSION,
                "warnings": warnings,
            },
        }

    # Apply institutional gate to each candidate
    spy = close_prices["SPY"].dropna() if "SPY" in close_prices.columns else pd.Series()
    scanner_candidates: list[dict[str, Any]] = []
    convexity_alerts: list[dict[str, Any]] = []

    for candidate in alpha_result["candidates"]:
        ticker = candidate["ticker"]
        yahoo_ticker = None
        
        for c in constituents_list:
            if c.ticker == ticker:
                yahoo_ticker = c.yahooTicker
                break
        
        if not yahoo_ticker or yahoo_ticker not in close_prices.columns:
            continue

        stock = close_prices[yahoo_ticker].dropna()
        sector_etf = None
        for c in constituents_list:
            if c.ticker == ticker:
                sector_etf = SECTOR_ETFS.get(c.sector, "SPY")
                break
        
        sector_series = (
            close_prices[sector_etf].dropna()
            if sector_etf and sector_etf in close_prices.columns
            else spy
        )

        # Run institutional gate (reuses existing functions)
        backtest = _run_walk_forward_backtest(
            stock,
            spy,
            sector_series,
            risk_mode=risk_mode,
            regime=regime,
        )

        simulation = _run_simulation_validation(
            stock,
            spy,
            sector_series,
            backtest,
            risk_mode=risk_mode,
            regime=regime,
        )

        confidence_data = _compute_confidence(
            backtest,
            simulation,
            candidate["alphaScore"],
            candidate["riskScore"],
        )

        gate = _apply_trade_gate(confidence_data, backtest, simulation)

        # Convexity alert (disabled unless real options data)
        convexity = _detect_convexity_alert(
            ticker,
            candidate["currentPrice"],
            candidate["volatility20d"],
            candidate["alphaScore"],
            candidate["expectedReturn20d"],
        )
        
        if convexity:
            convexity_alerts.append(convexity)

        scanner_candidates.append({
            **candidate,
            "backtest": backtest,
            "simulation": simulation,
            "confidence": confidence_data,
            "tradeGate": gate,
            "convexityAlert": convexity,
        })

    # Sort by confidence then TAKE decision
    scanner_candidates.sort(
        key=lambda x: (x["tradeGate"]["decision"] == "TAKE", x["confidence"]["confidence"]),
        reverse=True,
    )

    # Re-rank and limit
    for idx, candidate in enumerate(scanner_candidates[:limit], start=1):
        candidate["rank"] = idx

    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "marketRegime": alpha_result["marketRegime"],
        "candidates": scanner_candidates[:limit],
        "convexityAlerts": convexity_alerts,
        "meta": {
            **alpha_result["meta"],
            "universe": "NYSE SMID ($100M-$2B)",
            "universeType": "nyse_smid",
            "scannerVersion": INSTITUTIONAL_SCANNER_VERSION,
            "status": "complete",
            "backtestHorizon": "20-day walk-forward",
            "simulationScenarios": ["bull", "base", "bear", "high_vol"],
            "tradeGate": {
                "minConfidence": MIN_CONFIDENCE_FOR_TAKE,
                "minWinRate": MIN_BACKTEST_WIN_RATE,
                "minSampleSize": MIN_BACKTEST_SAMPLE_SIZE,
                "minAlphaVsBenchmark": MIN_ALPHA_VS_BENCHMARK,
            },
            "convexityAlert": {
                "minProbability": 10.0,
                "minReturn": "100x",
            },
            "warnings": warnings,
        },
    }


def _empty_response(
    *,
    error: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Return empty response with error/warnings."""
    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "marketRegime": {
            "state": "unknown",
            "spyTrend": "unknown",
            "spyDrawdownPct": None,
            "effectiveState": "unknown",
            "riskMode": "balanced",
        },
        "candidates": [],
        "convexityAlerts": [],
        "meta": {
            "universe": "NYSE SMID ($100M-$2B)",
            "universeType": "nyse_smid",
            "status": "error",
            "error": error,
            "warnings": warnings or [],
            "scannerVersion": INSTITUTIONAL_SCANNER_VERSION,
            "total": 0,
            "eligible": 0,
            "computed": 0,
            "returned": 0,
        },
    }
