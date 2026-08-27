from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from ..models import Constituent
from .alpha import (
    alpha_universe_tickers,
    apply_alpha_enrichment,
    catalyst_scores_from_info,
    compute_alpha_candidates,
)
from .cache import ALPHA_CACHE, MARKET_CONDITIONS_CACHE, cache_get, cache_set
from .market_conditions import fetch_all_market_conditions
from .prices import fetch_close_prices
from .sp500 import (
    get_sp500_constituents_cached,
    normalize_user_ticker,
    normalize_yahoo_ticker,
)


FORWARD_HORIZONS = (("1W", 7), ("1M", 30), ("3M", 90), ("6M", 180))


def run_agent_bot(
    tickers: list[str],
    *,
    mode: str = "watchlist",
    risk_mode: str = "balanced",
    regime: str = "auto",
    top_n: int = 10,
    min_score: float = 55.0,
    history: list[dict[str, Any]] | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """
    Run the autonomous agent bot over a ticker universe.

    Parameters
    ----------
    tickers: explicit list of display tickers (used when mode == "watchlist")
    mode: "sp500" or "watchlist"
    risk_mode: "balanced" | "aggressive" | "defensive"
    regime: "auto" | "risk_on" | "neutral" | "risk_off"
    top_n: number of top recommendations to surface
    min_score: minimum alpha score for a candidate to be recommended
    history: optional list of previous agent recommendations for outcome tracking

    Returns
    -------
    dict with briefing, recommendations, activeTracking, alerts, outcomes, catalysts, and meta
    """
    history = history or []
    warnings: list[str] = []
    end_date = date.today()
    start_date = end_date - timedelta(days=760)

    clean_tickers = [ticker for raw in tickers if (ticker := normalize_user_ticker(raw))]
    clean_tickers = list(dict.fromkeys(clean_tickers))
    if mode == "watchlist" and not clean_tickers:
        return _empty_response(
            mode,
            error="At least one valid ticker is required for watchlist mode.",
        )

    if mode == "sp500":
        try:
            constituents_list = get_sp500_constituents_cached(refresh=refresh)
        except Exception as exc:
            return _empty_response(
                mode,
                error="The S&P 500 universe could not be loaded.",
                warnings=[str(exc)],
            )
    else:
        try:
            reference = get_sp500_constituents_cached(refresh=False)
        except Exception as exc:
            reference = []
            warnings.append(f"S&P 500 company metadata unavailable: {exc}")
        constituents_list = _custom_constituents(clean_tickers, reference)

    if not constituents_list:
        return _empty_response(mode, error="No valid tickers or constituents.", warnings=warnings)

    market_conditions = _market_conditions_summary(refresh=refresh)

    price_tickers = alpha_universe_tickers(constituents_list)
    history_tickers = [
        normalize_yahoo_ticker(str(rec.get("ticker", "")))
        for rec in history
        if str(rec.get("action", "")).upper() in ("BUY", "SELL")
        and normalize_user_ticker(str(rec.get("ticker", "")))
    ]
    price_tickers = list(dict.fromkeys([*price_tickers, *history_tickers]))
    minimum_coverage = 90.0 if mode == "sp500" else 80.0
    close_prices = _cached_close_prices(
        price_tickers,
        start_date,
        end_date,
        refresh=refresh,
        minimum_coverage=minimum_coverage,
    )

    if close_prices.empty or "SPY" not in close_prices.columns:
        return _empty_response(
            mode,
            market_conditions=market_conditions,
            error="Reliable price history is unavailable, so the agent abstained.",
            warnings=warnings,
            meta={"requestedTickers": len(constituents_list)},
        )

    coverage = _price_history_coverage(
        close_prices,
        [c.yahooTicker for c in constituents_list],
        min_rows=220,
    )
    if coverage["coveragePct"] < minimum_coverage:
        return _empty_response(
            mode,
            market_conditions=market_conditions,
            error=(
                f"Only {coverage['coveragePct']}% of the requested universe has enough "
                "price history, so the agent abstained instead of using a partial scan."
            ),
            warnings=warnings,
            meta={
                "requestedTickers": coverage["requested"],
                "availableTickers": coverage["available"],
                "priceCoveragePct": coverage["coveragePct"],
                "missingTickers": coverage["missingTickers"],
            },
        )

    # Regime-aware floor adjustments
    adjusted_min_score, adjusted_max_beta = _regime_aware_filters(
        min_score, risk_mode, market_conditions.get("riskLevel", "Normal")
    )

    payload = compute_alpha_candidates(
        constituents_list,
        close_prices,
        limit=max(top_n * 3, 30),
        min_score=0.0,
        sector=None,
        max_beta=adjusted_max_beta,
        risk_mode=risk_mode,
        regime_override=regime,
        enrich_top=0,
        include_lowest=max(top_n * 3, 30),
    )

    candidate_pool = payload.get("candidates", [])
    candidates = _select_agent_candidates(candidate_pool, top_n, adjusted_min_score)

    _enrich_with_catalysts(candidates)
    candidates = _select_agent_candidates(candidates, top_n, adjusted_min_score)
    for rank, c in enumerate(candidates, start=1):
        c["agentRank"] = rank
        c["whyNow"] = _why_now(c)

    recommendations = [_recommendation_from_candidate(c) for c in candidates]
    active_tracking = _tracking_from_history(history, close_prices, candidates, constituents_list)

    alerts = _generate_alerts(candidates)

    # Forward outcomes for prior history
    outcomes = _compute_outcomes(history, close_prices)

    # Forward performance journal
    forward_journal = _forward_journal(history, close_prices)

    briefing = _generate_briefing(
        payload.get("marketRegime", {}),
        market_conditions,
        recommendations,
        active_tracking,
        alerts,
        outcomes,
    )

    catalysts = {c["ticker"]: c.get("catalystData", {}) for c in candidates}
    status = "ok"
    if market_conditions.get("riskLevel") == "Unknown" or warnings:
        status = "degraded"

    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "briefing": briefing,
        "recommendations": recommendations,
        "activeTracking": active_tracking,
        "alerts": alerts,
        "outcomes": outcomes,
        "forwardJournal": forward_journal,
        "catalysts": catalysts,
        "meta": {
            **payload.get("meta", {}),
            "mode": mode,
            "riskMode": risk_mode,
            "regime": regime,
            "topN": top_n,
            "minScore": min_score,
            "adjustedMinScore": adjusted_min_score,
            "adjustedMaxBeta": adjusted_max_beta,
            "marketConditions": market_conditions,
            "status": status,
            "priceCoveragePct": coverage["coveragePct"],
            "availableTickers": coverage["available"],
            "missingTickers": coverage["missingTickers"],
            "warnings": [*payload.get("meta", {}).get("warnings", []), *warnings],
        },
    }


def _cached_close_prices(
    tickers: list[str],
    start_date: date,
    end_date: date,
    *,
    refresh: bool,
    minimum_coverage: float,
) -> pd.DataFrame:
    cache_key = (
        "agent_prices",
        tuple(tickers),
        start_date.isoformat(),
        end_date.isoformat(),
    )
    if not refresh:
        cached = cache_get(ALPHA_CACHE, cache_key)
        if cached is not None:
            return cached
    prices = fetch_close_prices(tickers, start_date, end_date)
    if not prices.empty and float(prices.attrs.get("coveragePct", 0.0)) >= minimum_coverage:
        cache_set(ALPHA_CACHE, cache_key, prices)
    return prices


def _price_history_coverage(
    close_prices: pd.DataFrame,
    tickers: list[str],
    *,
    min_rows: int,
) -> dict[str, Any]:
    requested = list(dict.fromkeys(tickers))
    available = [
        ticker
        for ticker in requested
        if ticker in close_prices.columns and len(close_prices[ticker].dropna()) >= min_rows
    ]
    return {
        "requested": len(requested),
        "available": len(available),
        "coveragePct": round(len(available) / len(requested) * 100, 1) if requested else 0.0,
        "missingTickers": [ticker for ticker in requested if ticker not in available],
    }


def _market_conditions_summary(*, refresh: bool = False) -> dict[str, Any]:
    try:
        cache_key = "market_conditions_fetch"
        data = None if refresh else cache_get(MARKET_CONDITIONS_CACHE, cache_key)
        if data is None:
            data = fetch_all_market_conditions()
            if data.get("meta", {}).get("fetchedCount", 0) > 0:
                cache_set(MARKET_CONDITIONS_CACHE, cache_key, data)
        meta = data.get("meta", {})
        return {
            "riskLevel": meta.get("riskLevel", "Unknown"),
            "coveragePct": meta.get("coveragePct", 0),
            "triggeredCount": meta.get("triggeredCount", 0),
            "confidence": meta.get("confidence", "Insufficient"),
            "asOf": data.get("asOf"),
            "warnings": meta.get("warnings", []),
        }
    except Exception as exc:
        return {
            "riskLevel": "Unknown",
            "coveragePct": 0,
            "triggeredCount": 0,
            "confidence": "Insufficient",
            "asOf": None,
            "warnings": [str(exc)],
        }


def _regime_aware_filters(
    min_score: float,
    risk_mode: str,
    risk_level: str,
) -> tuple[float, Optional[float]]:
    """Raise score floor and cap beta when macro risk is elevated."""
    adjusted_min = min_score
    max_beta: Optional[float] = None

    if risk_level in ("Elevated", "Extreme"):
        adjusted_min = min(80.0, min_score + 10.0)
        max_beta = 1.1 if risk_mode != "aggressive" else 1.3
    elif risk_level == "Watch":
        adjusted_min = min(75.0, min_score + 5.0)
        max_beta = 1.25 if risk_mode != "aggressive" else 1.5

    if risk_mode == "defensive":
        adjusted_min = min(85.0, adjusted_min + 5.0)
        max_beta = min(max_beta or 1.5, 1.0)
    elif risk_mode == "aggressive":
        adjusted_min = max(0.0, adjusted_min - 10.0)

    return round(adjusted_min, 1), max_beta


def _custom_constituents(tickers: list[str], reference: list[Constituent]) -> list[Constituent]:
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


def _empty_forward_journal() -> dict[str, Any]:
    return {
        "entries": [],
        "aggregates": {
            label: {"count": 0, "avgReturn": None} for label, _ in FORWARD_HORIZONS
        },
    }


def _empty_briefing(risk_level: str = "Unknown") -> dict[str, Any]:
    return {
        "summary": "The agent abstained because there is not enough reliable data.",
        "regime": "unknown",
        "riskLevel": risk_level,
        "topBuy": None,
        "topSell": None,
        "topWatch": None,
        "topAvoid": None,
        "counts": {"buy": 0, "sell": 0, "watch": 0, "avoid": 0, "alerts": 0, "outcomes": 0},
    }


def _empty_response(
    mode: str,
    *,
    error: str,
    market_conditions: Optional[dict[str, Any]] = None,
    warnings: Optional[list[str]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    conditions = market_conditions or {
        "riskLevel": "Unknown",
        "coveragePct": 0,
        "triggeredCount": 0,
        "confidence": "Insufficient",
        "asOf": None,
        "warnings": [],
    }
    briefing = _empty_briefing(conditions.get("riskLevel", "Unknown"))
    briefing["summary"] = error
    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "briefing": briefing,
        "recommendations": [],
        "activeTracking": [],
        "alerts": [],
        "outcomes": [],
        "forwardJournal": _empty_forward_journal(),
        "catalysts": {},
        "meta": {
            "status": "insufficient_data",
            "error": error,
            "warnings": warnings or [],
            "marketConditions": conditions,
            **(meta or {}),
        },
    }


def _select_agent_candidates(
    candidates: list[dict[str, Any]],
    top_n: int,
    min_score: float,
) -> list[dict[str, Any]]:
    def confidence(row: dict[str, Any]) -> float:
        return float(row.get("tradePlan", {}).get("confidence", 0.0))

    buys = sorted(
        (
            row
            for row in candidates
            if row.get("tradePlan", {}).get("action") == "BUY"
            and float(row.get("alphaScore", 0.0)) >= min_score
        ),
        key=lambda row: (confidence(row), float(row.get("riskAdjustedScore", 0.0))),
        reverse=True,
    )
    sells = sorted(
        (row for row in candidates if row.get("tradePlan", {}).get("action") == "SELL"),
        key=lambda row: (confidence(row), -float(row.get("alphaScore", 100.0))),
        reverse=True,
    )
    actionable = sorted([*buys, *sells], key=confidence, reverse=True)
    selected = actionable[:top_n]

    if top_n >= 2:
        for required in (buys[:1], sells[:1]):
            if not required or any(row is required[0] for row in selected):
                continue
            if len(selected) >= top_n:
                selected[-1] = required[0]
            else:
                selected.append(required[0])

    selected_ids = {id(row) for row in selected}
    watches = sorted(
        (
            row
            for row in candidates
            if row.get("tradePlan", {}).get("action") == "WATCH"
            and float(row.get("alphaScore", 0.0)) >= max(50.0, min_score - 10.0)
        ),
        key=confidence,
        reverse=True,
    )
    avoids = sorted(
        (row for row in candidates if row.get("tradePlan", {}).get("action") == "AVOID"),
        key=lambda row: float(row.get("alphaScore", 100.0)),
    )
    for row in [*watches, *avoids]:
        if len(selected) >= top_n:
            break
        if id(row) not in selected_ids:
            selected.append(row)
            selected_ids.add(id(row))
    return selected[:top_n]


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        out = float(v)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _enrich_with_catalysts(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        return
    with ThreadPoolExecutor(max_workers=min(6, len(candidates))) as pool:
        futures = {
            pool.submit(_fetch_catalyst_data, c["ticker"], float(c["currentPrice"])): c
            for c in candidates
        }
        for future in as_completed(futures):
            c = futures[future]
            try:
                data = future.result()
            except Exception:
                data = {"available": False, "revisionNotes": ["Catalyst data unavailable."]}
            alpha_enrichment = data.pop("_alphaEnrichment", None)
            c["catalystData"] = data
            if alpha_enrichment is not None:
                apply_alpha_enrichment(c, alpha_enrichment)


def _fetch_catalyst_data(ticker: str, current_price: float) -> dict[str, Any]:
    try:
        import yfinance as yf

        info = yf.Ticker(normalize_yahoo_ticker(ticker)).info or {}
    except Exception:
        return {"available": False, "revisionNotes": ["Catalyst data unavailable."]}

    earnings_date = (
        info.get("earningsTimestamp")
        or info.get("earningsDate")
        or info.get("earningsTimestampStart")
    )
    if isinstance(earnings_date, list):
        earnings_date = earnings_date[0]

    target = _safe_float(info.get("targetMeanPrice"))
    recommendation = str(info.get("recommendationKey") or "").replace("_", " ").title()
    analyst_count = _safe_float(info.get("numberOfAnalystOpinions"))
    revenue_growth = _safe_float(info.get("revenueGrowth"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))
    eps_growth = _safe_float(info.get("earningsQuarterlyGrowth"))
    dividend_yield = _safe_float(info.get("dividendYield"))
    ex_div = _format_earnings_date(info.get("exDividendDate"))

    revisions: list[str] = []
    if revenue_growth is not None and revenue_growth > 0.05:
        revisions.append(f"Revenue growth {revenue_growth * 100:.1f}%")
    if earnings_growth is not None and earnings_growth > 0.05:
        revisions.append(f"Earnings growth {earnings_growth * 100:.1f}%")
    if eps_growth is not None and eps_growth > 0.05:
        revisions.append(f"Quarterly EPS growth {eps_growth * 100:.1f}%")

    return {
        "available": bool(info),
        "earningsDate": _format_earnings_date(earnings_date),
        "exDividendDate": ex_div,
        "targetMeanPrice": target,
        "analystRecommendation": recommendation,
        "analystCount": int(analyst_count) if analyst_count is not None else None,
        "revenueGrowth": revenue_growth,
        "earningsGrowth": earnings_growth,
        "epsGrowth": eps_growth,
        "dividendYield": dividend_yield,
        "revisionNotes": revisions or ["No strong revision signal."],
        "_alphaEnrichment": catalyst_scores_from_info(info, current_price),
    }


def _format_earnings_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return value[:10]
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]
    if isinstance(value, (int, float)):
        try:
            timestamp = float(value)
            if timestamp > 10_000_000_000:
                timestamp /= 1000.0
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    return str(value)[:10]


def _why_now(c: dict[str, Any]) -> str:
    reasons: list[str] = []

    for signal in c.get("signals", []):
        if signal["state"] == "bullish":
            reasons.append(f"{signal['label'].lower()} is bullish ({signal['detail']})")
        elif signal["state"] == "bearish":
            reasons.append(f"{signal['label'].lower()} is bearish ({signal['detail']})")

    catalyst = c.get("catalystData", {})
    if catalyst.get("revisionNotes") and "No strong" not in catalyst["revisionNotes"][0]:
        reasons.append(catalyst["revisionNotes"][0])
    if catalyst.get("earningsDate"):
        reasons.append(f"earnings on {catalyst['earningsDate']}")

    if not reasons:
        return "No standout technical or catalyst signal; monitor."

    top = reasons[:3]
    action = c.get("tradePlan", {}).get("action")
    lead = {
        "BUY": "has a bullish setup because ",
        "SELL": "has a bearish setup because ",
        "AVOID": "should be avoided because ",
    }.get(action, "is worth monitoring because ")
    return f"{c['ticker']} {lead}" + "; ".join(top) + "."


def _recommendation_from_candidate(c: dict[str, Any]) -> dict[str, Any]:
    plan = c["tradePlan"]
    return {
        "rank": c.get("agentRank", c["rank"]),
        "ticker": c["ticker"],
        "companyName": c["companyName"],
        "sector": c["sector"],
        "action": plan["action"],
        "confidence": plan["confidence"],
        "alphaScore": c["alphaScore"],
        "riskAdjustedScore": c["riskAdjustedScore"],
        "expectedReturn20d": c["expectedReturn20d"],
        "horizon": plan["horizon"],
        "entry": plan["entry"],
        "buyBelow": plan.get("buyBelow"),
        "sellAbove": plan.get("sellAbove"),
        "stop": plan.get("stop"),
        "target1": plan.get("target1"),
        "target2": plan.get("target2"),
        "riskReward": plan.get("riskReward"),
        "optionStrategy": plan.get("optionStrategy"),
        "optionDirection": plan.get("optionDirection"),
        "optionStrike": plan.get("optionStrike"),
        "optionExpiry": plan.get("optionExpiry"),
        "optionRationale": plan.get("optionRationale"),
        "optionCategory": plan.get("optionCategory"),
        "optionDte": plan.get("optionDte"),
        "optionIvProxy": plan.get("optionIvProxy"),
        "optionIvGate": plan.get("optionIvGate"),
        "optionRules": plan.get("optionRules", []),
        "rationale": plan["rationale"],
        "signals": c["signals"],
        "whyNow": c.get("whyNow", ""),
        "backtests": c.get("backtests", []),
        "catalyst": c.get("catalystData", {}),
    }


def _tracking_from_history(
    history: list[dict[str, Any]],
    close_prices: pd.DataFrame,
    candidates: list[dict[str, Any]],
    constituents: list[Constituent],
) -> list[dict[str, Any]]:
    candidate_map: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_map[normalize_yahoo_ticker(candidate["ticker"])] = candidate
    company_map = {
        c.yahooTicker: c for c in constituents
    }
    tracked: list[dict[str, Any]] = []
    for rec in history:
        action = str(rec.get("action", "")).upper()
        if action not in ("BUY", "SELL") or rec.get("closed"):
            continue
        display = normalize_user_ticker(str(rec.get("ticker", "")))
        symbol = normalize_yahoo_ticker(display)
        entry = _safe_float(rec.get("entryPrice"))
        current, price_date = _price_with_date_on_or_before(close_prices, symbol, date.today())
        if not display or entry is None or current is None or price_date is None:
            continue
        candidate = candidate_map.get(symbol)
        constituent = company_map.get(symbol)
        plan = candidate.get("tradePlan", {}) if candidate else {}
        tracked.append(
            {
                "id": rec.get("id"),
                "ticker": display,
                "companyName": (
                    candidate.get("companyName")
                    if candidate
                    else constituent.companyName
                    if constituent
                    else display
                ),
                "action": action,
                "entry": round(entry, 2),
                "currentPrice": round(current, 2),
                "priceDate": price_date,
                "stop": plan.get("stop"),
                "target1": plan.get("target1"),
                "target2": plan.get("target2"),
                "unrealizedReturnPct": _unrealized(current, entry, action),
                "alphaScore": float(candidate.get("alphaScore", 0.0)) if candidate else 0.0,
                "whyNow": (
                    candidate.get("whyNow", "")
                    if candidate
                    else "Open saved recommendation; it is outside the current top-ranked set."
                ),
            }
        )
    return tracked


def _unrealized(current: float, entry: float, action: str = "BUY") -> Optional[float]:
    try:
        if entry is None or entry == 0:
            return None
        raw = (current / entry - 1.0) * 100.0
        return round(raw if action == "BUY" else -raw, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _generate_alerts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for c in candidates:
        price = float(c["currentPrice"])
        plan = c["tradePlan"]
        action = plan.get("action")
        stop = plan.get("stop")
        target1 = plan.get("target1")
        target2 = plan.get("target2")

        stop_near = (
            action == "BUY" and stop is not None and price <= float(stop) * 1.01
        ) or (
            action == "SELL" and stop is not None and price >= float(stop) * 0.99
        )
        target1_near = (
            action == "BUY" and target1 is not None and price >= float(target1) * 0.98
        ) or (
            action == "SELL" and target1 is not None and price <= float(target1) * 1.02
        )
        target2_near = (
            action == "BUY" and target2 is not None and price >= float(target2) * 0.98
        ) or (
            action == "SELL" and target2 is not None and price <= float(target2) * 1.02
        )

        if stop_near:
            alerts.append({
                "ticker": c["ticker"],
                "type": "stop proximity",
                "severity": "high",
                "message": f"{c['ticker']} is near its suggested stop at {stop:.2f}.",
            })
        if target1_near:
            alerts.append({
                "ticker": c["ticker"],
                "type": "target hit",
                "severity": "medium",
                "message": f"{c['ticker']} is near its first target at {target1:.2f}.",
            })
        if target2_near:
            alerts.append({
                "ticker": c["ticker"],
                "type": "target 2 hit",
                "severity": "medium",
                "message": f"{c['ticker']} is near its second target at {target2:.2f}.",
            })

        catalyst = c.get("catalystData", {})
        earnings_date = _parse_date(catalyst.get("earningsDate"))
        if earnings_date is not None and date.today() <= earnings_date <= date.today() + timedelta(days=45):
            alerts.append({
                "ticker": c["ticker"],
                "type": "earnings upcoming",
                "severity": "low",
                "message": f"{c['ticker']} has earnings around {catalyst['earningsDate']}.",
            })

        score = float(c["alphaScore"])
        if action == "BUY" and score >= 75:
            alerts.append({
                "ticker": c["ticker"],
                "type": "strong score",
                "severity": "low",
                "message": f"{c['ticker']} alpha score is {score:.1f} — among the strongest today.",
            })
        elif action == "SELL" and score <= 35:
            alerts.append({
                "ticker": c["ticker"],
                "type": "strong bearish score",
                "severity": "low",
                "message": f"{c['ticker']} alpha score is {score:.1f}, a strong bearish reading today.",
            })
    return alerts


def _compute_outcomes(history: list[dict[str, Any]], close_prices: pd.DataFrame) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for rec in history:
        ticker = normalize_user_ticker(str(rec.get("ticker", "")))
        symbol = normalize_yahoo_ticker(ticker)
        entry = _safe_float(rec.get("entryPrice"))
        action = str(rec.get("action", "")).upper()
        if action not in ("BUY", "SELL"):
            continue
        current = None
        if rec.get("closed"):
            current = _safe_float(rec.get("exitPrice"))
        if current is None:
            current = _price_on_or_before(close_prices, symbol, date.today())
        if not ticker or current is None or entry is None or entry == 0:
            continue
        raw_return = (current / entry - 1.0) * 100.0
        signed_return = raw_return if action == "BUY" else -raw_return
        outcomes.append({
            "id": rec.get("id"),
            "ticker": ticker,
            "action": action,
            "entryPrice": round(entry, 2),
            "currentPrice": round(current, 2),
            "returnPct": round(signed_return, 2),
            "recommendedAt": _iso_datetime(rec.get("recommendedAt")),
            "status": "closed" if rec.get("closed") else "open",
        })
    return outcomes


def _forward_journal(history: list[dict[str, Any]], close_prices: pd.DataFrame) -> dict[str, Any]:
    """Compute forward returns at 1W, 1M, 3M, 6M for each historical recommendation."""
    entries: list[dict[str, Any]] = []
    for rec in history:
        ticker = normalize_user_ticker(str(rec.get("ticker", "")))
        symbol = normalize_yahoo_ticker(ticker)
        entry_price = _safe_float(rec.get("entryPrice"))
        action = str(rec.get("action", "")).upper()
        recommended_at = rec.get("recommendedAt")
        if action not in ("BUY", "SELL") or not ticker or not entry_price or not recommended_at:
            continue
        rec_date = _parse_date(recommended_at)
        if rec_date is None:
            continue

        forward_returns: dict[str, Any] = {}
        latest_date = _latest_price_date(close_prices, symbol)
        for label, days in FORWARD_HORIZONS:
            target_date = rec_date + timedelta(days=days)
            fwd_price = (
                _price_on_or_before(close_prices, symbol, target_date)
                if latest_date is not None and target_date <= latest_date
                else None
            )
            if fwd_price is not None:
                raw = (fwd_price / entry_price - 1.0) * 100.0
                forward_returns[label] = round(raw if action == "BUY" else -raw, 2)
            else:
                forward_returns[label] = None

        entries.append({
            "id": rec.get("id"),
            "ticker": ticker,
            "action": action,
            "entryPrice": round(entry_price, 2),
            "recommendedAt": _iso_datetime(recommended_at),
            "closed": rec.get("closed", False),
            "forwardReturns": forward_returns,
        })

    # Aggregate averages per horizon across closed / all recommendations
    aggregates: dict[str, dict[str, Any]] = {}
    for label, _ in FORWARD_HORIZONS:
        vals = [e["forwardReturns"][label] for e in entries if e["forwardReturns"].get(label) is not None]
        aggregates[label] = {
            "count": len(vals),
            "avgReturn": round(sum(vals) / len(vals), 2) if vals else None,
        }

    return {"entries": entries, "aggregates": aggregates}


def _price_on_or_before(close_prices: pd.DataFrame, ticker: str, target_date: date) -> Optional[float]:
    price, _ = _price_with_date_on_or_before(close_prices, ticker, target_date)
    return price


def _price_with_date_on_or_before(
    close_prices: pd.DataFrame,
    ticker: str,
    target_date: date,
) -> tuple[Optional[float], Optional[date]]:
    symbol = normalize_yahoo_ticker(ticker)
    if symbol not in close_prices.columns:
        return None, None
    series = close_prices[symbol].dropna().sort_index()
    for position in range(len(series) - 1, -1, -1):
        raw_date = series.index[position]
        price_date = raw_date.date() if hasattr(raw_date, "date") else raw_date
        if isinstance(price_date, datetime):
            price_date = price_date.date()
        if isinstance(price_date, date) and price_date <= target_date:
            return _safe_float(series.iloc[position]), price_date
    return None, None


def _latest_price_date(close_prices: pd.DataFrame, ticker: str) -> Optional[date]:
    _, price_date = _price_with_date_on_or_before(close_prices, ticker, date.max)
    return price_date


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return None


def _iso_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc).isoformat()
    parsed = str(value)
    return parsed if _parse_date(parsed) is not None else None


def _generate_briefing(
    market_regime: dict[str, Any],
    market_conditions: dict[str, Any],
    recommendations: list[dict[str, Any]],
    active_tracking: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    regime = market_regime.get("effectiveState", "unknown")
    regime_label = str(regime).replace("_", " ")
    risk_level = market_conditions.get("riskLevel", "Unknown")

    buys = [r for r in recommendations if r["action"] == "BUY"]
    sells = [r for r in recommendations if r["action"] == "SELL"]
    watches = [r for r in recommendations if r["action"] == "WATCH"]
    avoids = [r for r in recommendations if r["action"] == "AVOID"]

    top_buy = buys[0] if buys else None
    top_sell = sells[0] if sells else None
    top_watch = watches[0] if watches else None
    top_avoid = avoids[0] if avoids else None

    parts: list[str] = []
    if risk_level == "Unknown":
        parts.append(
            f"Market regime is {regime_label}; macro coverage is insufficient, so no macro-risk assumption was applied."
        )
    else:
        parts.append(f"Market regime is {regime_label}; macro risk level is {risk_level}.")

    if risk_level in ("Elevated", "Extreme"):
        parts.append("Agent has tightened score and beta filters accordingly.")

    if top_buy:
        parts.append(
            f"Top BUY idea is {top_buy['ticker']} (score {top_buy['alphaScore']:.1f}) — {top_buy['whyNow']}"
        )
    if top_sell:
        parts.append(
            f"Top SELL/SHORT idea is {top_sell['ticker']} (score {top_sell['alphaScore']:.1f}) — {top_sell['whyNow']}"
        )
    if top_watch:
        parts.append(
            f"Best WATCH candidate is {top_watch['ticker']} (score {top_watch['alphaScore']:.1f})."
        )
    if top_avoid:
        parts.append(
            f"Top AVOID name is {top_avoid['ticker']} (score {top_avoid['alphaScore']:.1f})."
        )
    if alerts:
        parts.append(f"There are {len(alerts)} active alert(s).")
    if outcomes:
        avg_return = sum(o["returnPct"] for o in outcomes) / len(outcomes)
        parts.append(f"Tracked recommendations average return: {avg_return:+.2f}%.")

    return {
        "summary": " ".join(parts),
        "regime": regime_label,
        "riskLevel": risk_level,
        "topBuy": top_buy,
        "topSell": top_sell,
        "topWatch": top_watch,
        "topAvoid": top_avoid,
        "counts": {
            "buy": len(buys),
            "sell": len(sells),
            "watch": len(watches),
            "avoid": len(avoids),
            "alerts": len(alerts),
            "outcomes": len(outcomes),
        },
    }
