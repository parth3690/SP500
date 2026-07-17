from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from ..models import Constituent
from .sp500 import normalize_yahoo_ticker


SECTOR_ETFS: dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

ALPHA_SIGNAL_IDS = [
    "momentum",
    "relative_strength",
    "trend",
    "risk_adjusted",
    "factor_exposure",
    "regime_fit",
    "catalyst",
    "revision_proxy",
]


def alpha_universe_tickers(constituents: Iterable[Constituent]) -> list[str]:
    constituents_list = list(constituents)
    tickers = [c.yahooTicker for c in constituents_list]
    sector_tickers = [SECTOR_ETFS.get(c.sector) for c in constituents_list]
    tickers.extend(["SPY", *[ticker for ticker in sector_tickers if ticker]])
    return list(dict.fromkeys([t for t in tickers if t]))


def _pct(series: pd.Series, periods: int, idx: Optional[int] = None) -> Optional[float]:
    if idx is None:
        idx = len(series) - 1
    start_idx = idx - periods
    if start_idx < 0 or idx < 0 or idx >= len(series):
        return None
    start = series.iloc[start_idx]
    end = series.iloc[idx]
    if pd.isna(start) or pd.isna(end) or float(start) == 0:
        return None
    return (float(end) / float(start) - 1.0) * 100.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        out = float(v)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _annualized_vol(ret: pd.Series) -> Optional[float]:
    r = ret.dropna()
    if len(r) < 10:
        return None
    return float(r.std() * math.sqrt(252.0) * 100.0)


def _beta(stock_ret: pd.Series, spy_ret: pd.Series) -> Optional[float]:
    joined = pd.concat([stock_ret, spy_ret], axis=1).dropna()
    if len(joined) < 30:
        return None
    market_var = float(joined.iloc[:, 1].var())
    if market_var == 0:
        return None
    return float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / market_var)


def _max_drawdown(series: pd.Series, window: int = 63) -> Optional[float]:
    s = series.dropna().iloc[-window:]
    if len(s) < 5:
        return None
    peak = s.cummax()
    dd = (s / peak - 1.0) * 100.0
    return float(dd.min())


def _market_regime(spy: pd.Series) -> dict[str, Any]:
    s = spy.dropna()
    if len(s) < 200:
        return {"state": "neutral", "spyTrend": "unknown", "spyDrawdownPct": None}
    price = float(s.iloc[-1])
    sma50 = float(s.rolling(50).mean().iloc[-1])
    sma200 = float(s.rolling(200).mean().iloc[-1])
    dd = _max_drawdown(s, 63)
    if price > sma50 > sma200 and (dd is None or dd > -8):
        state = "risk_on"
    elif price < sma200 or (dd is not None and dd <= -10):
        state = "risk_off"
    else:
        state = "neutral"
    return {
        "state": state,
        "spyTrend": "above_50_200" if price > sma50 > sma200 else "mixed_or_below",
        "spyDrawdownPct": round(dd, 2) if dd is not None else None,
    }


def _score_core(
    *,
    momentum20: float,
    momentum63: float,
    rs_spy20: float,
    rs_sector20: float,
    price: float,
    sma50: Optional[float],
    sma200: Optional[float],
    volatility20: Optional[float],
    beta_vs_spy: Optional[float],
    drawdown63: Optional[float],
    sector_strength20: float,
    regime: str,
    risk_mode: str,
) -> dict[str, float | str]:
    trend_points = 0.0
    if sma50 is not None:
        trend_points += 10.0 if price > sma50 else -10.0
    if sma200 is not None:
        trend_points += 8.0 if price > sma200 else -8.0
    if sma50 is not None and sma200 is not None:
        trend_points += 7.0 if sma50 > sma200 else -7.0
    trend_score = _clamp(50.0 + trend_points, 0.0, 100.0)

    momentum_score = _clamp(50.0 + momentum20 * 1.1 + momentum63 * 0.45, 0.0, 100.0)
    rs_score = _clamp(50.0 + rs_spy20 * 2.0 + rs_sector20 * 1.5, 0.0, 100.0)
    vol = volatility20 if volatility20 is not None else 35.0
    dd = drawdown63 if drawdown63 is not None else -10.0
    beta = beta_vs_spy if beta_vs_spy is not None else 1.0
    risk_score = _clamp(100.0 - vol * 1.05 + dd * 0.8, 0.0, 100.0)
    if 0.65 <= beta <= 1.35:
        factor_score = 65.0
    elif beta < 0.65:
        factor_score = 58.0
    else:
        factor_score = max(20.0, 65.0 - (beta - 1.35) * 30.0)
    factor_score = _clamp(factor_score + sector_strength20 * 1.2, 0.0, 100.0)

    if regime == "risk_off":
        regime_score = 72.0 if beta <= 1.05 and vol <= 35.0 and rs_spy20 > 0 else 42.0
    elif regime == "risk_on":
        regime_score = 70.0 if rs_spy20 > 0 and momentum20 > 0 else 46.0
    else:
        regime_score = 60.0 if rs_spy20 > -2 and momentum63 > -5 else 45.0

    if risk_mode == "aggressive":
        weights = {
            "momentum": 0.28,
            "relativeStrength": 0.28,
            "trend": 0.18,
            "risk": 0.08,
            "factor": 0.08,
            "regime": 0.10,
        }
    elif risk_mode == "defensive":
        weights = {
            "momentum": 0.16,
            "relativeStrength": 0.20,
            "trend": 0.17,
            "risk": 0.24,
            "factor": 0.12,
            "regime": 0.11,
        }
    else:
        weights = {
            "momentum": 0.22,
            "relativeStrength": 0.25,
            "trend": 0.18,
            "risk": 0.15,
            "factor": 0.10,
            "regime": 0.10,
        }

    technical_score = (
        momentum_score * weights["momentum"]
        + rs_score * weights["relativeStrength"]
        + trend_score * weights["trend"]
        + risk_score * weights["risk"]
        + factor_score * weights["factor"]
        + regime_score * weights["regime"]
    )

    risk_adjusted = technical_score * (risk_score / 100.0)
    expected_return20 = (
        momentum20 * 0.18
        + rs_spy20 * 0.22
        + rs_sector20 * 0.12
        + max(0.0, trend_score - 50.0) * 0.035
        - max(0.0, vol - 30.0) * 0.035
    )

    return {
        "technicalScore": round(technical_score, 2),
        "momentumScore": round(momentum_score, 2),
        "relativeStrengthScore": round(rs_score, 2),
        "trendScore": round(trend_score, 2),
        "riskScore": round(risk_score, 2),
        "factorScore": round(factor_score, 2),
        "regimeScore": round(regime_score, 2),
        "riskAdjustedScore": round(risk_adjusted, 2),
        "expectedReturn20d": round(expected_return20, 2),
    }


def _backtest_alpha_signal(
    close: pd.Series,
    spy: pd.Series,
    sector: Optional[pd.Series],
    *,
    risk_mode: str,
    regime_override: str,
    signal_directions: dict[str, str],
) -> list[dict[str, Any]]:
    close = close.dropna()
    spy = spy.dropna()
    if sector is None:
        sector = spy
    sector = sector.dropna()
    common = pd.concat([close, spy, sector], axis=1, join="inner").dropna()
    if len(common) < 260:
        return []

    stock = common.iloc[:, 0]
    market = common.iloc[:, 1]
    sec = common.iloc[:, 2]
    stock_ret = stock.pct_change()
    spy_ret = market.pct_change()
    thresholds = {
        "alpha_score": (68.0, 42.0),
        "momentum": (62.0, 45.0),
        "relative_strength": (62.0, 45.0),
        "trend": (62.0, 45.0),
        "risk_adjusted": (62.0, 45.0),
        "factor_exposure": (62.0, 45.0),
        "regime_fit": (62.0, 45.0),
    }
    active_directions = {
        signal: direction
        for signal, direction in signal_directions.items()
        if signal in thresholds and direction in ("bullish", "bearish")
    }
    samples: dict[str, dict[int, list[tuple[float, float]]]] = {
        signal: {20: [], 60: []} for signal in active_directions
    }

    for idx in range(200, len(stock) - 61, 5):
        p = float(stock.iloc[idx])
        sma50 = stock.iloc[: idx + 1].rolling(50).mean().iloc[-1]
        sma200 = stock.iloc[: idx + 1].rolling(200).mean().iloc[-1]
        m20 = _pct(stock, 21, idx) or 0.0
        m63 = _pct(stock, 63, idx) or 0.0
        spy20 = _pct(market, 21, idx) or 0.0
        sec20 = _pct(sec, 21, idx) or 0.0
        vol = _annualized_vol(stock_ret.iloc[max(0, idx - 21): idx + 1])
        beta = _beta(stock_ret.iloc[max(0, idx - 63): idx + 1], spy_ret.iloc[max(0, idx - 63): idx + 1])
        dd_series = stock.iloc[max(0, idx - 63): idx + 1]
        dd = _max_drawdown(dd_series, len(dd_series))
        sample_regime = (
            _market_regime(market.iloc[: idx + 1])["state"]
            if regime_override == "auto"
            else regime_override
        )
        scores = _score_core(
            momentum20=m20,
            momentum63=m63,
            rs_spy20=m20 - spy20,
            rs_sector20=m20 - sec20,
            price=p,
            sma50=_safe_float(sma50),
            sma200=_safe_float(sma200),
            volatility20=vol,
            beta_vs_spy=beta,
            drawdown63=dd,
            sector_strength20=sec20 - spy20,
            regime=sample_regime,
            risk_mode=risk_mode,
        )
        score_by_signal = {
            "alpha_score": float(scores["technicalScore"]),
            "momentum": float(scores["momentumScore"]),
            "relative_strength": float(scores["relativeStrengthScore"]),
            "trend": float(scores["trendScore"]),
            "risk_adjusted": float(scores["riskScore"]),
            "factor_exposure": float(scores["factorScore"]),
            "regime_fit": float(scores["regimeScore"]),
        }
        forward: dict[int, tuple[float, float]] = {}
        for horizon in (20, 60):
            stock_fwd = _pct(stock, horizon, idx + horizon)
            spy_fwd = _pct(market, horizon, idx + horizon)
            if stock_fwd is not None and spy_fwd is not None:
                forward[horizon] = (stock_fwd, spy_fwd)

        for signal, direction in active_directions.items():
            high, low = thresholds[signal]
            score = score_by_signal[signal]
            triggered = score >= high if direction == "bullish" else score <= low
            if not triggered:
                continue
            sign = 1.0 if direction == "bullish" else -1.0
            for horizon, (stock_fwd, spy_fwd) in forward.items():
                samples[signal][horizon].append((stock_fwd * sign, spy_fwd * sign))

    out: list[dict[str, Any]] = []
    for signal, horizons in samples.items():
        for horizon, vals in horizons.items():
            if not vals:
                continue
            stock_vals = [v[0] for v in vals]
            spy_vals = [v[1] for v in vals]
            alpha_vals = [a - b for a, b in vals]
            out.append(
                {
                    "signal": signal,
                    "horizonDays": horizon,
                    "sampleSize": len(vals),
                    "winRate": round(sum(1 for v in stock_vals if v > 0) / len(stock_vals) * 100.0, 1),
                    "avgReturn": round(float(np.mean(stock_vals)), 2),
                    "medianReturn": round(float(np.median(stock_vals)), 2),
                    "benchmarkAvgReturn": round(float(np.mean(spy_vals)), 2),
                    "alphaAvgReturn": round(float(np.mean(alpha_vals)), 2),
                }
            )
    return out


def catalyst_scores_from_info(info: dict[str, Any], current_price: float) -> dict[str, Any]:
    target = _safe_float(info.get("targetMeanPrice"))
    recommendation = str(info.get("recommendationKey") or "").replace("_", " ").title()
    analyst_count = _safe_float(info.get("numberOfAnalystOpinions"))
    revenue_growth = _safe_float(info.get("revenueGrowth"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))
    earnings_q_growth = _safe_float(info.get("earningsQuarterlyGrowth"))

    target_upside = None
    if target is not None and current_price > 0:
        target_upside = (target / current_price - 1.0) * 100.0

    catalyst_score = 0.0
    revision_score = 0.0
    notes: list[str] = []

    if target_upside is not None:
        catalyst_score += _clamp(target_upside * 0.45, -10.0, 18.0)
        notes.append(f"Analyst target upside {target_upside:.1f}%.")
    if recommendation:
        if "Buy" in recommendation:
            catalyst_score += 6.0
        elif "Sell" in recommendation:
            catalyst_score -= 8.0
        notes.append(f"Consensus: {recommendation}.")
    if analyst_count is not None:
        notes.append(f"{int(analyst_count)} analyst opinion(s).")

    for label, value in [
        ("Revenue growth", revenue_growth),
        ("Earnings growth", earnings_growth),
        ("Quarterly earnings growth", earnings_q_growth),
    ]:
        if value is None:
            continue
        revision_score += _clamp(value * 55.0, -12.0, 14.0)
        notes.append(f"{label}: {value * 100:.1f}%.")

    return {
        "catalystScore": round(_clamp(catalyst_score, -15.0, 25.0), 2),
        "revisionScore": round(_clamp(revision_score, -15.0, 25.0), 2),
        "catalystNotes": notes[:4] or ["No lightweight catalyst signal found."],
    }


def _enrich_ticker(symbol: str, current_price: float) -> dict[str, Any]:
    try:
        import yfinance as yf

        info = yf.Ticker(normalize_yahoo_ticker(symbol)).info or {}
    except Exception:
        return {
            "catalystScore": 0.0,
            "revisionScore": 0.0,
            "catalystNotes": ["No lightweight catalyst data available."],
        }
    return catalyst_scores_from_info(info, current_price)


def apply_alpha_enrichment(row: dict[str, Any], data: dict[str, Any]) -> None:
    row.update(data)
    row["alphaScore"] = round(
        _clamp(
            row["technicalScore"]
            + data["catalystScore"] * 0.18
            + data["revisionScore"] * 0.22,
            0.0,
            100.0,
        ),
        2,
    )
    row["signals"] = _candidate_signals(row)
    row["tradePlan"] = _trade_plan(row)


def _enrich_candidates(rows: list[dict[str, Any]], enrich_top: int) -> None:
    if enrich_top <= 0:
        return
    top = rows[:enrich_top]
    with ThreadPoolExecutor(max_workers=min(6, len(top))) as pool:
        futures = {
            pool.submit(_enrich_ticker, row["ticker"], row["currentPrice"]): row
            for row in top
        }
        for future in as_completed(futures):
            row = futures[future]
            apply_alpha_enrichment(row, future.result())


def _signal_state(score: float, high: float = 62.0, low: float = 45.0) -> str:
    if score >= high:
        return "bullish"
    if score <= low:
        return "bearish"
    return "neutral"


def _candidate_signals(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "momentum",
            "label": "Momentum",
            "state": _signal_state(row["momentumScore"]),
            "contribution": round(row["momentumScore"] - 50.0, 2),
            "detail": f"20d {row['momentum20d']:.1f}%, 63d {row['momentum63d']:.1f}%.",
        },
        {
            "id": "relative_strength",
            "label": "Relative strength",
            "state": _signal_state(row["relativeStrengthScore"]),
            "contribution": round(row["relativeStrengthScore"] - 50.0, 2),
            "detail": f"Vs SPY {row['rsVsSpy20d']:.1f}%, vs sector {row['rsVsSector20d']:.1f}%.",
        },
        {
            "id": "trend",
            "label": "Trend",
            "state": _signal_state(row["trendScore"]),
            "contribution": round(row["trendScore"] - 50.0, 2),
            "detail": row["trendState"],
        },
        {
            "id": "risk_adjusted",
            "label": "Risk adjusted",
            "state": _signal_state(row["riskScore"]),
            "contribution": round(row["riskScore"] - 50.0, 2),
            "detail": f"Vol {row['volatility20d']:.1f}%, beta {row['betaVsSpy']:.2f}.",
        },
        {
            "id": "factor_exposure",
            "label": "Factor exposure",
            "state": _signal_state(row["factorScore"]),
            "contribution": round(row["factorScore"] - 50.0, 2),
            "detail": row["factorExposure"],
        },
        {
            "id": "regime_fit",
            "label": "Regime fit",
            "state": _signal_state(row["regimeScore"]),
            "contribution": round(row["regimeScore"] - 50.0, 2),
            "detail": row["regimeFit"],
        },
        {
            "id": "catalyst",
            "label": "Catalyst",
            "state": _signal_state(50.0 + row.get("catalystScore", 0.0), high=56.0, low=43.0),
            "contribution": row.get("catalystScore", 0.0),
            "detail": " ".join(row.get("catalystNotes", [])[:2]),
        },
        {
            "id": "revision_proxy",
            "label": "Revision proxy",
            "state": _signal_state(50.0 + row.get("revisionScore", 0.0), high=56.0, low=43.0),
            "contribution": row.get("revisionScore", 0.0),
            "detail": "Lightweight proxy from growth, target, and recommendation data.",
        },
    ]


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def _next_monthly_expiry(days_out: int = 45, *, as_of: Optional[date] = None) -> str:
    target = (as_of or date.today()) + timedelta(days=days_out)
    expiry = _third_friday(target.year, target.month)
    if expiry < target:
        year = target.year + (1 if target.month == 12 else 0)
        month = 1 if target.month == 12 else target.month + 1
        expiry = _third_friday(year, month)
    return expiry.isoformat()


def _round_strike(price: float) -> float:
    if price >= 500:
        step = 10.0
    elif price >= 100:
        step = 5.0
    elif price >= 25:
        step = 2.5
    else:
        step = 1.0
    return round(round(price / step) * step, 2)


def _trade_plan(row: dict[str, Any]) -> dict[str, Any]:
    price = float(row["currentPrice"])
    score = float(row["alphaScore"])
    risk_score = float(row["riskScore"])
    expected = float(row["expectedReturn20d"])
    rs_spy = float(row["rsVsSpy20d"])
    vol = max(float(row["volatility20d"]), 12.0)
    beta = float(row["betaVsSpy"])
    stop_pct = _clamp(vol / 7.0, 4.0, 14.0)
    target1_pct = _clamp(max(expected, 3.0) + vol / 10.0, 5.0, 18.0)
    target2_pct = _clamp(max(expected, 5.0) + vol / 5.0, 9.0, 30.0)

    if score >= 68 and expected > 0 and rs_spy > 0:
        action = "BUY"
        entry = price
        buy_below = price * 1.01
        stop = price * (1.0 - stop_pct / 100.0)
        target1 = price * (1.0 + target1_pct / 100.0)
        target2 = price * (1.0 + target2_pct / 100.0)
        option_direction = "bullish"
        option_strategy = "Call debit spread" if price >= 150 else "Long call"
        option_strike = _round_strike(price * 1.03)
        rationale = (
            f"BUY: alpha score {score:.1f}, positive 20d expected return, "
            f"and relative strength vs SPY is {rs_spy:.1f}%."
        )
    elif score <= 42 and expected < 0 and rs_spy < 0:
        action = "SELL"
        entry = price
        buy_below = None
        stop = price * (1.0 + stop_pct / 100.0)
        target1 = price * (1.0 - target1_pct / 100.0)
        target2 = price * (1.0 - target2_pct / 100.0)
        option_direction = "bearish"
        option_strategy = "Put debit spread" if price >= 100 else "Long put"
        option_strike = _round_strike(price * 0.97)
        rationale = (
            f"SELL/SHORT: alpha score {score:.1f}, negative expected return, "
            f"and relative strength vs SPY is {rs_spy:.1f}%."
        )
    elif score < 50 or risk_score < 35:
        action = "AVOID"
        entry = price
        buy_below = None
        stop = None
        target1 = None
        target2 = None
        option_direction = None
        option_strategy = None
        option_strike = None
        rationale = (
            f"AVOID: signal quality is weak or risk is high. Alpha score {score:.1f}, "
            f"risk score {risk_score:.1f}, beta {beta:.2f}."
        )
    else:
        action = "WATCH"
        entry = price
        buy_below = price * 0.99 if expected >= 0 else None
        stop = None
        target1 = price * (1.0 + target1_pct / 100.0) if expected >= 0 else None
        target2 = price * (1.0 + target2_pct / 100.0) if expected >= 0 else None
        option_direction = "bullish" if expected > 0 else None
        option_strategy = "Call debit spread" if expected > 0 and price >= 150 else ("Long call" if expected > 0 else None)
        option_strike = _round_strike(price * 1.03) if expected > 0 else None
        rationale = (
            f"WATCH: alpha score {score:.1f} is not strong enough for a fresh trade. "
            "Wait for score improvement or cleaner relative strength."
        )

    risk_reward = None
    if stop is not None and target1 is not None:
        risk = abs(entry - stop)
        reward = abs(target1 - entry)
        if risk > 0:
            risk_reward = reward / risk

    option_rationale = None
    option_expiry = _next_monthly_expiry() if option_strategy else None
    if option_strategy and option_strike:
        option_rationale = (
            f"Use {option_strategy.lower()} around {option_strike:.2f} strike expiring {option_expiry} "
            "when stock capital, share price, or defined-risk sizing makes common stock less practical."
        )

    confidence_basis = score
    if action == "SELL":
        confidence_basis = 100.0 - score
    elif action == "AVOID":
        confidence_basis = max(100.0 - score, 100.0 - risk_score)

    return {
        "action": action,
        "confidence": round(
            _clamp(
                confidence_basis * 0.65
                + risk_score * 0.25
                + min(abs(expected) * 3.0, 10.0),
                0.0,
                100.0,
            ),
            1,
        ),
        "horizon": "20-60 trading days",
        "entry": round(entry, 2),
        "buyBelow": round(buy_below, 2) if buy_below is not None else None,
        "sellAbove": round(entry, 2) if action == "SELL" else None,
        "stop": round(stop, 2) if stop is not None else None,
        "target1": round(target1, 2) if target1 is not None else None,
        "target2": round(target2, 2) if target2 is not None else None,
        "riskReward": round(risk_reward, 2) if risk_reward is not None else None,
        "optionStrategy": option_strategy,
        "optionDirection": option_direction,
        "optionStrike": option_strike,
        "optionExpiry": option_expiry,
        "optionRationale": option_rationale,
        "rationale": rationale,
    }


def compute_alpha_candidates(
    constituents: Iterable[Constituent],
    close_prices: pd.DataFrame,
    *,
    limit: int = 50,
    min_score: float = 55.0,
    sector: Optional[str] = None,
    max_beta: Optional[float] = None,
    risk_mode: str = "balanced",
    regime_override: str = "auto",
    enrich_top: int = 20,
    include_lowest: int = 0,
) -> dict[str, Any]:
    if close_prices is None or close_prices.empty or "SPY" not in close_prices.columns:
        return {
            "asOf": datetime.now(timezone.utc).isoformat(),
            "candidates": [],
            "marketRegime": {"state": "unknown", "spyTrend": "unknown", "spyDrawdownPct": None},
            "meta": {"computed": 0, "total": 0, "warnings": ["Missing SPY or price data."]},
        }

    close_prices = close_prices.sort_index()
    constituents_list = list(constituents)
    spy = close_prices["SPY"].dropna()
    regime = _market_regime(spy)
    effective_regime = regime["state"] if regime_override == "auto" else regime_override
    spy_return20 = _pct(spy, 21) or 0.0

    eligible_constituents = [c for c in constituents_list if not sector or c.sector == sector]
    rows: list[dict[str, Any]] = []
    skipped = 0

    for c in eligible_constituents:
        if c.yahooTicker not in close_prices.columns:
            skipped += 1
            continue
        s = close_prices[c.yahooTicker].dropna()
        if len(s) < 220:
            skipped += 1
            continue
        price = float(s.iloc[-1])
        price_date = s.index[-1]
        momentum20 = _pct(s, 21) or 0.0
        momentum63 = _pct(s, 63) or 0.0
        sector_etf = SECTOR_ETFS.get(c.sector, "SPY")
        sector_series = close_prices[sector_etf].dropna() if sector_etf in close_prices.columns else spy
        sector_return20 = _pct(sector_series, 21) or spy_return20
        rs_spy20 = momentum20 - spy_return20
        rs_sector20 = momentum20 - sector_return20
        stock_ret = s.pct_change()
        beta_vs_spy = _beta(stock_ret.iloc[-126:], spy.pct_change().iloc[-126:])
        volatility20 = _annualized_vol(stock_ret.iloc[-21:])
        drawdown63 = _max_drawdown(s, 63)
        sma50 = _safe_float(s.rolling(50).mean().iloc[-1])
        sma200 = _safe_float(s.rolling(200).mean().iloc[-1])
        if max_beta is not None and beta_vs_spy is not None and beta_vs_spy > max_beta:
            continue

        scores = _score_core(
            momentum20=momentum20,
            momentum63=momentum63,
            rs_spy20=rs_spy20,
            rs_sector20=rs_sector20,
            price=price,
            sma50=sma50,
            sma200=sma200,
            volatility20=volatility20,
            beta_vs_spy=beta_vs_spy,
            drawdown63=drawdown63,
            sector_strength20=sector_return20 - spy_return20,
            regime=effective_regime,
            risk_mode=risk_mode,
        )
        alpha_score = float(scores["technicalScore"])
        if alpha_score < min_score and include_lowest <= 0:
            continue

        above50 = sma50 is not None and price > sma50
        above200 = sma200 is not None and price > sma200
        trend_state = (
            "price above 50/200DMA"
            if above50 and above200
            else "price below 50/200DMA"
            if not above50 and not above200
            else "price above 50DMA only"
            if above50
            else "mixed trend"
        )
        beta_value = beta_vs_spy if beta_vs_spy is not None else 1.0
        factor_exposure = (
            "balanced beta"
            if 0.65 <= beta_value <= 1.35
            else ("defensive beta" if beta_value < 0.65 else "high beta")
        )
        regime_fit = (
            f"fits {effective_regime.replace('_', ' ')}"
            if float(scores["regimeScore"]) >= 60.0
            else f"weak fit for {effective_regime.replace('_', ' ')}"
        )
        row = {
            "rank": 0,
            "ticker": c.ticker,
            "companyName": c.companyName,
            "sector": c.sector,
            "currentPrice": round(price, 2),
            "priceDate": price_date.date() if hasattr(price_date, "date") else price_date,
            "alphaScore": round(alpha_score, 2),
            "technicalScore": scores["technicalScore"],
            "riskAdjustedScore": scores["riskAdjustedScore"],
            "expectedReturn20d": scores["expectedReturn20d"],
            "momentum20d": round(momentum20, 2),
            "momentum63d": round(momentum63, 2),
            "rsVsSpy20d": round(rs_spy20, 2),
            "rsVsSector20d": round(rs_sector20, 2),
            "sectorStrength20d": round(sector_return20 - spy_return20, 2),
            "volatility20d": round(volatility20 if volatility20 is not None else 0.0, 2),
            "betaVsSpy": round(beta_value, 2),
            "maxDrawdown63d": round(drawdown63 if drawdown63 is not None else 0.0, 2),
            "trendState": trend_state,
            "factorExposure": factor_exposure,
            "regimeFit": regime_fit,
            "momentumScore": scores["momentumScore"],
            "relativeStrengthScore": scores["relativeStrengthScore"],
            "trendScore": scores["trendScore"],
            "riskScore": scores["riskScore"],
            "factorScore": scores["factorScore"],
            "regimeScore": scores["regimeScore"],
            "catalystScore": 0.0,
            "revisionScore": 0.0,
            "catalystNotes": ["Catalyst enrichment is applied only to top names."],
            "tradePlan": {},
            "signals": [],
            "backtests": [],
        }
        row["signals"] = _candidate_signals(row)
        row["tradePlan"] = _trade_plan(row)
        rows.append(row)

    rows.sort(key=lambda r: (r["alphaScore"], r["riskAdjustedScore"]), reverse=True)
    _enrich_candidates(rows, min(enrich_top, max(limit, 0)))
    for row in rows:
        row["tradePlan"] = _trade_plan(row)
    rows.sort(key=lambda r: (r["alphaScore"], r["riskAdjustedScore"]), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    if include_lowest > 0:
        selected = [*rows[:limit], *sorted(rows[-include_lowest:], key=lambda r: r["alphaScore"])]
        filtered = list({row["ticker"]: row for row in selected}.values())
    else:
        filtered = rows[:limit]

    by_ticker = {c.ticker: c for c in constituents_list}
    for row in filtered:
        c = by_ticker.get(row["ticker"])
        if c is None or c.yahooTicker not in close_prices.columns:
            continue
        stock = close_prices[c.yahooTicker].dropna()
        sector_etf = SECTOR_ETFS.get(c.sector, "SPY")
        sector_series = close_prices[sector_etf].dropna() if sector_etf in close_prices.columns else spy
        signal_directions = {s["id"]: s["state"] for s in row["signals"]}
        action = row["tradePlan"]["action"]
        signal_directions["alpha_score"] = (
            "bullish" if action == "BUY" else "bearish" if action == "SELL" else "neutral"
        )
        row["backtests"] = _backtest_alpha_signal(
            stock,
            spy,
            sector_series,
            risk_mode=risk_mode,
            regime_override=regime_override,
            signal_directions=signal_directions,
        )
    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "marketRegime": {
            **regime,
            "effectiveState": effective_regime,
            "riskMode": risk_mode,
        },
        "candidates": filtered,
        "meta": {
            "total": len(constituents_list),
            "eligible": len(eligible_constituents),
            "computed": len(rows),
            "returned": len(filtered),
            "skipped": skipped,
            "priceCoveragePct": round(
                (len(eligible_constituents) - skipped) / len(eligible_constituents) * 100,
                1,
            ) if eligible_constituents else 0.0,
            "minScore": min_score,
            "sector": sector,
            "maxBeta": max_beta,
            "signals": ALPHA_SIGNAL_IDS,
            "warnings": [
                "Catalyst and revision fields are lightweight proxies, enriched only for top-ranked names.",
                "Catalyst and revision signals are not historically backtested without point-in-time fundamentals.",
            ],
        },
    }
