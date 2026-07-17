from __future__ import annotations

import datetime as dt
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import pandas as pd
import yfinance as yf

from .prices import fetch_fmp_price_history

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _fred_api_key() -> str:
    return os.environ.get("FRED_API_KEY", "").strip()

MANUAL_ENV_KEYS = {
    "cb_consumer_confidence": "MARKET_cb_consumer_confidence",
    "cb_net_pct_stocks_higher": "MARKET_cb_net_pct_stocks_higher",
    "sell_side_indicator": "MARKET_sell_side_indicator",
    "ltg_5yr_z": "MARKET_ltg_5yr_z",
    "mna_10yr_z": "MARKET_mna_10yr_z",
    "credit_stress_indicator": "MARKET_credit_stress_indicator",
}


def _bool_state(triggered: Optional[bool]) -> str:
    if triggered is None:
        return "unknown"
    return "triggered" if triggered else "not_triggered"


def _fred_series(
    series_id: str,
    start: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    sort_order: str = "asc",
) -> list[tuple[str, float]]:
    if _fred_api_key():
        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": _fred_api_key(),
            "file_type": "json",
            "sort_order": sort_order,
        }
        if start:
            params["observation_start"] = start
        if limit is not None:
            params["limit"] = str(limit)
        with httpx.Client(timeout=30) as client:
            r = client.get(FRED_BASE, params=params)
            r.raise_for_status()
            payload = r.json()
        out: list[tuple[str, float]] = []
        for obs in payload.get("observations", []):
            v = obs.get("value", ".")
            if v not in (".", "", None):
                try:
                    out.append((obs["date"], float(v)))
                except ValueError:
                    pass
    else:
        with httpx.Client(timeout=30) as client:
            r = client.get(FRED_CSV_BASE, params={"id": series_id})
            r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if "observation_date" not in df.columns or series_id not in df.columns:
            return []
        if start:
            df = df[df["observation_date"] >= start]
        out = []
        for _, row in df.iterrows():
            try:
                v = float(row[series_id])
            except (TypeError, ValueError):
                continue
            out.append((str(row["observation_date"]), v))
        if sort_order == "desc":
            out = list(reversed(out))
        if limit is not None:
            out = out[:limit]
    if sort_order == "desc":
        out.reverse()
    return out


def _zscore(series: list[float], window: int) -> Optional[float]:
    if len(series) < window + 1:
        window = len(series) - 1
    if window < 12:
        return None
    sample = series[-(window + 1) : -1]
    mean = sum(sample) / len(sample)
    var = sum((x - mean) ** 2 for x in sample) / len(sample)
    sd = var**0.5
    if sd == 0:
        return None
    return (series[-1] - mean) / sd


def _fetch_multpl_pe() -> dict[str, float]:
    url = "https://www.multpl.com/s-p-500-pe-ratio/table/by-month"
    try:
        with httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0"}) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text
    except Exception:
        return {}
    out: dict[str, float] = {}
    for chunk in html.split("<tr"):
        m_date = re.search(r"([A-Z][a-z]{2})\s+\d+,\s+(\d{4})</td>", chunk)
        if not m_date:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", chunk, re.DOTALL)
        if len(tds) < 2:
            continue
        val_text = re.sub(r"<[^>]+>", "", tds[1])
        val_text = val_text.replace("\n", " ").strip()
        m_num = re.search(r"([\d.]+)", val_text)
        if not m_num:
            continue
        try:
            val = float(m_num.group(1))
        except ValueError:
            continue
        mon, year = m_date.group(1), m_date.group(2)
        try:
            month_num = dt.datetime.strptime(mon, "%b").month
            out[f"{year}-{month_num:02d}"] = val
        except ValueError:
            continue
    return out


def _read_manual(condition_id: str) -> Optional[float]:
    env_key = MANUAL_ENV_KEYS.get(condition_id)
    if not env_key:
        return None
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _fetch_inverted_curve() -> dict[str, Any]:
    start = (dt.date.today() - dt.timedelta(days=200)).isoformat()
    data = _fred_series("T10Y2Y", start=start)
    if not data:
        return {"value": None, "state": "unknown", "fetched": False, "note": "No T10Y2Y data"}
    latest = data[-1][1]
    inverted = any(v < 0 for _, v in data)
    return {
        "value": "yes" if inverted else "no",
        "state": "triggered" if inverted else "not_triggered",
        "fetched": True,
        "note": f"Latest 10y-2y spread: {latest:.2f}%",
    }


def _fetch_sloos() -> dict[str, Any]:
    data = _fred_series("DRTSCILM", limit=8, sort_order="desc")
    if not data:
        return {"value": None, "state": "unknown", "fetched": False, "note": "No SLOOS data"}
    latest = data[-1][1]
    triggered = latest > 0
    return {
        "value": round(latest, 2),
        "state": "triggered" if triggered else "not_triggered",
        "fetched": True,
        "note": f"SLOOS net % tightening (latest): {latest:.1f}",
    }


def _fetch_valuation_z() -> dict[str, Any]:
    cpi = _fred_series("CPIAUCSL", start="2000-01-01", limit=400)
    yoy: dict[str, float] = {}
    idx = {d[:7]: v for d, v in cpi}
    for k in sorted(idx):
        y, m = int(k[:4]), int(k[5:7])
        prev = f"{y-1:04d}-{m:02d}"
        if prev in idx and idx[prev]:
            yoy[k] = (idx[k] / idx[prev] - 1) * 100
    pe = _fetch_multpl_pe()
    if not pe:
        return {"value": None, "state": "unknown", "fetched": False, "note": "Could not load trailing P/E"}
    combined: list[float] = []
    for k in sorted(set(pe) & set(yoy)):
        combined.append(pe[k] + yoy[k])
    z = _zscore(combined, 120)
    if z is None:
        return {"value": None, "state": "unknown", "fetched": False, "note": "Insufficient history for z-score"}
    triggered = z > 1
    return {
        "value": round(z, 2),
        "state": "triggered" if triggered else "not_triggered",
        "fetched": True,
        "note": "10yr z of (trailing S&P 500 P/E + YoY CPI)",
    }


def _fetch_low_minus_high_pe_6m() -> dict[str, Any]:
    """Proxy: 6m return spread value ETF (VLUE) minus growth ETF (IVW), in ppt."""

    def _etf_return(sym: str) -> Optional[float]:
        closes = _download_close(sym, period="6mo")
        if closes is None:
            return None
        if len(closes) < 2:
            return None
        return (float(closes.iloc[-1]) / float(closes.iloc[0]) - 1) * 100

    try:
        low_ret = _etf_return("VLUE")
        high_ret = _etf_return("IVW")
        if low_ret is None or high_ret is None:
            return {"value": None, "state": "unknown", "fetched": False, "note": "Could not compute factor returns"}
        ppt = low_ret - high_ret
        triggered = ppt <= -2.5
        return {
            "value": round(ppt, 2),
            "state": "triggered" if triggered else "not_triggered",
            "fetched": True,
            "note": "Proxy: VLUE − IVW 6m return (ppt)",
        }
    except Exception as exc:
        return {"value": None, "state": "unknown", "fetched": False, "note": str(exc)}


def _download_close(symbol: str, *, period: str = "1y") -> Optional[pd.Series]:
    try:
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist is not None and not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if len(closes) > 1:
                closes.attrs["source"] = "Yahoo Finance"
                return closes
    except Exception:
        pass

    period_days = {"3mo": 120, "6mo": 220, "1y": 400}.get(period, 400)
    start = dt.date.today() - dt.timedelta(days=period_days)
    fred_symbol = {"SPY": "SP500", "^VIX": "VIXCLS"}.get(symbol)
    if fred_symbol:
        try:
            observations = _fred_series(fred_symbol, start=start.isoformat())
            if observations:
                index = pd.to_datetime([row[0] for row in observations])
                closes = pd.Series(
                    [row[1] for row in observations], index=index, name="Close"
                ).sort_index()
                closes.attrs["source"] = (
                    "FRED S&P 500 index proxy" if symbol == "SPY" else "FRED VIX"
                )
                return closes
        except Exception:
            pass

    history = fetch_fmp_price_history(symbol, start, dt.date.today() + dt.timedelta(days=1))
    if history.empty or "Close" not in history.columns:
        return None
    closes = history["Close"].dropna()
    closes.attrs["source"] = "Financial Modeling Prep"
    return closes if len(closes) > 1 else None


def _fetch_spy_below_200dma(closes: Optional[pd.Series] = None) -> dict[str, Any]:
    closes = closes if closes is not None else _download_close("SPY", period="1y")
    if closes is None or len(closes) < 200:
        return {"value": None, "state": "unknown", "fetched": False, "note": "Could not load enough SPY history"}
    latest = float(closes.iloc[-1])
    sma200 = float(closes.rolling(200).mean().iloc[-1])
    triggered = latest < sma200
    return {
        "value": round((latest / sma200 - 1) * 100, 2),
        "state": "triggered" if triggered else "not_triggered",
        "fetched": True,
        "note": (
            f"{closes.attrs.get('source', 'SPY')} latest {latest:.2f} vs 200DMA {sma200:.2f}; "
            "value is % above/below 200DMA"
        ),
    }


def _fetch_spy_50_below_200dma(closes: Optional[pd.Series] = None) -> dict[str, Any]:
    closes = closes if closes is not None else _download_close("SPY", period="1y")
    if closes is None or len(closes) < 200:
        return {"value": None, "state": "unknown", "fetched": False, "note": "Could not load enough SPY history"}
    sma50 = float(closes.rolling(50).mean().iloc[-1])
    sma200 = float(closes.rolling(200).mean().iloc[-1])
    triggered = sma50 < sma200
    return {
        "value": round((sma50 / sma200 - 1) * 100, 2),
        "state": "triggered" if triggered else "not_triggered",
        "fetched": True,
        "note": f"{closes.attrs.get('source', 'SPY')} 50DMA {sma50:.2f} vs 200DMA {sma200:.2f}",
    }


def _fetch_spy_3m_drawdown(closes: Optional[pd.Series] = None) -> dict[str, Any]:
    closes = closes if closes is not None else _download_close("SPY", period="1y")
    if closes is None or len(closes) < 50:
        return {"value": None, "state": "unknown", "fetched": False, "note": "Could not load enough SPY history"}
    recent = closes.iloc[-63:] if len(closes) >= 63 else closes
    dd = (float(recent.iloc[-1]) / float(recent.max()) - 1) * 100
    triggered = dd <= -8
    return {
        "value": round(dd, 2),
        "state": "triggered" if triggered else "not_triggered",
        "fetched": True,
        "note": f"{closes.attrs.get('source', 'SPY')} drawdown from 3-month high",
    }


def _fetch_vix_elevated(closes: Optional[pd.Series] = None) -> dict[str, Any]:
    closes = closes if closes is not None else _download_close("^VIX", period="3mo")
    if closes is None:
        return {"value": None, "state": "unknown", "fetched": False, "note": "Could not load VIX"}
    latest = float(closes.iloc[-1])
    triggered = latest > 25
    return {
        "value": round(latest, 2),
        "state": "triggered" if triggered else "not_triggered",
        "fetched": True,
        "note": f"{closes.attrs.get('source', 'VIX')} latest close",
    }


def _fetch_high_yield_spread() -> dict[str, Any]:
    data = _fred_series("BAMLH0A0HYM2", start=(dt.date.today() - dt.timedelta(days=120)).isoformat())
    if not data:
        return {"value": None, "state": "unknown", "fetched": False, "note": "No high-yield OAS data"}
    latest = data[-1][1]
    triggered = latest > 5
    return {
        "value": round(latest, 2),
        "state": "triggered" if triggered else "not_triggered",
        "fetched": True,
        "note": "ICE BofA US High Yield OAS",
    }


def _fetch_sahm_rule() -> dict[str, Any]:
    data = _fred_series("SAHMREALTIME", start=(dt.date.today() - dt.timedelta(days=730)).isoformat())
    if not data:
        return {"value": None, "state": "unknown", "fetched": False, "note": "No Sahm rule data"}
    latest = data[-1][1]
    triggered = latest >= 0.5
    return {
        "value": round(latest, 2),
        "state": "triggered" if triggered else "not_triggered",
        "fetched": True,
        "note": "Sahm recession indicator, real-time",
    }


def _fetch_manual(condition_id: str, predicate) -> dict[str, Any]:
    val = _read_manual(condition_id)
    if val is None:
        return {
            "value": None,
            "state": "unknown",
            "fetched": False,
            "note": f"Set {MANUAL_ENV_KEYS.get(condition_id, 'env')} in backend .env",
        }
    triggered = predicate(val)
    display = val
    if condition_id == "sell_side_indicator":
        display = "Sell" if val >= 1 else "Hold/Buy"
    return {
        "value": display,
        "state": _bool_state(triggered),
        "fetched": True,
        "note": "From backend .env override",
    }


def _risk_assessment(fetched_count: int, triggered_count: int, total_count: int) -> tuple[int, str, str]:
    coverage_pct = round((fetched_count / total_count) * 100) if total_count else 0
    if coverage_pct < 50:
        return coverage_pct, "Unknown", "Insufficient"
    triggered_ratio = triggered_count / max(fetched_count, 1)
    risk_level = (
        "Extreme"
        if triggered_ratio >= 0.70
        else "Elevated"
        if triggered_ratio >= 0.50
        else "Watch"
        if triggered_ratio >= 0.30
        else "Normal"
    )
    confidence = "High" if coverage_pct >= 80 else "Medium"
    return coverage_pct, risk_level, confidence


def fetch_all_market_conditions() -> dict[str, Any]:
    warnings: list[str] = []
    results: dict[str, dict[str, Any]] = {}

    fred_ok = bool(_fred_api_key())
    if not fred_ok:
        warnings.append("FRED_API_KEY not set — using public FRED CSV fallback where available.")

    # One shared market series keeps all trend signals on the same provider and as-of date.
    spy_closes = _download_close("SPY", period="1y")
    vix_closes = _download_close("^VIX", period="3mo")

    public_fetchers = (
        ("inverted_curve", _fetch_inverted_curve),
        ("sloos_tightening", _fetch_sloos),
        ("valuation_z", _fetch_valuation_z),
        ("high_yield_spread", _fetch_high_yield_spread),
        ("sahm_rule", _fetch_sahm_rule),
        ("spy_below_200dma", lambda: _fetch_spy_below_200dma(spy_closes)),
        ("spy_50_below_200dma", lambda: _fetch_spy_50_below_200dma(spy_closes)),
        ("spy_3m_drawdown", lambda: _fetch_spy_3m_drawdown(spy_closes)),
        ("vix_elevated", lambda: _fetch_vix_elevated(vix_closes)),
        ("low_minus_high_pe_6m", _fetch_low_minus_high_pe_6m),
    )
    with ThreadPoolExecutor(max_workers=min(6, len(public_fetchers))) as pool:
        futures = {pool.submit(fn): cid for cid, fn in public_fetchers}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                results[cid] = future.result()
            except Exception as exc:
                results[cid] = {
                    "value": None,
                    "state": "unknown",
                    "fetched": False,
                    "note": str(exc),
                }
                warnings.append(f"{cid}: {exc}")

    manual_predicates = {
        "cb_consumer_confidence": lambda v: v > 110,
        "cb_net_pct_stocks_higher": lambda v: v > 20,
        "sell_side_indicator": lambda v: v >= 1,
        "ltg_5yr_z": lambda v: v > 1,
        "mna_10yr_z": lambda v: v > 1,
        "credit_stress_indicator": lambda v: v < 0.25,
    }
    for cid, pred in manual_predicates.items():
        row = _fetch_manual(cid, pred)
        results[cid] = row
        if not row["fetched"]:
            warnings.append(row.get("note") or f"{cid}: manual value not set")

    condition_ids = [
        "spy_below_200dma",
        "spy_50_below_200dma",
        "spy_3m_drawdown",
        "vix_elevated",
        "cb_consumer_confidence",
        "cb_net_pct_stocks_higher",
        "sell_side_indicator",
        "ltg_5yr_z",
        "mna_10yr_z",
        "valuation_z",
        "low_minus_high_pe_6m",
        "inverted_curve",
        "high_yield_spread",
        "credit_stress_indicator",
        "sloos_tightening",
        "sahm_rule",
    ]

    rows = []
    fetched_count = 0
    unknown_count = 0
    triggered_count = 0
    for cid in condition_ids:
        row = results.get(cid, {"value": None, "state": "unknown", "fetched": False})
        if row.get("fetched"):
            fetched_count += 1
        if row.get("state") == "unknown":
            unknown_count += 1
        if row.get("state") == "triggered":
            triggered_count += 1
        rows.append(
            {
                "id": cid,
                "value": row.get("value"),
                "state": row.get("state", "unknown"),
                "fetched": bool(row.get("fetched")),
                "note": row.get("note"),
            }
        )

    coverage_pct, risk_level, confidence = _risk_assessment(
        fetched_count, triggered_count, len(condition_ids)
    )
    if confidence == "Insufficient":
        warnings.append("Market-risk assessment withheld because fewer than half of the indicators have data.")

    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "conditions": rows,
        "meta": {
            "fredConfigured": fred_ok,
            "fetchedCount": fetched_count,
            "unknownCount": unknown_count,
            "triggeredCount": triggered_count,
            "coveragePct": coverage_pct,
            "riskLevel": risk_level,
            "confidence": confidence,
            "warnings": warnings,
        },
    }
