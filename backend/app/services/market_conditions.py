from __future__ import annotations

import datetime as dt
import os
import re
import statistics
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import yfinance as yf

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

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
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not set")
    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
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
            html = client.get(url).text
    except Exception:
        return {}
    out: dict[str, float] = {}
    for m in re.finditer(
        r"([A-Z][a-z]{2})\s+\d+,\s+(\d{4})</td>\s*<td[^>]*>\s*([\d.]+)",
        html,
    ):
        mon, year, val = m.group(1), m.group(2), m.group(3)
        try:
            month_num = dt.datetime.strptime(mon, "%b").month
            out[f"{year}-{month_num:02d}"] = float(val)
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
        hist = yf.Ticker(sym).history(period="6mo", auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 2:
            return None
        closes = hist["Close"].dropna()
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


def fetch_all_market_conditions() -> dict[str, Any]:
    warnings: list[str] = []
    results: dict[str, dict[str, Any]] = {}

    fred_ok = bool(FRED_API_KEY)
    if not fred_ok:
        warnings.append("FRED_API_KEY not set — yield curve, SLOOS, and valuation z-score will be skipped.")

    if fred_ok:
        for cid, fn in (
            ("inverted_curve", _fetch_inverted_curve),
            ("sloos_tightening", _fetch_sloos),
            ("valuation_z", _fetch_valuation_z),
        ):
            try:
                results[cid] = fn()
            except Exception as exc:
                results[cid] = {
                    "value": None,
                    "state": "unknown",
                    "fetched": False,
                    "note": str(exc),
                }
                warnings.append(f"{cid}: {exc}")
    else:
        for cid in ("inverted_curve", "sloos_tightening", "valuation_z"):
            results[cid] = {
                "value": None,
                "state": "unknown",
                "fetched": False,
                "note": "FRED_API_KEY not configured",
            }

    try:
        results["low_minus_high_pe_6m"] = _fetch_low_minus_high_pe_6m()
    except Exception as exc:
        results["low_minus_high_pe_6m"] = {
            "value": None,
            "state": "unknown",
            "fetched": False,
            "note": str(exc),
        }

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
        "cb_consumer_confidence",
        "cb_net_pct_stocks_higher",
        "sell_side_indicator",
        "ltg_5yr_z",
        "mna_10yr_z",
        "valuation_z",
        "low_minus_high_pe_6m",
        "inverted_curve",
        "credit_stress_indicator",
        "sloos_tightening",
    ]

    rows = []
    fetched_count = 0
    unknown_count = 0
    for cid in condition_ids:
        row = results.get(cid, {"value": None, "state": "unknown", "fetched": False})
        if row.get("fetched"):
            fetched_count += 1
        if row.get("state") == "unknown":
            unknown_count += 1
        rows.append(
            {
                "id": cid,
                "value": row.get("value"),
                "state": row.get("state", "unknown"),
                "fetched": bool(row.get("fetched")),
                "note": row.get("note"),
            }
        )

    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "conditions": rows,
        "meta": {
            "fredConfigured": fred_ok,
            "fetchedCount": fetched_count,
            "unknownCount": unknown_count,
            "warnings": warnings,
        },
    }
