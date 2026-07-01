from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from .cache import MULTIBAGGER_CACHE, cache_get, cache_set
from .sp500 import get_sp500_constituents_cached, normalize_yahoo_ticker

CRITERIA: dict[str, Any] = {
    "min_market_cap": 1_000_000_000,
    "max_peg": 1.0,
    "pe_below_sector": True,
    "min_roe": 0.20,
    "min_roic": 0.15,
    "max_debt_to_equity": 0.5,
    "min_insider_holding": 0.15,
    "min_revenue_growth": 0.15,
    "min_earnings_growth": 0.15,
    "min_operating_margin": 0.15,
    "max_price_to_sales": 10.0,
    "max_ev_ebitda": 25.0,
}

SOFT_CHECKS = {"PEG", "ROIC", "InsiderHolding", "RevenueGrowth", "EarningsGrowth"}

CRITERION_META: list[dict[str, str]] = [
    {"id": "MarketCap", "name": "Market cap", "threshold": "> $1B"},
    {"id": "PEG", "name": "PEG ratio", "threshold": "< 1.0"},
    {"id": "PE<Sector", "name": "P/E vs sector", "threshold": "P/E < sector median"},
    {"id": "ROE", "name": "Return on equity", "threshold": "> 20%"},
    {"id": "ROIC", "name": "ROIC (ROA proxy)", "threshold": "> 15%"},
    {"id": "DebtEquity", "name": "Debt / equity", "threshold": "< 0.5"},
    {"id": "InsiderHolding", "name": "Insider holding", "threshold": "> 15%"},
    {"id": "RevenueGrowth", "name": "Revenue growth", "threshold": "> 15%"},
    {"id": "EarningsGrowth", "name": "Earnings growth", "threshold": "> 15%"},
    {"id": "OperatingMargin", "name": "Operating margin", "threshold": "> 15%"},
    {"id": "PriceToSales", "name": "Price / sales", "threshold": "< 10"},
    {"id": "EV/EBITDA", "name": "EV / EBITDA", "threshold": "< 25"},
]


def _safe(d: dict[str, Any], key: str) -> Any:
    v = d.get(key)
    if v in (None, "Infinity", "None"):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _deep_metrics(stock: yf.Ticker) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {"roe_5y": None, "rev_cagr_3y": None, "eps_cagr_5y": None}
    try:
        inc = stock.financials
        bs = stock.balance_sheet
        if inc is None or inc.empty:
            return out

        def row(df: pd.DataFrame | None, *names: str) -> pd.Series | None:
            if df is None:
                return None
            for n in names:
                if n in df.index:
                    return df.loc[n].dropna()
            return None

        rev = row(inc, "Total Revenue")
        ni = row(inc, "Net Income", "Net Income Common Stockholders")
        eq = row(bs, "Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity")

        if ni is not None and eq is not None:
            roes: list[float] = []
            for col in ni.index:
                if col in eq.index and eq[col] not in (0, None):
                    try:
                        roes.append(float(ni[col]) / float(eq[col]))
                    except (TypeError, ZeroDivisionError):
                        pass
            if roes:
                out["roe_5y"] = statistics.mean(roes)

        def cagr(series: pd.Series) -> Optional[float]:
            s = series.dropna()
            if len(s) < 2:
                return None
            newest, oldest = float(s.iloc[0]), float(s.iloc[-1])
            yrs = len(s) - 1
            if oldest <= 0 or newest <= 0:
                return None
            return (newest / oldest) ** (1 / yrs) - 1

        if rev is not None:
            out["rev_cagr_3y"] = cagr(rev)
        if ni is not None:
            out["eps_cagr_5y"] = cagr(ni)
    except Exception:
        pass
    return out


def fetch_metrics(ticker: str, *, deep: bool = False) -> Optional[dict[str, Any]]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        if not info or _safe(info, "marketCap") is None:
            return None

        de = _safe(info, "debtToEquity")
        if de is not None:
            de = de / 100.0

        metrics = {
            "ticker": ticker.upper(),
            "name": _safe(info, "shortName") or ticker.upper(),
            "sector": _safe(info, "sector") or "Unknown",
            "marketCap": _safe(info, "marketCap"),
            "pe": _safe(info, "trailingPE"),
            "peg": _safe(info, "trailingPegRatio") or _safe(info, "pegRatio"),
            "roe": _safe(info, "returnOnEquity"),
            "debtToEquity": de,
            "insider": _safe(info, "heldPercentInsiders"),
            "revGrowth": _safe(info, "revenueGrowth"),
            "earnGrowth": _safe(info, "earningsGrowth"),
            "opMargin": _safe(info, "operatingMargins"),
            "priceToSales": _safe(info, "priceToSalesTrailing12Months"),
            "evEbitda": _safe(info, "enterpriseToEbitda"),
            "roic": _safe(info, "returnOnAssets"),
        }

        if deep:
            d = _deep_metrics(stock)
            if d["roe_5y"] is not None:
                metrics["roe"] = d["roe_5y"]
            if d["rev_cagr_3y"] is not None:
                metrics["revGrowth"] = d["rev_cagr_3y"]
            if d["eps_cagr_5y"] is not None:
                metrics["earnGrowth"] = d["eps_cagr_5y"]
        return metrics
    except Exception:
        return None


def _sector_pe_median(sector: str) -> Optional[float]:
    cache_key = ("sector_pe", sector)
    cached = cache_get(MULTIBAGGER_CACHE, cache_key)
    if cached is not None:
        return cached

    constituents = get_sp500_constituents_cached()
    sector_tickers = [
        normalize_yahoo_ticker(c.ticker)
        for c in constituents
        if c.sector == sector
    ]
    pes: list[float] = []
    for sym in sector_tickers:
        try:
            info = yf.Ticker(sym).info or {}
            pe = _safe(info, "trailingPE")
            if pe is not None and pe > 0:
                pes.append(float(pe))
        except Exception:
            continue
        time.sleep(0.05)

    median = statistics.median(pes) if pes else None
    cache_set(MULTIBAGGER_CACHE, cache_key, median)
    return median


def _fmt_pct(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    return f"{v * 100:.1f}%"


def _fmt_money_b(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    return f"${v / 1e9:.2f}B"


def _fmt_num(v: Optional[float], digits: int = 2) -> Optional[str]:
    if v is None:
        return None
    return f"{v:.{digits}f}"


def _metric_value_display(metric_id: str, m: dict[str, Any], sector_pe: Optional[float]) -> Optional[str]:
    mapping = {
        "MarketCap": lambda: _fmt_money_b(m.get("marketCap")),
        "PEG": lambda: _fmt_num(m.get("peg")),
        "PE<Sector": lambda: (
            f"{_fmt_num(m.get('pe'))} vs sector {_fmt_num(sector_pe)}"
            if m.get("pe") is not None and sector_pe is not None
            else _fmt_num(m.get("pe"))
        ),
        "ROE": lambda: _fmt_pct(m.get("roe")),
        "ROIC": lambda: _fmt_pct(m.get("roic")),
        "DebtEquity": lambda: _fmt_num(m.get("debtToEquity")),
        "InsiderHolding": lambda: _fmt_pct(m.get("insider")),
        "RevenueGrowth": lambda: _fmt_pct(m.get("revGrowth")),
        "EarningsGrowth": lambda: _fmt_pct(m.get("earnGrowth")),
        "OperatingMargin": lambda: _fmt_pct(m.get("opMargin")),
        "PriceToSales": lambda: _fmt_num(m.get("priceToSales")),
        "EV/EBITDA": lambda: _fmt_num(m.get("evEbitda")),
    }
    fn = mapping.get(metric_id)
    return fn() if fn else None


def evaluate_ticker(metrics: dict[str, Any], sector_pe_median: dict[str, Optional[float]]) -> dict[str, Any]:
    green: list[str] = []
    fails: list[str] = []
    skipped: list[str] = []

    def check(name: str, value: Any, ok: bool) -> None:
        if value is None:
            (skipped if name in SOFT_CHECKS else fails).append(name)
            return
        (green if ok else fails).append(name)

    m = metrics
    check("MarketCap", m["marketCap"], m["marketCap"] and m["marketCap"] > CRITERIA["min_market_cap"])
    check("PEG", m["peg"], m["peg"] is not None and m["peg"] < CRITERIA["max_peg"])
    check("ROE", m["roe"], m["roe"] is not None and m["roe"] > CRITERIA["min_roe"])
    check("ROIC", m["roic"], m["roic"] is not None and m["roic"] > CRITERIA["min_roic"])
    check(
        "DebtEquity",
        m["debtToEquity"],
        m["debtToEquity"] is not None and m["debtToEquity"] < CRITERIA["max_debt_to_equity"],
    )
    check(
        "InsiderHolding",
        m["insider"],
        m["insider"] is not None and m["insider"] > CRITERIA["min_insider_holding"],
    )
    check(
        "RevenueGrowth",
        m["revGrowth"],
        m["revGrowth"] is not None and m["revGrowth"] > CRITERIA["min_revenue_growth"],
    )
    check(
        "EarningsGrowth",
        m["earnGrowth"],
        m["earnGrowth"] is not None and m["earnGrowth"] > CRITERIA["min_earnings_growth"],
    )
    check(
        "OperatingMargin",
        m["opMargin"],
        m["opMargin"] is not None and m["opMargin"] > CRITERIA["min_operating_margin"],
    )
    check(
        "PriceToSales",
        m["priceToSales"],
        m["priceToSales"] is not None and m["priceToSales"] < CRITERIA["max_price_to_sales"],
    )
    check(
        "EV/EBITDA",
        m["evEbitda"],
        m["evEbitda"] is not None and m["evEbitda"] < CRITERIA["max_ev_ebitda"],
    )

    sector_med = sector_pe_median.get(m["sector"])
    if CRITERIA["pe_below_sector"]:
        if m["pe"] is not None and sector_med is None:
            skipped.append("PE<Sector")
        else:
            check(
                "PE<Sector",
                m["pe"],
                m["pe"] is not None and sector_med is not None and m["pe"] < sector_med,
            )

    criteria_rows = []
    for meta in CRITERION_META:
        cid = meta["id"]
        if cid == "PE<Sector" and not CRITERIA["pe_below_sector"]:
            continue
        if cid in green:
            status = "pass"
        elif cid in fails:
            status = "fail"
        elif cid in skipped:
            status = "skip"
        else:
            status = "skip"
        criteria_rows.append(
            {
                "id": cid,
                "name": meta["name"],
                "threshold": meta["threshold"],
                "valueDisplay": _metric_value_display(cid, m, sector_med),
                "status": status,
                "soft": cid in SOFT_CHECKS,
            }
        )

    n_green = len(green)
    n_total = len(criteria_rows)
    passed_all = len(fails) == 0 and len(skipped) == 0

    return {
        "green": green,
        "fails": fails,
        "skipped": skipped,
        "nGreen": n_green,
        "nTotal": n_total,
        "passedAll": passed_all,
        "criteria": criteria_rows,
        "sectorPeMedian": sector_med,
    }


def scan_ticker(ticker: str, *, deep: bool = False) -> dict[str, Any]:
    sym = normalize_yahoo_ticker(ticker.strip().upper())
    if not sym:
        raise ValueError("Ticker is required")

    metrics = fetch_metrics(sym, deep=deep)
    if metrics is None:
        raise LookupError(f"No fundamental data found for {sym}")

    sector_pe = _sector_pe_median(metrics["sector"])
    sector_map = {metrics["sector"]: sector_pe}
    score = evaluate_ticker(metrics, sector_map)

    return {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "ticker": metrics["ticker"],
        "name": metrics["name"],
        "sector": metrics["sector"],
        "deep": deep,
        "metrics": {
            "marketCap": metrics["marketCap"],
            "pe": metrics["pe"],
            "peg": metrics["peg"],
            "roe": metrics["roe"],
            "roic": metrics["roic"],
            "debtToEquity": metrics["debtToEquity"],
            "insider": metrics["insider"],
            "revGrowth": metrics["revGrowth"],
            "earnGrowth": metrics["earnGrowth"],
            "opMargin": metrics["opMargin"],
            "priceToSales": metrics["priceToSales"],
            "evEbitda": metrics["evEbitda"],
        },
        "sectorPeMedian": sector_pe,
        "score": score,
        "criteria": score["criteria"],
        "nGreen": score["nGreen"],
        "nTotal": score["nTotal"],
        "passedAll": score["passedAll"],
        "green": score["green"],
        "fails": score["fails"],
        "skipped": score["skipped"],
    }
