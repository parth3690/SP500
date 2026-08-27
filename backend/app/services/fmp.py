"""
Financial Modeling Prep (FMP) API — live quote overlay for research.

Dashboard: https://site.financialmodelingprep.com/developer/docs/dashboard
Stable quote docs: https://site.financialmodelingprep.com/developer/docs/stable/quote

Set FMP_API_KEY in the environment. When unset, research uses Yahoo last bar only.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Any, Optional

import httpx


def fmp_symbol(yahoo_or_display: str) -> str:
    """FMP expects exchange-style symbols; map Yahoo BRK.B → BRK-B."""
    return yahoo_or_display.strip().upper().replace(".", "-")


def fetch_fmp_quote(symbol: str, *, timeout: float = 12.0) -> Optional[dict[str, Any]]:
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        return None

    base = os.getenv("FMP_API_BASE", "https://financialmodelingprep.com/stable").rstrip("/")
    url = f"{base}/quote"

    try:
        resp = httpx.get(url, params={"symbol": symbol, "apikey": api_key}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if isinstance(data, list) and data:
        row = data[0]
        return row if isinstance(row, dict) else None
    if isinstance(data, dict) and data.get("symbol") is not None:
        return data
    return None


def _fetch_fmp_rows(endpoint: str, symbol: str) -> list[dict[str, Any]]:
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        return []
    base = os.getenv("FMP_API_BASE", "https://financialmodelingprep.com/stable").rstrip("/")
    params: dict[str, Any] = {"symbol": fmp_symbol(symbol), "apikey": api_key}
    if endpoint == "analyst-estimates":
        params.update({"period": "annual", "page": 0, "limit": 5})
    try:
        response = httpx.get(f"{base}/{endpoint}", params=params, timeout=15.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _percent_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * 100.0 if abs(value) <= 1.0 else value


def _latest_row(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: str(row.get("date") or row.get("filingDate") or row.get("acceptedDate") or ""),
        reverse=True,
    )[0]


def _recent_13f_quarters(today: Optional[date] = None, lookback: int = 8) -> list[tuple[int, int]]:
    current = today or date.today()
    quarter = (current.month - 1) // 3 + 1
    year = current.year
    out: list[tuple[int, int]] = []
    for _ in range(lookback):
        out.append((year, quarter))
        quarter -= 1
        if quarter == 0:
            quarter = 4
            year -= 1
    return out


def parse_fmp_institutional_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract Finviz-like institutional ownership and 13F share-change metrics."""
    latest = _latest_row(rows)
    if latest is None:
        return {
            "institutionalOwnershipPct": None,
            "institutionalTransactionPct": None,
            "institutionalSourceDate": None,
            "institutionalDataSource": None,
            "institutionalNotes": ["FMP institutional 13F data was unavailable."],
        }

    ownership = _percent_value(
        _first_float(
            latest,
            (
                "ownershipPercent",
                "institutionalOwnershipPercent",
                "institutionalOwnershipPercentage",
                "percentOfSharesOutstanding",
                "pctOfSharesOutstanding",
            ),
        )
    )
    transaction = _percent_value(
        _first_float(
            latest,
            (
                "numberOf13FsharesChangePercent",
                "sharesChangePct",
                "sharesChangePercent",
                "ownershipChangePct",
                "ownershipPercentChange",
                "institutionalOwnershipChangePercentage",
            ),
        )
    )

    current_shares = _first_float(latest, ("numberOf13Fshares", "sharesHeld", "totalShares"))
    previous_shares = _first_float(latest, ("lastNumberOf13Fshares", "previousNumberOf13Fshares"))
    changed_shares = _first_float(latest, ("numberOf13FsharesChange", "sharesChange", "changeInShares"))
    if transaction is None and changed_shares is not None and previous_shares and previous_shares > 0:
        transaction = changed_shares / previous_shares * 100.0
    if transaction is None and current_shares is not None and previous_shares and previous_shares > 0:
        transaction = (current_shares / previous_shares - 1.0) * 100.0

    current_holders = _first_float(latest, ("investorsHolding", "holderCount", "holders"))
    previous_holders = _first_float(latest, ("lastInvestorsHolding", "previousInvestorsHolding"))
    changed_holders = _first_float(latest, ("investorsHoldingChange", "holderCountChange"))
    if transaction is None and changed_holders is not None and previous_holders and previous_holders > 0:
        transaction = changed_holders / previous_holders * 100.0
    if transaction is None and current_holders is not None and previous_holders and previous_holders > 0:
        transaction = (current_holders / previous_holders - 1.0) * 100.0

    source_date = latest.get("date") or latest.get("filingDate") or latest.get("acceptedDate") or latest.get("_fmpQuarter")
    notes: list[str] = []
    if ownership is not None:
        notes.append(f"Institutional ownership {ownership:.1f}%.")
    if transaction is not None:
        notes.append(f"13F institutional share change {transaction:+.1f}%.")
    if source_date:
        notes.append(f"FMP 13F period {source_date}.")
    if transaction is None:
        notes.append("Institutional transaction/change field was unavailable.")

    return {
        "institutionalOwnershipPct": round(ownership, 2) if ownership is not None else None,
        "institutionalTransactionPct": round(transaction, 2) if transaction is not None else None,
        "institutionalSourceDate": str(source_date) if source_date else None,
        "institutionalDataSource": "FMP 13F" if ownership is not None or transaction is not None else None,
        "institutionalNotes": notes,
    }


def fetch_fmp_institutional_metrics(symbol: str) -> dict[str, Any]:
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        return {}

    base = os.getenv("FMP_API_BASE", "https://financialmodelingprep.com/stable").rstrip("/")
    for year, quarter in _recent_13f_quarters():
        params = {
            "symbol": fmp_symbol(symbol),
            "year": year,
            "quarter": quarter,
            "apikey": api_key,
        }
        try:
            response = httpx.get(
                f"{base}/institutional-ownership/symbol-positions-summary",
                params=params,
                timeout=15.0,
            )
            if response.status_code in (401, 402, 403):
                return {}
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue

        rows: list[dict[str, Any]]
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            nested = payload.get("data") or payload.get("results")
            rows = [row for row in nested if isinstance(row, dict)] if isinstance(nested, list) else [payload]
        else:
            rows = []
        if rows:
            for row in rows:
                row.setdefault("_fmpQuarter", f"{year}Q{quarter}")
            return parse_fmp_institutional_metrics(rows)
    return {}


def fetch_fmp_research_fundamentals(symbol: str) -> dict[str, Any]:
    """Fetch compact research fundamentals without relying on Yahoo metadata."""
    endpoints = ("profile", "ratios-ttm", "analyst-estimates")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {endpoint: pool.submit(_fetch_fmp_rows, endpoint, symbol) for endpoint in endpoints}
        payloads = {endpoint: future.result() for endpoint, future in futures.items()}

    profile = payloads["profile"][0] if payloads["profile"] else {}
    ratios = payloads["ratios-ttm"][0] if payloads["ratios-ttm"] else {}
    price = profile.get("price")
    forward_pe = None
    estimates = sorted(payloads["analyst-estimates"], key=lambda row: str(row.get("date", "")))
    future_estimates = [row for row in estimates if str(row.get("date", "")) >= date.today().isoformat()]
    estimate = future_estimates[0] if future_estimates else (estimates[-1] if estimates else {})
    try:
        eps = float(estimate.get("epsAvg"))
        if price is not None and eps > 0:
            forward_pe = float(price) / eps
    except (TypeError, ValueError, ZeroDivisionError):
        forward_pe = None

    return {
        "trailingPE": ratios.get("priceToEarningsRatioTTM"),
        "forwardPE": forward_pe,
        "marketCap": profile.get("marketCap"),
        "beta": profile.get("beta"),
        "dividendYield": ratios.get("dividendYieldTTM"),
        "source": "Financial Modeling Prep" if profile or ratios else None,
    }


def merge_fmp_live_into_research(
    payload: dict[str, Any],
    *,
    fmp_symbol_str: str,
    quote: Optional[dict[str, Any]] = None,
) -> None:
    """
    If FMP returns a quote, overlay header fields with live price / change / volume.
    Preserves Yahoo last-bar close as chartLastClose for chart context.
    """
    q = quote or fetch_fmp_quote(fmp_symbol_str)
    if not q:
        return

    price = q.get("price")
    if price is None:
        return
    try:
        price_f = float(price)
    except (TypeError, ValueError):
        return

    # Preserve pre-merge values (last daily close from Yahoo in selected range)
    payload["chartLastClose"] = round(float(payload.get("currentPrice", price_f)), 2)

    prev = q.get("previousClose")
    ch = q.get("change")
    chp = q.get("changePercentage")
    vol = q.get("volume")

    payload["currentPrice"] = round(price_f, 2)
    if prev is not None:
        try:
            payload["previousClose"] = round(float(prev), 2)
        except (TypeError, ValueError):
            pass
    if ch is not None:
        try:
            payload["change"] = round(float(ch), 2)
        except (TypeError, ValueError):
            pass
    if chp is not None:
        try:
            payload["changePct"] = round(float(chp), 2)
        except (TypeError, ValueError):
            pass
    elif prev is not None:
        try:
            p = float(prev)
            if p != 0:
                payload["changePct"] = round(((price_f - p) / p) * 100.0, 2)
        except (TypeError, ValueError):
            pass

    if vol is not None:
        try:
            payload["volume"] = int(float(vol))
        except (TypeError, ValueError):
            pass

    ts = q.get("timestamp")
    as_of: Optional[str] = None
    if ts is not None:
        try:
            as_of = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            pass

    payload["liveQuote"] = {
        "source": "Financial Modeling Prep",
        "providerUrl": "https://site.financialmodelingprep.com/developer/docs",
        "symbol": q.get("symbol"),
        "asOf": as_of,
    }
