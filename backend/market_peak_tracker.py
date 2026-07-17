#!/usr/bin/env python3
"""
Market Peak Signpost Tracker
============================================================
A reconstruction of the Bank of America style "signals to watch for a
market peak" checklist. It tracks 10 signposts across three categories
(Sentiment, Valuation, Macro) and reports how many are currently
"triggered", then compares that against prior S&P 500 peaks.

WHAT IT CAN AND CANNOT FETCH
----------------------------------------------------------------
Only a few of the original signposts have a free public data feed.
The rest are proprietary (BofA, Conference Board, IBES) and must be
supplied manually from a terminal or the research report itself.

  FETCHABLE (FRED, free API key):
    - Inverted yield curve              -> series T10Y2Y
    - Tightening credit (SLOOS)         -> series DRTSCILM
    - CPI half of the valuation z-score -> series CPIAUCSL

  COMPUTED (best effort, needs a PE source):
    - Trailing PE + YoY CPI z-score     -> multpl.com + FRED

  MANUAL / PROPRIETARY (drop the latest reading into MANUAL_OVERRIDES):
    - Conf Board Consumer Confidence > 110
    - Conf Board Net % Expecting Stocks Higher > 20
    - Sell Side Indicator "Sell"
    - S&P 500 LTG 5yr Z score > 1
    - 10yr Z score of M&A deals (3m sum) > 1
    - Low PE underperforms High PE by 2.5ppt over 6m
    - BofA Credit Stress Indicator < 0.25

This is an educational reconstruction, not investment advice, and it does
not reproduce BofA's proprietary indicators exactly.

USAGE
----------------------------------------------------------------
    export FRED_API_KEY=your_key_here      # free at fredstlouisfed.org
    python market_peak_tracker.py          # live
    python market_peak_tracker.py --demo   # offline sample data
"""

from __future__ import annotations

import os
import sys
import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import requests
except ImportError:
    sys.exit("Please `pip install requests` first.")

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


# ----------------------------------------------------------------------
# Manual / proprietary readings.
# Fill these in with the LATEST value from your data source. Set to None
# to leave a signal as "unknown" (it will be excluded from % triggered).
# ----------------------------------------------------------------------
MANUAL_OVERRIDES: dict[str, Optional[float]] = {
    "cb_consumer_confidence": None,   # index level, e.g. 112.0  (trigger > 110)
    "cb_net_pct_stocks_higher": None, # percent,      e.g. 22.5  (trigger > 20)
    "sell_side_indicator": None,      # 1 if "Sell" signal active else 0
    "ltg_5yr_z": None,                # z score,       e.g. 1.4   (trigger > 1)
    "mna_10yr_z": None,               # z score,       e.g. 0.8   (trigger > 1)
    "low_minus_high_pe_6m": None,     # ppt, low-PE minus high-PE return over 6m
                                      #   (trigger when low underperforms by >= 2.5,
                                      #    i.e. value <= -2.5)
    "credit_stress_indicator": None,  # level,         e.g. 0.18  (trigger < 0.25)
}


# ----------------------------------------------------------------------
# Historical reference (transcribed from the chart). The per-cell trigger
# matrix for older peaks is approximate; the % triggered and S&P levels
# are read directly off the chart and are reliable.
# ----------------------------------------------------------------------
HISTORY = {
    #  peak       % triggered   S&P 500 level
    "Jul-90": (0.88, 369),
    "Mar-00": (0.90, 1527),
    "Oct-07": (0.80, 1565),
    "Sep-18": (0.60, 2931),
    "Feb-20": (0.50, 3386),
    "Jan-22": (0.50, 4797),
    "Feb-25": (0.70, 6144),
    # last three months from the chart
    "Mar-26": (0.40, 6529),
    "Apr-26": (0.50, 7209),
    "May-26": (0.70, 7580),
}


# ----------------------------------------------------------------------
# FRED helpers
# ----------------------------------------------------------------------
def fred_series(series_id: str, start: Optional[str] = None) -> list[tuple[str, float]]:
    """Return [(date, value), ...] for a FRED series, skipping missing values."""
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not set")
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    out = []
    for obs in r.json().get("observations", []):
        v = obs.get("value", ".")
        if v not in (".", "", None):
            try:
                out.append((obs["date"], float(v)))
            except ValueError:
                pass
    return out


def _zscore(series: list[float], window: int) -> Optional[float]:
    """z-score of the latest point vs the trailing `window` observations."""
    if len(series) < window + 1:
        window = len(series) - 1
    if window < 12:
        return None
    sample = series[-(window + 1):-1]
    mean = sum(sample) / len(sample)
    var = sum((x - mean) ** 2 for x in sample) / len(sample)
    sd = var ** 0.5
    if sd == 0:
        return None
    return (series[-1] - mean) / sd


# ----------------------------------------------------------------------
# Evaluators. Each returns True (triggered), False (not), or None (unknown).
# `demo` swaps in canned values so the script runs with no network.
# ----------------------------------------------------------------------
def eval_inverted_curve(demo: bool) -> Optional[bool]:
    """Triggered if the 10y-2y spread inverted at any point in the prior 6 months."""
    if demo:
        return True
    start = (dt.date.today() - dt.timedelta(days=200)).isoformat()
    data = fred_series("T10Y2Y", start=start)
    return any(v < 0 for _, v in data) if data else None


def eval_sloos_tightening(demo: bool) -> Optional[bool]:
    """Triggered if the latest SLOOS net % of banks tightening C&I standards > 0."""
    if demo:
        return True
    data = fred_series("DRTSCILM")
    return (data[-1][1] > 0) if data else None


def eval_valuation_z(demo: bool) -> Optional[bool]:
    """
    Triggered if the 10yr (120-month) z-score of (trailing S&P 500 PE + YoY CPI) > 1.
    Trailing PE comes from multpl.com; CPI from FRED. Best effort.
    """
    if demo:
        return True
    try:
        cpi = fred_series("CPIAUCSL", start="2000-01-01")
        # monthly YoY CPI
        yoy = {}
        idx = {d[:7]: v for d, v in cpi}
        keys = sorted(idx)
        for k in keys:
            y, m = int(k[:4]), int(k[5:7])
            prev = f"{y-1:04d}-{m:02d}"
            if prev in idx and idx[prev]:
                yoy[k] = (idx[k] / idx[prev] - 1) * 100
        pe = _fetch_multpl_pe()  # {YYYY-MM: trailing PE}
        if not pe:
            return None
        combined = []
        for k in sorted(set(pe) & set(yoy)):
            combined.append(pe[k] + yoy[k])
        z = _zscore(combined, 120)
        return (z is not None and z > 1)
    except Exception:
        return None


def _fetch_multpl_pe() -> dict[str, float]:
    """Scrape monthly trailing S&P 500 PE from multpl.com. Returns {YYYY-MM: PE}."""
    url = "https://www.multpl.com/s-p-500-pe-ratio/table/by-month"
    try:
        html = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text
    except Exception:
        return {}
    out: dict[str, float] = {}
    import re
    # rows look like: <td ...>Jan 1, 2020</td><td ...>24.88</td>
    for m in re.finditer(
        r"([A-Z][a-z]{2})\s+\d+,\s+(\d{4})</td>\s*<td[^>]*>\s*([\d.]+)", html
    ):
        mon, year, val = m.group(1), m.group(2), m.group(3)
        try:
            month_num = dt.datetime.strptime(mon, "%b").month
            out[f"{year}-{month_num:02d}"] = float(val)
        except ValueError:
            continue
    return out


def make_manual_eval(key: str, predicate: Callable[[float], bool]) -> Callable[[bool], Optional[bool]]:
    def _eval(demo: bool) -> Optional[bool]:
        val = MANUAL_OVERRIDES.get(key)
        if val is None:
            return None
        return predicate(val)
    return _eval


# ----------------------------------------------------------------------
# Signpost definitions
# ----------------------------------------------------------------------
@dataclass
class Signpost:
    key: str
    name: str
    category: str          # Sentiment | Valuation | Macro
    source: str            # FRED | Computed | Manual
    evaluate: Callable[[bool], Optional[bool]]


SIGNPOSTS: list[Signpost] = [
    Signpost("cb_consumer_confidence",
             "Conf Board Consumer Confidence > 110 (prior 6m)", "Sentiment", "Manual",
             make_manual_eval("cb_consumer_confidence", lambda v: v > 110)),
    Signpost("cb_net_pct_stocks_higher",
             "Conf Board: Net % Expecting Stocks Higher > 20", "Sentiment", "Manual",
             make_manual_eval("cb_net_pct_stocks_higher", lambda v: v > 20)),
    Signpost("sell_side_indicator",
             'Sell Side Indicator: "Sell" signal triggered', "Sentiment", "Manual",
             make_manual_eval("sell_side_indicator", lambda v: v >= 1)),
    Signpost("ltg_5yr_z",
             "S&P 500 LT growth expectations (LTG): 5yr Z > 1", "Sentiment", "Manual",
             make_manual_eval("ltg_5yr_z", lambda v: v > 1)),
    Signpost("mna_10yr_z",
             "10yr Z of # of M&A deals (3m sum) > 1", "Sentiment", "Manual",
             make_manual_eval("mna_10yr_z", lambda v: v > 1)),
    Signpost("valuation_z",
             "10yr Z of (trailing S&P 500 PE + YoY CPI) > 1", "Valuation", "Computed",
             eval_valuation_z),
    Signpost("low_minus_high_pe_6m",
             "Low PE underperforms High PE by 2.5ppt over 6m", "Valuation", "Manual",
             make_manual_eval("low_minus_high_pe_6m", lambda v: v <= -2.5)),
    Signpost("inverted_curve",
             "Inverted yield curve (prior 6m)", "Macro", "FRED",
             eval_inverted_curve),
    Signpost("credit_stress_indicator",
             "Credit Stress Indicator drops below 0.25", "Macro", "Manual",
             make_manual_eval("credit_stress_indicator", lambda v: v < 0.25)),
    Signpost("sloos_tightening",
             "Tightening credit conditions (SLOOS)", "Macro", "FRED",
             eval_sloos_tightening),
]


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def build_scorecard(demo: bool) -> list[tuple[Signpost, Optional[bool]]]:
    results = []
    for sp in SIGNPOSTS:
        try:
            state = sp.evaluate(demo)
        except Exception:
            state = None
        results.append((sp, state))
    return results


def render(results: list[tuple[Signpost, Optional[bool]]]) -> None:
    mark = {True: "TRIGGERED", False: "  -  ", None: " n/a "}
    print("\n" + "=" * 78)
    print(" MARKET PEAK SIGNPOST TRACKER".center(78))
    print(" (BofA-style reconstruction — not investment advice)".center(78))
    print("=" * 78)
    print(f"{'Signpost':<52}{'Category':<11}{'State'}")
    print("-" * 78)
    for sp, state in results:
        print(f"{sp.name[:51]:<52}{sp.category:<11}{mark[state]}")
    print("-" * 78)

    known = [s for _, s in results if s is not None]
    triggered = sum(1 for s in known if s)
    pct = (triggered / len(known) * 100) if known else 0.0
    unknown = sum(1 for _, s in results if s is None)
    print(f"% triggered (of {len(known)} known signals): {pct:.0f}%"
          f"   |   {unknown} unknown/manual not set")

    print("\nHistorical reference (% triggered at prior peaks):")
    line = "  ".join(f"{k} {int(v[0]*100)}%" for k, v in HISTORY.items())
    print("  " + line)
    print("=" * 78)
    print("Note: 'manual'/proprietary signals stay n/a until you set MANUAL_OVERRIDES.")
    print("Set FRED_API_KEY for the live yield-curve, SLOOS and CPI feeds.\n")


def main() -> None:
    demo = "--demo" in sys.argv
    if demo:
        print("[demo mode: using canned values, no network calls]")
    elif not FRED_API_KEY:
        print("[no FRED_API_KEY set: public signals will show n/a. "
              "Run with --demo to see sample output.]")
    render(build_scorecard(demo))


if __name__ == "__main__":
    main()
