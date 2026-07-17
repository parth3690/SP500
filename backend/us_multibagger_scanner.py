#!/usr/bin/env python3
"""
US Market Fundamental Scanner — "multibagger-style" screen.

Adapted from an Indian-market checklist for US equities. It keeps looping on a
fixed interval, re-pulling fundamentals for a universe of US stocks and printing
the names that pass ALL active criteria.

DATA SOURCE: Yahoo Finance via the free `yfinance` library.
  -> Run this on YOUR machine / a server with internet. Yahoo Finance is NOT
     reachable from restricted sandboxes.

INSTALL:
    pip install yfinance pandas lxml

RUN:
    python us_multibagger_scanner.py --once                 # single scan, S&P 500
    python us_multibagger_scanner.py                        # loop every 24h
    python us_multibagger_scanner.py --interval 3600        # loop every 1h
    python us_multibagger_scanner.py --tickers AAPL,MSFT,NVDA
    python us_multibagger_scanner.py --file my_tickers.csv  # one ticker per line
    python us_multibagger_scanner.py --deep                 # also compute true
                                                            # 5y averages / CAGRs
                                                            # (slower, more API calls)

NOTE: This is a research/screening tool, not investment advice. Passing a screen
does not mean a stock will go up. Verify every number before acting on it.
"""

import argparse
import statistics
import sys
import time
from datetime import datetime

import pandas as pd
import yfinance as yf


# ----------------------------------------------------------------------------
# CRITERIA  (edit these freely — every threshold lives here)
# ----------------------------------------------------------------------------
CRITERIA = {
    "min_market_cap":        1_000_000_000,  # $1B
    "max_peg":               1.0,
    "pe_below_sector":       True,           # PE < sector median PE (this run)
    "min_roe":               0.20,           # 20%
    "min_roic":              0.15,           # 15%  (US stand-in for ROCE)
    "max_debt_to_equity":    0.5,
    "min_insider_holding":   0.15,           # 15%  (US stand-in for promoter holding)
    "min_revenue_growth":    0.15,           # 15%
    "min_earnings_growth":   0.15,           # 15%
    "min_operating_margin":  0.15,           # OPM > 15%
    "max_price_to_sales":    10.0,
    "max_ev_ebitda":         25.0,
}

# A check is "soft" when data is commonly missing on Yahoo. A missing soft metric
# does NOT fail the stock (it is skipped with a note); a missing hard metric does.
SOFT_CHECKS = {"PEG", "ROIC", "InsiderHolding", "RevenueGrowth", "EarningsGrowth"}

# Tiny fallback universe if S&P 500 fetch fails and you pass no tickers.
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "AVGO", "LLY", "V", "MA",
    "UNH", "COST", "ADBE", "CRM", "AMD", "ASML", "NOW", "INTU", "ANET", "MELI",
]


# ----------------------------------------------------------------------------
# DATA FETCH
# ----------------------------------------------------------------------------
def get_sp500_tickers():
    """Pull the current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url)[0]
        return [t.replace(".", "-") for t in table["Symbol"].tolist()]
    except Exception as e:
        print(f"  ! Could not fetch S&P 500 list ({e}); using fallback universe.")
        return FALLBACK_UNIVERSE


def _clean_symbols(syms):
    """Drop test issues, warrants/units/preferreds, and yfinance-friendly format."""
    out = []
    for s in syms:
        s = s.strip().upper()
        if not s or any(c in s for c in "$^"):
            continue
        # skip obvious non-common-stock suffixes (warrants 'W', units 'U', rights 'R'
        # on 5-letter NASDAQ symbols) — heuristic, keeps it simple
        if len(s) == 5 and s[-1] in "WUR":
            continue
        out.append(s.replace(".", "-"))
    return sorted(set(out))


def get_all_us_tickers():
    """Every common stock on NASDAQ + NYSE + AMEX.
    Primary: NASDAQ Trader symbol files (authoritative, updated daily).
    Fallback: a GitHub mirror (works behind restrictive networks)."""
    # --- primary: NASDAQ Trader ---
    try:
        base = "https://www.nasdaqtrader.com/dynamic/SymDir/"
        nd = pd.read_csv(base + "nasdaqlisted.txt", sep="|")
        ot = pd.read_csv(base + "otherlisted.txt", sep="|")
        nd = nd[(nd["Test Issue"] == "N") & (nd["ETF"] == "N")]
        ot = ot[(ot["Test Issue"] == "N") & (ot["ETF"] == "N")]
        syms = list(nd["Symbol"]) + list(ot["ACT Symbol"])
        syms = _clean_symbols(syms)
        if len(syms) > 1000:
            print(f"  Loaded {len(syms)} US tickers from NASDAQ Trader.")
            return syms
    except Exception as e:
        print(f"  ! NASDAQ Trader fetch failed ({e}); trying GitHub mirror.")

    # --- fallback: GitHub mirror ---
    try:
        import urllib.request
        root = ("https://raw.githubusercontent.com/rreichel3/"
                "US-Stock-Symbols/main")
        syms = []
        for ex in ("nasdaq", "nyse", "amex"):
            url = f"{root}/{ex}/{ex}_tickers.txt"
            txt = urllib.request.urlopen(url, timeout=30).read().decode()
            syms += txt.splitlines()
        syms = _clean_symbols(syms)
        print(f"  Loaded {len(syms)} US tickers from GitHub mirror.")
        return syms
    except Exception as e:
        print(f"  ! All-US fetch failed ({e}); using fallback universe.")
        return FALLBACK_UNIVERSE


def safe(d, key):
    v = d.get(key)
    if v in (None, "Infinity", "None"):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def deep_metrics(stock):
    """Best-effort TRUE multi-year ROE avg + revenue/earnings CAGR from statements.
    Returns dict; any value may be None when statements are too sparse."""
    out = {"roe_5y": None, "rev_cagr_3y": None, "eps_cagr_5y": None}
    try:
        inc = stock.financials                       # annual income statement
        bs = stock.balance_sheet                     # annual balance sheet
        if inc is None or inc.empty:
            return out

        def row(df, *names):
            for n in names:
                if df is not None and n in df.index:
                    return df.loc[n].dropna()
            return None

        rev = row(inc, "Total Revenue")
        ni = row(inc, "Net Income", "Net Income Common Stockholders")
        eq = row(bs, "Stockholders Equity", "Total Stockholder Equity",
                 "Common Stock Equity")

        # multi-year average ROE
        if ni is not None and eq is not None:
            roes = []
            for col in ni.index:
                if col in eq.index and eq[col] not in (0, None):
                    try:
                        roes.append(float(ni[col]) / float(eq[col]))
                    except (TypeError, ZeroDivisionError):
                        pass
            if roes:
                out["roe_5y"] = statistics.mean(roes)

        # CAGR helper (columns are newest-first in yfinance)
        def cagr(series):
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


def fetch(ticker, deep=False):
    """Return a flat metrics dict for one ticker, or None on hard failure."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        if not info or safe(info, "marketCap") is None:
            return None

        de = safe(info, "debtToEquity")
        if de is not None:
            de = de / 100.0  # yfinance reports D/E as a percentage

        m = {
            "ticker":        ticker,
            "name":          safe(info, "shortName") or ticker,
            "sector":        safe(info, "sector") or "Unknown",
            "marketCap":     safe(info, "marketCap"),
            "pe":            safe(info, "trailingPE"),
            "peg":           safe(info, "trailingPegRatio") or safe(info, "pegRatio"),
            "roe":           safe(info, "returnOnEquity"),
            "debtToEquity":  de,
            "insider":       safe(info, "heldPercentInsiders"),
            "revGrowth":     safe(info, "revenueGrowth"),
            "earnGrowth":    safe(info, "earningsGrowth"),
            "opMargin":      safe(info, "operatingMargins"),
            "priceToSales":  safe(info, "priceToSalesTrailing12Months"),
            "evEbitda":      safe(info, "enterpriseToEbitda"),
            # ROIC isn't in .info — approximate with return on assets as a proxy.
            "roic":          safe(info, "returnOnAssets"),
        }

        if deep:
            d = deep_metrics(stock)
            if d["roe_5y"] is not None:
                m["roe"] = d["roe_5y"]
            if d["rev_cagr_3y"] is not None:
                m["revGrowth"] = d["rev_cagr_3y"]
            if d["eps_cagr_5y"] is not None:
                m["earnGrowth"] = d["eps_cagr_5y"]
        return m
    except Exception:
        return None


# ----------------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------------
def evaluate(m, sector_pe_median):
    """Return a scorecard dict:
       green   -> list of criteria the stock PASSES
       fails   -> hard criteria it misses
       skipped -> soft criteria with no data (don't count for/against)
       passed_all -> True only if no fails AND nothing skipped
    """
    green, fails, skipped = [], [], []

    def check(name, value, ok):
        if value is None:
            (skipped if name in SOFT_CHECKS else fails).append(name)
            return
        (green if ok else fails).append(name)

    check("MarketCap",      m["marketCap"],    m["marketCap"] and m["marketCap"] > CRITERIA["min_market_cap"])
    check("PEG",            m["peg"],          m["peg"] is not None and m["peg"] < CRITERIA["max_peg"])
    check("ROE",            m["roe"],          m["roe"] is not None and m["roe"] > CRITERIA["min_roe"])
    check("ROIC",           m["roic"],         m["roic"] is not None and m["roic"] > CRITERIA["min_roic"])
    check("DebtEquity",     m["debtToEquity"], m["debtToEquity"] is not None and m["debtToEquity"] < CRITERIA["max_debt_to_equity"])
    check("InsiderHolding", m["insider"],      m["insider"] is not None and m["insider"] > CRITERIA["min_insider_holding"])
    check("RevenueGrowth",  m["revGrowth"],    m["revGrowth"] is not None and m["revGrowth"] > CRITERIA["min_revenue_growth"])
    check("EarningsGrowth", m["earnGrowth"],   m["earnGrowth"] is not None and m["earnGrowth"] > CRITERIA["min_earnings_growth"])
    check("OperatingMargin",m["opMargin"],     m["opMargin"] is not None and m["opMargin"] > CRITERIA["min_operating_margin"])
    check("PriceToSales",   m["priceToSales"], m["priceToSales"] is not None and m["priceToSales"] < CRITERIA["max_price_to_sales"])
    check("EV/EBITDA",      m["evEbitda"],     m["evEbitda"] is not None and m["evEbitda"] < CRITERIA["max_ev_ebitda"])

    if CRITERIA["pe_below_sector"]:
        med = sector_pe_median.get(m["sector"])
        check("PE<Sector", m["pe"],
              m["pe"] is not None and med is not None and m["pe"] < med)

    return {
        "green": green, "fails": fails, "skipped": skipped,
        "n_green": len(green),
        "passed_all": len(fails) == 0 and len(skipped) == 0,
    }


def build_sector_pe_medians(rows):
    by_sector = {}
    for m in rows:
        if m["pe"] and m["pe"] > 0:
            by_sector.setdefault(m["sector"], []).append(m["pe"])
    return {s: statistics.median(v) for s, v in by_sector.items() if v}


# ----------------------------------------------------------------------------
# ONE SCAN
# ----------------------------------------------------------------------------
def run_once(tickers, deep=False, top_n=15, email_cfg=None):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== Scan @ {stamp}  |  {len(tickers)} tickers  |  deep={deep} ===")

    rows = []
    for i, t in enumerate(tickers, 1):
        m = fetch(t, deep=deep)
        if m:
            rows.append(m)
        if i % 25 == 0:
            print(f"  ...fetched {i}/{len(tickers)}")
        time.sleep(0.2)  # be polite to the API

    sector_pe_median = build_sector_pe_medians(rows)
    return report(rows, sector_pe_median, top_n=top_n, email_cfg=email_cfg)


def report(rows, sector_pe_median, top_n=15, email_cfg=None):
    """Score every row, print full-passers; if none, print the max-green board."""
    scored = []
    for m in rows:
        sc = evaluate(m, sector_pe_median)
        scored.append({**m, **sc})

    # rank: most green first, then fewest hard fails
    scored.sort(key=lambda r: (r["n_green"], -len(r["fails"])), reverse=True)
    full = [r for r in scored if r["passed_all"]]

    if full:
        print(f"\n--- {len(full)} stock(s) passed ALL criteria ---")
        board, full_pass = full, True
    else:
        best = scored[0]["n_green"] if scored else 0
        print(f"\n--- 0 passed everything. Showing best fits "
              f"(top green count = {best}) ---")
        board, full_pass = scored[:top_n], False

    df = pd.DataFrame([{
        "Ticker": r["ticker"],
        "Green": f"{r['n_green']}/12",
        "Name": r["name"][:24],
        "Sector": r["sector"][:14],
        "MktCap$B": round(r["marketCap"] / 1e9, 1) if r["marketCap"] else None,
        "Missed": ", ".join(r["fails"]) or "-",
        "NoData": ", ".join(r["skipped"]) or "-",
    } for r in board])

    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    fname = f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(fname, index=False)
    print(f"\nSaved -> {fname}")

    if email_cfg:
        send_email(board, full_pass, email_cfg)
    return board


# ----------------------------------------------------------------------------
# EMAIL ALERT
# ----------------------------------------------------------------------------
def send_email(board, full_pass, cfg):
    """Email the leaderboard. cfg keys: host, port, user, password, to.
    Reads from CLI args or env vars (EMAIL_USER, EMAIL_PASSWORD, EMAIL_TO,
    SMTP_HOST, SMTP_PORT)."""
    import smtplib
    import ssl
    from email.mime.text import MIMEText

    if not (cfg["user"] and cfg["password"] and cfg["to"]):
        print("  ! Email not sent: missing user/password/to "
              "(set via --email-* args or EMAIL_* env vars).")
        return

    header = (f"{len(board)} stock(s) passed ALL criteria"
              if full_pass else
              f"No full pass — top {len(board)} by green count")
    lines = [header, ""]
    for r in board:
        miss = ", ".join(r["fails"]) or "none"
        lines.append(f"{r['ticker']:<6} {r['n_green']}/12  "
                     f"{r['name'][:28]:<28}  missed: {miss}")
    body = "\n".join(lines)

    msg = MIMEText(body)
    msg["Subject"] = f"[Scanner] {header} — {datetime.now():%Y-%m-%d %H:%M}"
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
            server.starttls(context=ctx)
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["user"], cfg["to"].split(","), msg.as_string())
        print(f"  ✉  Alert emailed to {cfg['to']}")
    except Exception as e:
        print(f"  ! Email failed: {e}")


# ----------------------------------------------------------------------------
# THE LOOP
# ----------------------------------------------------------------------------
def load_universe(args):
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.file:
        with open(args.file) as f:
            return [ln.strip().upper() for ln in f if ln.strip()]
    if args.universe == "all":
        return get_all_us_tickers()
    return get_sp500_tickers()


def main():
    import os
    p = argparse.ArgumentParser(description="US fundamental multibagger-style scanner")
    p.add_argument("--once", action="store_true", help="run a single scan and exit")
    p.add_argument("--interval", type=int, default=86400,
                   help="seconds between scans when looping (default 86400 = 24h)")
    p.add_argument("--universe", choices=["sp500", "all"], default="sp500",
                   help="'sp500' (~500, fast) or 'all' (~7500 US-listed, slow)")
    p.add_argument("--tickers", type=str, help="comma-separated tickers, e.g. AAPL,MSFT")
    p.add_argument("--file", type=str, help="text file, one ticker per line")
    p.add_argument("--deep", action="store_true",
                   help="compute true 5y ROE / CAGRs from statements (slower)")
    # email alerting
    p.add_argument("--email", action="store_true", help="email the board after each scan")
    p.add_argument("--email-to", type=str, default=os.environ.get("EMAIL_TO"))
    p.add_argument("--email-user", type=str, default=os.environ.get("EMAIL_USER"),
                   help="sender address / SMTP login")
    p.add_argument("--email-password", type=str, default=os.environ.get("EMAIL_PASSWORD"),
                   help="SMTP password or app-password (use an env var, not the CLI)")
    p.add_argument("--smtp-host", type=str,
                   default=os.environ.get("SMTP_HOST", "smtp.gmail.com"))
    p.add_argument("--smtp-port", type=int,
                   default=int(os.environ.get("SMTP_PORT", "587")))
    args = p.parse_args()

    email_cfg = None
    if args.email:
        email_cfg = {
            "host": args.smtp_host, "port": args.smtp_port,
            "user": args.email_user, "password": args.email_password,
            "to": args.email_to,
        }

    tickers = load_universe(args)
    if not tickers:
        print("No tickers to scan."); sys.exit(1)
    if len(tickers) > 1500:
        print(f"  Heads up: scanning {len(tickers)} tickers will take a while "
              f"(yfinance rate limits). Let it run.")

    if args.once:
        run_once(tickers, deep=args.deep, email_cfg=email_cfg)
        return

    print(f"Looping every {args.interval}s. Ctrl+C to stop.")
    while True:
        try:
            run_once(tickers, deep=args.deep, email_cfg=email_cfg)
        except KeyboardInterrupt:
            print("\nStopped."); break
        except Exception as e:
            print(f"  ! Scan error: {e}")
        print(f"\nSleeping {args.interval}s until next scan...")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped."); break


if __name__ == "__main__":
    main()
