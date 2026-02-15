from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd


# ── Data Fetching ──────────────────────────────────────────────────────────


def fetch_single_ticker_ohlcv(yahoo_ticker: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily OHLCV data for a single ticker via yfinance."""
    import yfinance as yf

    end_date = date.today() + timedelta(days=1)
    # Extra 250 calendar days so 200-DMA has valid values in the display range
    start_date = end_date - timedelta(days=days + 250)

    try:
        df = yf.download(
            tickers=yahoo_ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {yahoo_ticker}: {e}")

    if df is None or df.empty:
        raise ValueError(f"No data available for {yahoo_ticker}")

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing {col} column for {yahoo_ticker}")

    return df[required].dropna(subset=["Close"]).sort_index()


def fetch_ticker_info(yahoo_ticker: str) -> dict[str, Any]:
    """Fetch fundamental data (P/E, market cap, beta) from yfinance."""
    import yfinance as yf

    try:
        t = yf.Ticker(yahoo_ticker)
        info = t.info or {}
        return {
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "marketCap": info.get("marketCap"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "dividendYield": info.get("dividendYield"),
        }
    except Exception:
        return {}


# ── Utility ────────────────────────────────────────────────────────────────


def _clean(v: Any) -> Any:
    """Convert NaN/Inf to None, round floats."""
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, float):
        return round(v, 4)
    return v


def _clean_list(lst: list) -> list:
    return [_clean(v) for v in lst]


# ── Technical Indicators ──────────────────────────────────────────────────


def compute_sma(prices: list[float], period: int) -> list[Optional[float]]:
    result: list[Optional[float]] = []
    for i in range(len(prices)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(prices[i - period + 1 : i + 1]) / period)
    return result


def compute_ema(prices: list[float], period: int) -> list[float]:
    if not prices:
        return []
    k = 2.0 / (period + 1)
    result = [prices[0]]
    for i in range(1, len(prices)):
        result.append(prices[i] * k + result[-1] * (1 - k))
    return result


def compute_rsi(prices: list[float], period: int = 14) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(prices)
    if len(prices) < period + 1:
        return result

    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(c, 0) for c in changes]
    losses = [max(-c, 0) for c in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def compute_macd(prices: list[float]) -> dict[str, list[float]]:
    ema12 = compute_ema(prices, 12)
    ema26 = compute_ema(prices, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(prices))]
    signal_line = compute_ema(macd_line, 9)
    histogram = [macd_line[i] - signal_line[i] for i in range(len(prices))]
    return {
        "macdLine": macd_line,
        "signalLine": signal_line,
        "histogram": histogram,
    }


def compute_bollinger(
    prices: list[float], period: int = 20, num_std: float = 2.0
) -> dict[str, list[Optional[float]]]:
    upper: list[Optional[float]] = [None] * len(prices)
    middle: list[Optional[float]] = [None] * len(prices)
    lower: list[Optional[float]] = [None] * len(prices)

    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1 : i + 1]
        mean = sum(window) / period
        std = (sum((p - mean) ** 2 for p in window) / period) ** 0.5
        middle[i] = mean
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std

    return {"upper": upper, "middle": middle, "lower": lower}


def compute_fibonacci(prices: list[float]) -> dict[str, float]:
    high = max(prices)
    low = min(prices)
    diff = high - low
    return {
        "high": round(high, 2),
        "low": round(low, 2),
        "level_0": round(high, 2),
        "level_236": round(high - diff * 0.236, 2),
        "level_382": round(high - diff * 0.382, 2),
        "level_500": round(high - diff * 0.500, 2),
        "level_618": round(high - diff * 0.618, 2),
        "level_786": round(high - diff * 0.786, 2),
        "level_1000": round(low, 2),
    }


# ── Quantitative Strategies ───────────────────────────────────────────────


def _volatility(prices: list[float], period: int = 20) -> float:
    if len(prices) < period + 1:
        return 0.02
    recent = prices[-period:]
    returns = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent)) if recent[i - 1] != 0]
    if not returns:
        return 0.02
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return variance ** 0.5


def strategy_trend_following(prices: list[float], volumes: list[float]) -> dict[str, Any]:
    sma20 = compute_sma(prices, 20)
    sma50 = compute_sma(prices, 50)
    macd_data = compute_macd(prices)

    current_price = prices[-1]
    sma50_val = sma50[-1]
    sma20_val = sma20[-1]

    if sma50_val is None or sma20_val is None:
        return {
            "name": "Time-Series Trend Following", "icon": "📈",
            "description": "Identifies market trends using moving averages and MACD",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
        }

    trend_strength = ((current_price - sma50_val) / sma50_val) * 100.0
    histogram = macd_data["histogram"][-1]

    signal, confidence = "NEUTRAL", 30.0
    if sma20_val > sma50_val and histogram > 0:
        signal, confidence = "BUY", min(90.0, 50.0 + abs(trend_strength) * 2.0)
    elif sma20_val < sma50_val and histogram < 0:
        signal, confidence = "SELL", min(90.0, 50.0 + abs(trend_strength) * 2.0)

    return {
        "name": "Time-Series Trend Following", "icon": "📈",
        "description": "Identifies market trends using moving averages and MACD",
        "signal": signal, "confidence": round(confidence, 1),
        "metrics": {
            "SMA 20": f"${sma20_val:.2f}", "SMA 50": f"${sma50_val:.2f}",
            "MACD": f"{macd_data['macdLine'][-1]:.3f}",
            "Trend Strength": f"{trend_strength:.2f}%",
        },
    }


def strategy_multi_factor(prices: list[float], volumes: list[float]) -> dict[str, Any]:
    current_price = prices[-1]
    returns_30d = (current_price - prices[-30]) / prices[-30] if len(prices) >= 30 else 0.0
    momentum_factor = 1 if returns_30d > 0.05 else (-1 if returns_30d < -0.05 else 0)

    avg_volume = sum(volumes) / len(volumes) if volumes else 1.0
    current_vol = volumes[-1] if volumes else 1.0
    volume_ratio = current_vol / avg_volume if avg_volume else 1.0
    quality_factor = 1 if 0.8 < volume_ratio < 1.5 else -1

    vol = _volatility(prices)
    low_vol_factor = 1 if vol < 0.02 else (-1 if vol > 0.04 else 0)

    composite = (momentum_factor * 2 + quality_factor + low_vol_factor) / 4.0

    signal = "BUY" if composite > 0.3 else ("SELL" if composite < -0.3 else "NEUTRAL")
    confidence = min(85.0, abs(composite) * 100.0)

    return {
        "name": "Multi-Factor Equity Model", "icon": "⚖️",
        "description": "Combines value, momentum, quality, and low-volatility factors",
        "signal": signal, "confidence": round(confidence, 1),
        "metrics": {
            "Momentum (30d)": f"{returns_30d * 100:.2f}%",
            "Quality Score": f"{quality_factor * 100}%",
            "Volatility": f"{vol * 100:.2f}%",
            "Composite Score": f"{composite * 100:.1f}%",
        },
    }


def strategy_momentum(prices: list[float]) -> dict[str, Any]:
    current_price = prices[-1]
    rsi_series = compute_rsi(prices)
    rsi = rsi_series[-1] if rsi_series[-1] is not None else 50.0

    returns_1w = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0.0
    returns_1m = (current_price - prices[-21]) / prices[-21] if len(prices) >= 21 else 0.0
    returns_3m = (current_price - prices[-63]) / prices[-63] if len(prices) >= 63 else 0.0

    rs_score = (returns_1w * 0.5 + returns_1m * 0.3 + returns_3m * 0.2) * 100.0

    signal, confidence = "NEUTRAL", 50.0
    if rsi > 70 and rs_score > 10:
        signal, confidence = "SELL", min(80.0, 50.0 + abs(rsi - 70))
    elif rsi < 30 and rs_score < -10:
        signal, confidence = "BUY", min(80.0, 50.0 + abs(30 - rsi))
    elif rs_score > 5 and rsi > 50:
        signal, confidence = "BUY", 60.0
    elif rs_score < -5 and rsi < 50:
        signal, confidence = "SELL", 60.0

    return {
        "name": "Cross-Sectional Momentum", "icon": "🚀",
        "description": "Analyzes relative strength and momentum indicators",
        "signal": signal, "confidence": round(confidence, 1),
        "metrics": {
            "RSI": f"{rsi:.2f}", "RS Score": f"{rs_score:.2f}%",
            "1W Return": f"{returns_1w * 100:.2f}%",
            "1M Return": f"{returns_1m * 100:.2f}%",
        },
    }


def strategy_stat_arb(prices: list[float]) -> dict[str, Any]:
    sma20 = compute_sma(prices, 20)
    mean = sma20[-1]
    current_price = prices[-1]

    if mean is None:
        return {
            "name": "Statistical Arbitrage", "icon": "🎯",
            "description": "Mean reversion and pairs trading opportunities",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
        }

    recent = prices[-20:]
    variance = sum((p - mean) ** 2 for p in recent) / 20.0
    std_dev = variance ** 0.5

    if std_dev == 0:
        return {
            "name": "Statistical Arbitrage", "icon": "🎯",
            "description": "Mean reversion and pairs trading opportunities",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
        }

    z_score = (current_price - mean) / std_dev
    upper_band = mean + std_dev * 2
    lower_band = mean - std_dev * 2

    signal, confidence = "NEUTRAL", abs(z_score) * 25.0
    if z_score < -2:
        signal, confidence = "BUY", min(85.0, 50.0 + abs(z_score) * 15.0)
    elif z_score > 2:
        signal, confidence = "SELL", min(85.0, 50.0 + abs(z_score) * 15.0)
    elif z_score < -1:
        signal, confidence = "BUY", 40.0
    elif z_score > 1:
        signal, confidence = "SELL", 40.0

    return {
        "name": "Statistical Arbitrage", "icon": "🎯",
        "description": "Mean reversion and pairs trading opportunities",
        "signal": signal, "confidence": round(confidence, 1),
        "metrics": {
            "Z-Score": f"{z_score:.2f}",
            "Upper Band": f"${upper_band:.2f}",
            "Lower Band": f"${lower_band:.2f}",
            "Mean Price": f"${mean:.2f}",
        },
    }


def strategy_ml_alpha(prices: list[float], volumes: list[float]) -> dict[str, Any]:
    rsi_series = compute_rsi(prices)
    rsi = rsi_series[-1] if rsi_series[-1] is not None else 50.0
    macd_data = compute_macd(prices)
    vol = _volatility(prices)

    avg_vol = sum(volumes[-20:-1]) / 19.0 if len(volumes) >= 20 else (sum(volumes) / len(volumes) if volumes else 1.0)
    current_vol = volumes[-1] if volumes else 0.0
    volume_anomaly = current_vol > avg_vol * 1.5

    momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0.0

    ml_score = (
        momentum * 0.3
        + (1 if rsi < 30 else (-1 if rsi > 70 else 0)) * 0.25
        + (1 if macd_data["histogram"][-1] > 0 else -1) * 0.2
        + (1 if vol < 0.02 else (-1 if vol > 0.04 else 0)) * 0.15
        + (1 if volume_anomaly else 0) * 0.1
    )

    signal = "BUY" if ml_score > 0.3 else ("SELL" if ml_score < -0.3 else "NEUTRAL")
    confidence = min(75.0, abs(ml_score) * 100.0)

    return {
        "name": "Machine Learning Alpha", "icon": "🤖",
        "description": "AI-driven signal generation using multiple features",
        "signal": signal, "confidence": round(confidence, 1),
        "metrics": {
            "ML Score": f"{ml_score * 100:.1f}%",
            "Feature Count": "5 active",
            "Momentum": f"{momentum * 100:.2f}%",
            "Volume Anomaly": "Detected" if volume_anomaly else "Normal",
        },
    }


# ── Main Compute ──────────────────────────────────────────────────────────


def compute_research(
    yahoo_ticker: str,
    company_name: str,
    sector: str,
    days: int = 365,
) -> dict[str, Any]:
    """
    Fetch OHLCV data and compute comprehensive technical analysis
    for a single ticker.
    """
    df = fetch_single_ticker_ohlcv(yahoo_ticker, days)

    close = df["Close"].tolist()
    open_ = df["Open"].tolist()
    high = df["High"].tolist()
    low = df["Low"].tolist()
    volume = df["Volume"].tolist()
    all_dates = [d.strftime("%Y-%m-%d") for d in df.index]

    # ── Compute indicators on full dataset ────────────────────────────────
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)
    rsi_series = compute_rsi(close)
    macd_data = compute_macd(close)
    bollinger = compute_bollinger(close)

    # ── Trim to display range ─────────────────────────────────────────────
    display_start = date.today() - timedelta(days=days)
    display_idx = 0
    for i, d in enumerate(all_dates):
        if d >= display_start.isoformat():
            display_idx = i
            break

    dates = all_dates[display_idx:]
    d_close = close[display_idx:]
    d_open = open_[display_idx:]
    d_high = high[display_idx:]
    d_low = low[display_idx:]
    d_volume = volume[display_idx:]
    d_sma50 = sma50[display_idx:]
    d_sma200 = sma200[display_idx:]
    d_rsi = rsi_series[display_idx:]
    d_macd_line = macd_data["macdLine"][display_idx:]
    d_signal_line = macd_data["signalLine"][display_idx:]
    d_histogram = macd_data["histogram"][display_idx:]
    d_bb_upper = bollinger["upper"][display_idx:]
    d_bb_middle = bollinger["middle"][display_idx:]
    d_bb_lower = bollinger["lower"][display_idx:]

    # ── Fibonacci on display range ────────────────────────────────────────
    fibonacci = compute_fibonacci(d_close)

    # ── Current values ────────────────────────────────────────────────────
    current_price = close[-1]
    prev_close = close[-2] if len(close) >= 2 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100.0 if prev_close != 0 else 0.0

    avg_volume = sum(d_volume[-20:]) / min(20, len(d_volume)) if d_volume else 0
    current_volume = d_volume[-1] if d_volume else 0

    # Latest RSI value
    latest_rsi = None
    for v in reversed(rsi_series):
        if v is not None:
            latest_rsi = round(v, 2)
            break

    # ── Crossover status ──────────────────────────────────────────────────
    dma50_val = sma50[-1]
    dma200_val = sma200[-1]
    crossover_signal = "none"
    gap_pct: Optional[float] = None

    if dma50_val is not None and dma200_val is not None and dma200_val != 0:
        gap_pct = ((dma50_val - dma200_val) / dma200_val) * 100.0
        if abs(gap_pct) <= 2.0:
            crossover_signal = "near_golden_cross" if dma50_val < dma200_val else "near_death_cross"
        elif dma50_val > dma200_val:
            crossover_signal = "golden_cross"
        else:
            crossover_signal = "death_cross"

    # ── Strategies ────────────────────────────────────────────────────────
    strategies = [
        strategy_trend_following(close, volume),
        strategy_multi_factor(close, volume),
        strategy_momentum(close),
        strategy_stat_arb(close),
        strategy_ml_alpha(close, volume),
    ]

    # ── Fundamental data (best-effort) ────────────────────────────────────
    fundamentals = fetch_ticker_info(yahoo_ticker)

    # ── Build response ────────────────────────────────────────────────────
    return {
        "ticker": yahoo_ticker.replace("-", "."),
        "companyName": company_name,
        "sector": sector,
        "currentPrice": round(current_price, 2),
        "previousClose": round(prev_close, 2),
        "change": round(change, 2),
        "changePct": round(change_pct, 2),
        "volume": int(current_volume),
        "avgVolume": int(avg_volume),
        "latestRSI": latest_rsi,
        "fundamentals": fundamentals,
        "ohlcv": {
            "dates": dates,
            "open": _clean_list(d_open),
            "high": _clean_list(d_high),
            "low": _clean_list(d_low),
            "close": _clean_list(d_close),
            "volume": [int(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0 for v in d_volume],
        },
        "indicators": {
            "sma50": _clean_list(d_sma50),
            "sma200": _clean_list(d_sma200),
            "rsi": _clean_list(d_rsi),
            "macd": {
                "macdLine": _clean_list(d_macd_line),
                "signalLine": _clean_list(d_signal_line),
                "histogram": _clean_list(d_histogram),
            },
            "bollinger": {
                "upper": _clean_list(d_bb_upper),
                "middle": _clean_list(d_bb_middle),
                "lower": _clean_list(d_bb_lower),
            },
        },
        "fibonacci": fibonacci,
        "crossover": {
            "dma50": round(dma50_val, 2) if dma50_val is not None else None,
            "dma200": round(dma200_val, 2) if dma200_val is not None else None,
            "gapPct": round(gap_pct, 2) if gap_pct is not None else None,
            "signal": crossover_signal,
        },
        "strategies": strategies,
    }
