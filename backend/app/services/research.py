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
            "reasoning": "Insufficient data to compute moving averages.",
        }

    trend_strength = ((current_price - sma50_val) / sma50_val) * 100.0
    histogram = macd_data["histogram"][-1]

    signal, confidence = "NEUTRAL", 30.0
    reasoning = ""
    if sma20_val > sma50_val and histogram > 0:
        signal, confidence = "BUY", min(90.0, 50.0 + abs(trend_strength) * 2.0)
        reasoning = (
            f"BUY because the 20-day SMA (${sma20_val:.2f}) is above the 50-day SMA (${sma50_val:.2f}), "
            f"confirming an uptrend. The MACD histogram is positive ({histogram:.3f}), indicating "
            f"bullish momentum is accelerating. Price is {abs(trend_strength):.1f}% above the 50-day average."
        )
    elif sma20_val < sma50_val and histogram < 0:
        signal, confidence = "SELL", min(90.0, 50.0 + abs(trend_strength) * 2.0)
        reasoning = (
            f"SELL because the 20-day SMA (${sma20_val:.2f}) has crossed below the 50-day SMA (${sma50_val:.2f}), "
            f"confirming a downtrend. The MACD histogram is negative ({histogram:.3f}), indicating "
            f"bearish momentum is increasing. Price is {abs(trend_strength):.1f}% below the 50-day average."
        )
    else:
        parts = []
        if sma20_val > sma50_val:
            parts.append(f"The 20-day SMA (${sma20_val:.2f}) is above the 50-day SMA (${sma50_val:.2f}), suggesting an uptrend")
        else:
            parts.append(f"The 20-day SMA (${sma20_val:.2f}) is below the 50-day SMA (${sma50_val:.2f}), suggesting a downtrend")
        if histogram > 0:
            parts.append(f"but MACD histogram is positive ({histogram:.3f}), hinting at potential reversal upward")
        else:
            parts.append(f"but MACD histogram is negative ({histogram:.3f}), hinting at potential reversal downward")
        reasoning = "NEUTRAL — mixed signals. " + ", ".join(parts) + ". Wait for confirmation before acting."

    return {
        "name": "Time-Series Trend Following", "icon": "📈",
        "description": "Identifies market trends using moving averages and MACD",
        "signal": signal, "confidence": round(confidence, 1),
        "reasoning": reasoning,
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

    # Build reasoning
    factors = []
    if momentum_factor == 1:
        factors.append(f"30-day return is strong at +{returns_30d*100:.1f}% (bullish momentum)")
    elif momentum_factor == -1:
        factors.append(f"30-day return is weak at {returns_30d*100:.1f}% (bearish momentum)")
    else:
        factors.append(f"30-day return is flat at {returns_30d*100:.1f}% (neutral momentum)")

    if quality_factor == 1:
        factors.append(f"volume ratio ({volume_ratio:.2f}x avg) is healthy, indicating institutional participation")
    else:
        factors.append(f"volume ratio ({volume_ratio:.2f}x avg) is abnormal, suggesting caution")

    if low_vol_factor == 1:
        factors.append(f"daily volatility is low ({vol*100:.2f}%), favorable for stable returns")
    elif low_vol_factor == -1:
        factors.append(f"daily volatility is high ({vol*100:.2f}%), increasing risk")
    else:
        factors.append(f"daily volatility is moderate ({vol*100:.2f}%)")

    reasoning = f"{signal} — composite score is {composite*100:.1f}%. " + ". ".join(f.capitalize() if i == 0 else f for i, f in enumerate(factors)) + "."

    return {
        "name": "Multi-Factor Equity Model", "icon": "⚖️",
        "description": "Combines value, momentum, quality, and low-volatility factors",
        "signal": signal, "confidence": round(confidence, 1),
        "reasoning": reasoning,
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
    reasoning = ""
    if rsi > 70 and rs_score > 10:
        signal, confidence = "SELL", min(80.0, 50.0 + abs(rsi - 70))
        reasoning = (
            f"SELL — RSI is {rsi:.1f} (above 70 = overbought territory), meaning the stock has been "
            f"bought aggressively and is likely overextended. Combined with a relative strength score "
            f"of +{rs_score:.1f}%, the price may be due for a pullback as profit-taking kicks in."
        )
    elif rsi < 30 and rs_score < -10:
        signal, confidence = "BUY", min(80.0, 50.0 + abs(30 - rsi))
        reasoning = (
            f"BUY — RSI is {rsi:.1f} (below 30 = oversold territory), meaning selling pressure has been "
            f"extreme and the stock is likely undervalued short-term. With a relative strength score of "
            f"{rs_score:.1f}%, a bounce-back rally is probable."
        )
    elif rs_score > 5 and rsi > 50:
        signal, confidence = "BUY", 60.0
        reasoning = (
            f"BUY — Moderate bullish momentum. RSI is {rsi:.1f} (above 50 centerline, positive bias) "
            f"and the relative strength score is +{rs_score:.1f}%, showing the stock is outperforming. "
            f"1-week return: {returns_1w*100:+.1f}%, 1-month: {returns_1m*100:+.1f}%."
        )
    elif rs_score < -5 and rsi < 50:
        signal, confidence = "SELL", 60.0
        reasoning = (
            f"SELL — Moderate bearish momentum. RSI is {rsi:.1f} (below 50 centerline, negative bias) "
            f"and the relative strength score is {rs_score:.1f}%, showing the stock is underperforming. "
            f"1-week return: {returns_1w*100:+.1f}%, 1-month: {returns_1m*100:+.1f}%."
        )
    else:
        reasoning = (
            f"NEUTRAL — No clear momentum direction. RSI is {rsi:.1f} (mid-range) and relative strength "
            f"score is {rs_score:+.1f}%. Neither overbought nor oversold. "
            f"1-week: {returns_1w*100:+.1f}%, 1-month: {returns_1m*100:+.1f}%, 3-month: {returns_3m*100:+.1f}%."
        )

    return {
        "name": "Cross-Sectional Momentum", "icon": "🚀",
        "description": "Analyzes relative strength and momentum indicators",
        "signal": signal, "confidence": round(confidence, 1),
        "reasoning": reasoning,
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

    if mean is None or mean == 0:
        return {
            "name": "Statistical Arbitrage", "icon": "🎯",
            "description": "Mean reversion and pairs trading opportunities",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
            "reasoning": "Insufficient data to compute statistical bands.",
        }

    recent = prices[-20:]
    variance = sum((p - mean) ** 2 for p in recent) / 20.0
    std_dev = variance ** 0.5

    if std_dev == 0:
        return {
            "name": "Statistical Arbitrage", "icon": "🎯",
            "description": "Mean reversion and pairs trading opportunities",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
            "reasoning": "Price volatility is zero — no statistical spread to trade.",
        }

    z_score = (current_price - mean) / std_dev
    upper_band = mean + std_dev * 2
    lower_band = mean - std_dev * 2
    dev_pct = ((current_price - mean) / mean) * 100.0

    signal, confidence = "NEUTRAL", abs(z_score) * 25.0
    reasoning = ""
    if z_score < -2:
        signal, confidence = "BUY", min(85.0, 50.0 + abs(z_score) * 15.0)
        reasoning = (
            f"BUY — Price (${current_price:.2f}) is {abs(dev_pct):.1f}% below the 20-day mean (${mean:.2f}) "
            f"with a z-score of {z_score:.2f}. This is below the lower Bollinger band (${lower_band:.2f}), "
            f"an extreme deviation. Historically, prices tend to revert to the mean from this level."
        )
    elif z_score > 2:
        signal, confidence = "SELL", min(85.0, 50.0 + abs(z_score) * 15.0)
        reasoning = (
            f"SELL — Price (${current_price:.2f}) is {dev_pct:.1f}% above the 20-day mean (${mean:.2f}) "
            f"with a z-score of +{z_score:.2f}. This is above the upper Bollinger band (${upper_band:.2f}), "
            f"an extreme deviation. Mean reversion suggests the price is likely to pull back."
        )
    elif z_score < -1:
        signal, confidence = "BUY", 40.0
        reasoning = (
            f"BUY (weak) — Price is moderately below the mean. Z-score of {z_score:.2f} suggests the stock "
            f"is trading ${abs(current_price - mean):.2f} below its 20-day average of ${mean:.2f}. "
            f"Partial mean reversion opportunity, but not at extreme levels yet."
        )
    elif z_score > 1:
        signal, confidence = "SELL", 40.0
        reasoning = (
            f"SELL (weak) — Price is moderately above the mean. Z-score of +{z_score:.2f} suggests the stock "
            f"is trading ${current_price - mean:.2f} above its 20-day average of ${mean:.2f}. "
            f"Partial overextension, but not at extreme levels yet."
        )
    else:
        reasoning = (
            f"NEUTRAL — Price (${current_price:.2f}) is close to the 20-day mean (${mean:.2f}) with a "
            f"z-score of {z_score:+.2f}. No significant deviation detected. The stock is trading within "
            f"normal statistical bounds (${lower_band:.2f} – ${upper_band:.2f})."
        )

    return {
        "name": "Statistical Arbitrage", "icon": "🎯",
        "description": "Mean reversion and pairs trading opportunities",
        "signal": signal, "confidence": round(confidence, 1),
        "reasoning": reasoning,
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

    # Feature scores
    f_mom = momentum * 0.3
    f_rsi = (1 if rsi < 30 else (-1 if rsi > 70 else 0)) * 0.25
    f_macd = (1 if macd_data["histogram"][-1] > 0 else -1) * 0.2
    f_vol = (1 if vol < 0.02 else (-1 if vol > 0.04 else 0)) * 0.15
    f_volano = (1 if volume_anomaly else 0) * 0.1

    ml_score = f_mom + f_rsi + f_macd + f_vol + f_volano

    signal = "BUY" if ml_score > 0.3 else ("SELL" if ml_score < -0.3 else "NEUTRAL")
    confidence = min(75.0, abs(ml_score) * 100.0)

    # Build feature-by-feature reasoning
    feature_notes = []
    if momentum > 0:
        feature_notes.append(f"5-day momentum is positive ({momentum*100:+.2f}%) → bullish")
    else:
        feature_notes.append(f"5-day momentum is negative ({momentum*100:+.2f}%) → bearish")

    if rsi < 30:
        feature_notes.append(f"RSI ({rsi:.1f}) signals oversold → bullish")
    elif rsi > 70:
        feature_notes.append(f"RSI ({rsi:.1f}) signals overbought → bearish")
    else:
        feature_notes.append(f"RSI ({rsi:.1f}) is neutral")

    feature_notes.append(f"MACD histogram is {'positive → bullish' if macd_data['histogram'][-1] > 0 else 'negative → bearish'}")

    if vol < 0.02:
        feature_notes.append(f"low volatility ({vol*100:.2f}%) → favorable")
    elif vol > 0.04:
        feature_notes.append(f"high volatility ({vol*100:.2f}%) → risky")
    else:
        feature_notes.append(f"moderate volatility ({vol*100:.2f}%) → neutral")

    if volume_anomaly:
        feature_notes.append("unusual volume spike detected → heightened activity")

    reasoning = (
        f"{signal} — The ML model combines 5 weighted features into a composite score of {ml_score*100:+.1f}%. "
        f"Feature breakdown: {'; '.join(feature_notes)}."
    )

    return {
        "name": "Machine Learning Alpha", "icon": "🤖",
        "description": "AI-driven signal generation using multiple features",
        "signal": signal, "confidence": round(confidence, 1),
        "reasoning": reasoning,
        "metrics": {
            "ML Score": f"{ml_score * 100:.1f}%",
            "Feature Count": "5 active",
            "Momentum": f"{momentum * 100:.2f}%",
            "Volume Anomaly": "Detected" if volume_anomaly else "Normal",
        },
    }


# ── New Strategies ────────────────────────────────────────────────────────


def strategy_bollinger_squeeze(prices: list[float]) -> dict[str, Any]:
    """Detects Bollinger Band squeeze (low volatility) and breakout direction."""
    bb = compute_bollinger(prices, 20, 2.0)
    current_price = prices[-1]

    upper = bb["upper"][-1]
    lower = bb["lower"][-1]
    middle = bb["middle"][-1]

    if upper is None or lower is None or middle is None or middle == 0:
        return {
            "name": "Bollinger Band Squeeze", "icon": "🔄",
            "description": "Detects volatility contraction and imminent breakout direction",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
            "reasoning": "Insufficient data for Bollinger Band calculation.",
        }

    bandwidth = ((upper - lower) / middle) * 100.0

    # Check recent bandwidth trend (narrowing = squeeze)
    recent_bw = []
    for i in range(-10, 0):
        u, l, m = bb["upper"][i], bb["lower"][i], bb["middle"][i]
        if u is not None and l is not None and m is not None and m != 0:
            recent_bw.append(((u - l) / m) * 100.0)

    avg_bw = sum(recent_bw) / len(recent_bw) if recent_bw else bandwidth
    is_squeeze = bandwidth < avg_bw * 0.85  # Band narrowing >15% from recent avg

    pct_b = (current_price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5

    signal, confidence = "NEUTRAL", 40.0
    reasoning = ""

    if is_squeeze and pct_b > 0.8:
        signal, confidence = "BUY", min(80.0, 55.0 + (pct_b - 0.8) * 100)
        reasoning = (
            f"BUY — Bollinger Bands are squeezing (bandwidth {bandwidth:.2f}% vs avg {avg_bw:.2f}%), "
            f"indicating a period of low volatility. Price (${current_price:.2f}) is near the upper band "
            f"(%B = {pct_b:.2f}), suggesting the breakout will likely be upward. Volatility expansions "
            f"after squeezes tend to produce strong directional moves."
        )
    elif is_squeeze and pct_b < 0.2:
        signal, confidence = "SELL", min(80.0, 55.0 + (0.2 - pct_b) * 100)
        reasoning = (
            f"SELL — Bollinger Bands are squeezing (bandwidth {bandwidth:.2f}% vs avg {avg_bw:.2f}%), "
            f"indicating a period of low volatility. Price (${current_price:.2f}) is near the lower band "
            f"(%B = {pct_b:.2f}), suggesting the breakout will likely be downward."
        )
    elif pct_b > 1.0:
        signal, confidence = "SELL", 55.0
        reasoning = (
            f"SELL — Price (${current_price:.2f}) has broken above the upper Bollinger Band (${upper:.2f}). "
            f"%B is {pct_b:.2f} (>1.0 = outside upper band). This overextension often precedes a reversion "
            f"back inside the bands."
        )
    elif pct_b < 0.0:
        signal, confidence = "BUY", 55.0
        reasoning = (
            f"BUY — Price (${current_price:.2f}) has broken below the lower Bollinger Band (${lower:.2f}). "
            f"%B is {pct_b:.2f} (<0 = outside lower band). This extreme selling often precedes a bounce "
            f"back inside the bands."
        )
    else:
        squeeze_note = f"Bands are {'squeezing — watch for imminent breakout' if is_squeeze else 'normal width'}."
        reasoning = (
            f"NEUTRAL — Price is within the Bollinger Bands. %B is {pct_b:.2f} "
            f"(0 = lower band, 1 = upper band). Bandwidth is {bandwidth:.2f}%. {squeeze_note}"
        )

    return {
        "name": "Bollinger Band Squeeze", "icon": "🔄",
        "description": "Detects volatility contraction and imminent breakout direction",
        "signal": signal, "confidence": round(confidence, 1),
        "reasoning": reasoning,
        "metrics": {
            "Bandwidth": f"{bandwidth:.2f}%",
            "%B": f"{pct_b:.2f}",
            "Upper Band": f"${upper:.2f}",
            "Lower Band": f"${lower:.2f}",
        },
    }


def _compute_stochastic(prices_high: list[float], prices_low: list[float], prices_close: list[float], k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
    """Compute Stochastic Oscillator %K and %D."""
    if len(prices_close) < k_period:
        return 50.0, 50.0

    k_values = []
    for i in range(k_period - 1, len(prices_close)):
        h = max(prices_high[i - k_period + 1 : i + 1])
        l = min(prices_low[i - k_period + 1 : i + 1])
        if h == l:
            k_values.append(50.0)
        else:
            k_values.append(((prices_close[i] - l) / (h - l)) * 100.0)

    pct_k = k_values[-1] if k_values else 50.0
    pct_d = sum(k_values[-d_period:]) / min(d_period, len(k_values)) if k_values else 50.0
    return pct_k, pct_d


def strategy_stochastic(high: list[float], low: list[float], close: list[float]) -> dict[str, Any]:
    """Stochastic Oscillator strategy — overbought/oversold with %K/%D crossovers."""
    pct_k, pct_d = _compute_stochastic(high, low, close)

    signal, confidence = "NEUTRAL", 40.0
    reasoning = ""

    if pct_k < 20 and pct_d < 20:
        signal, confidence = "BUY", min(80.0, 55.0 + (20 - pct_k))
        if pct_k > pct_d:
            reasoning = (
                f"BUY — Stochastic %K ({pct_k:.1f}) and %D ({pct_d:.1f}) are both below 20 (oversold zone). "
                f"%K has crossed above %D, which is a classic bullish crossover signal. This suggests selling "
                f"pressure is exhausted and a reversal upward is likely."
            )
        else:
            reasoning = (
                f"BUY — Both %K ({pct_k:.1f}) and %D ({pct_d:.1f}) are in extreme oversold territory (<20). "
                f"While %K hasn't crossed above %D yet, the extreme reading suggests the stock is due for a "
                f"relief bounce. Watch for %K to cross above %D for confirmation."
            )
    elif pct_k > 80 and pct_d > 80:
        signal, confidence = "SELL", min(80.0, 55.0 + (pct_k - 80))
        if pct_k < pct_d:
            reasoning = (
                f"SELL — Stochastic %K ({pct_k:.1f}) and %D ({pct_d:.1f}) are both above 80 (overbought zone). "
                f"%K has crossed below %D, which is a classic bearish crossover signal. This suggests buying "
                f"pressure is waning and a pullback is likely."
            )
        else:
            reasoning = (
                f"SELL — Both %K ({pct_k:.1f}) and %D ({pct_d:.1f}) are in extreme overbought territory (>80). "
                f"The stock has rallied significantly. Watch for %K to cross below %D for a confirmed sell signal."
            )
    else:
        zone = "upper" if pct_k > 50 else "lower"
        cross = "above" if pct_k > pct_d else "below"
        reasoning = (
            f"NEUTRAL — %K is {pct_k:.1f} and %D is {pct_d:.1f}, both in the {zone} half of the range. "
            f"%K is {cross} %D. Neither overbought (>80) nor oversold (<20). No actionable signal at this time."
        )

    return {
        "name": "Stochastic Oscillator", "icon": "📉",
        "description": "Identifies overbought/oversold conditions using %K and %D crossovers",
        "signal": signal, "confidence": round(confidence, 1),
        "reasoning": reasoning,
        "metrics": {
            "%K (14)": f"{pct_k:.1f}",
            "%D (3)": f"{pct_d:.1f}",
            "Zone": "Overbought" if pct_k > 80 else ("Oversold" if pct_k < 20 else "Neutral"),
            "Crossover": "Bullish" if pct_k > pct_d else "Bearish",
        },
    }


def _compute_adx(high: list[float], low: list[float], close: list[float], period: int = 14) -> tuple[float, float, float]:
    """Compute ADX, +DI, -DI."""
    if len(close) < period + 1:
        return 25.0, 25.0, 25.0

    plus_dm_list, minus_dm_list, tr_list = [], [], []

    for i in range(1, len(close)):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]

        plus_dm = h_diff if (h_diff > l_diff and h_diff > 0) else 0
        minus_dm = l_diff if (l_diff > h_diff and l_diff > 0) else 0

        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
        tr_list.append(tr)

    # Smoothed averages (Wilder's smoothing)
    atr = sum(tr_list[:period]) / period
    plus_di_smooth = sum(plus_dm_list[:period]) / period
    minus_di_smooth = sum(minus_dm_list[:period]) / period

    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_di_smooth = (plus_di_smooth * (period - 1) + plus_dm_list[i]) / period
        minus_di_smooth = (minus_di_smooth * (period - 1) + minus_dm_list[i]) / period

    plus_di = (plus_di_smooth / atr * 100) if atr != 0 else 0
    minus_di = (minus_di_smooth / atr * 100) if atr != 0 else 0

    dx_sum = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) != 0 else 0
    adx = dx_sum  # Simplified — in production you'd smooth DX over another period

    return adx, plus_di, minus_di


def strategy_adx(high: list[float], low: list[float], close: list[float]) -> dict[str, Any]:
    """ADX Trend Strength — measures how strong the current trend is."""
    adx, plus_di, minus_di = _compute_adx(high, low, close)

    signal, confidence = "NEUTRAL", 40.0
    reasoning = ""

    if adx > 25 and plus_di > minus_di:
        signal = "BUY"
        confidence = min(85.0, 45.0 + adx)
        reasoning = (
            f"BUY — ADX is {adx:.1f} (above 25 = trending market), confirming a strong trend is in place. "
            f"+DI ({plus_di:.1f}) is above -DI ({minus_di:.1f}), meaning the trend direction is bullish. "
            f"The combination of strong trend + bullish direction supports buying."
        )
    elif adx > 25 and minus_di > plus_di:
        signal = "SELL"
        confidence = min(85.0, 45.0 + adx)
        reasoning = (
            f"SELL — ADX is {adx:.1f} (above 25 = trending market), confirming a strong trend is in place. "
            f"-DI ({minus_di:.1f}) is above +DI ({plus_di:.1f}), meaning the trend direction is bearish. "
            f"The combination of strong trend + bearish direction supports selling."
        )
    elif adx < 20:
        reasoning = (
            f"NEUTRAL — ADX is {adx:.1f} (below 20 = weak/no trend). The market is ranging, not trending. "
            f"+DI is {plus_di:.1f}, -DI is {minus_di:.1f}. Trend-following strategies are ineffective in "
            f"low-ADX environments. Consider range-bound strategies instead."
        )
    else:
        direction = "bullish" if plus_di > minus_di else "bearish"
        reasoning = (
            f"NEUTRAL — ADX is {adx:.1f} (borderline trending). Direction is mildly {direction} "
            f"(+DI: {plus_di:.1f}, -DI: {minus_di:.1f}). The trend is not strong enough for high-confidence "
            f"entry. Wait for ADX to rise above 25 for confirmation."
        )

    return {
        "name": "ADX Trend Strength", "icon": "💪",
        "description": "Measures trend strength and direction using Average Directional Index",
        "signal": signal, "confidence": round(confidence, 1),
        "reasoning": reasoning,
        "metrics": {
            "ADX": f"{adx:.1f}",
            "+DI": f"{plus_di:.1f}",
            "-DI": f"{minus_di:.1f}",
            "Trend": "Strong" if adx > 25 else ("Weak" if adx < 20 else "Moderate"),
        },
    }


def strategy_obv(close: list[float], volumes: list[float]) -> dict[str, Any]:
    """On-Balance Volume — confirms price trends via volume flow."""
    if len(close) < 21 or len(volumes) < 21:
        return {
            "name": "OBV Volume Trend", "icon": "📊",
            "description": "Confirms price trends by analyzing cumulative volume flow",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
            "reasoning": "Insufficient data for OBV calculation.",
        }

    # Calculate OBV
    obv = [0.0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])

    # OBV trend (20-day SMA of OBV)
    obv_sma = compute_sma(obv, 20)
    current_obv = obv[-1]
    obv_sma_val = obv_sma[-1]

    # Price trend
    price_sma20 = compute_sma(close, 20)
    price_above_sma = close[-1] > (price_sma20[-1] or close[-1])

    # OBV direction over last 10 days
    obv_10d_change = obv[-1] - obv[-10] if len(obv) >= 10 else 0
    obv_rising = obv_10d_change > 0

    signal, confidence = "NEUTRAL", 40.0
    reasoning = ""

    if price_above_sma and obv_rising and obv_sma_val is not None and current_obv > obv_sma_val:
        signal, confidence = "BUY", 65.0
        reasoning = (
            f"BUY — Price is above its 20-day SMA and OBV is rising (10-day OBV change: "
            f"{obv_10d_change/1e6:+.1f}M). OBV is above its own 20-day average, confirming that volume "
            f"is supporting the uptrend. When smart money accumulates shares, OBV rises ahead of price."
        )
    elif not price_above_sma and not obv_rising and obv_sma_val is not None and current_obv < obv_sma_val:
        signal, confidence = "SELL", 65.0
        reasoning = (
            f"SELL — Price is below its 20-day SMA and OBV is falling (10-day OBV change: "
            f"{obv_10d_change/1e6:+.1f}M). OBV is below its own 20-day average, confirming that volume "
            f"is supporting the downtrend. Distribution (selling) is occurring."
        )
    elif price_above_sma and not obv_rising:
        reasoning = (
            f"NEUTRAL (bearish divergence) — Price is above its 20-day SMA but OBV is declining "
            f"(10-day OBV change: {obv_10d_change/1e6:+.1f}M). This divergence warns that the uptrend "
            f"may be losing steam — price is rising but volume isn't confirming."
        )
    elif not price_above_sma and obv_rising:
        reasoning = (
            f"NEUTRAL (bullish divergence) — Price is below its 20-day SMA but OBV is rising "
            f"(10-day OBV change: {obv_10d_change/1e6:+.1f}M). This divergence suggests accumulation — "
            f"smart money may be buying despite the price decline, hinting at a potential reversal."
        )
    else:
        reasoning = (
            f"NEUTRAL — OBV and price trend are inconclusive. 10-day OBV change: "
            f"{obv_10d_change/1e6:+.1f}M. No clear volume-price confirmation in either direction."
        )

    return {
        "name": "OBV Volume Trend", "icon": "📊",
        "description": "Confirms price trends by analyzing cumulative volume flow",
        "signal": signal, "confidence": round(confidence, 1),
        "reasoning": reasoning,
        "metrics": {
            "OBV Trend": "Rising" if obv_rising else "Falling",
            "10d OBV Chg": f"{obv_10d_change/1e6:+.1f}M",
            "Price vs SMA": "Above" if price_above_sma else "Below",
            "Confirmation": "Yes" if (price_above_sma == obv_rising) else "Divergence",
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
        strategy_bollinger_squeeze(close),
        strategy_stochastic(high, low, close),
        strategy_adx(high, low, close),
        strategy_obv(close, volume),
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
