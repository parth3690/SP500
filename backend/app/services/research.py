from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd


# ── Data Fetching ──────────────────────────────────────────────────────────


def fetch_single_ticker_ohlcv(
    yahoo_ticker: str,
    display_start: Optional[date] = None,
    display_end: Optional[date] = None,
    days: int = 365,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for a single ticker via yfinance."""
    import yfinance as yf

    if display_end is None:
        display_end = date.today()
    if display_start is None:
        display_start = display_end - timedelta(days=days)

    fetch_end = display_end + timedelta(days=1)
    fetch_start = display_start - timedelta(days=250)

    try:
        df = yf.download(
            tickers=yahoo_ticker,
            start=fetch_start.isoformat(),
            end=fetch_end.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {yahoo_ticker}: {e}")

    if df is None or df.empty:
        raise ValueError(f"No data available for {yahoo_ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0).tolist()
        if "Close" in level0:
            df.columns = level0
        else:
            df.columns = df.columns.get_level_values(1).tolist()

    df = df.loc[:, ~df.columns.duplicated()]

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing {col} column for {yahoo_ticker}")

    result = df[required].dropna(subset=["Close"]).sort_index()

    for col in required:
        if isinstance(result[col], pd.DataFrame):
            result[col] = result[col].iloc[:, 0]

    return result


def fetch_ticker_info(yahoo_ticker: str) -> dict[str, Any]:
    """Fetch fundamental data from yfinance."""
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


def _clean_series(s: pd.Series) -> list:
    """Convert a pandas Series to a list, replacing NaN/Inf with None and rounding."""
    arr = s.to_numpy(dtype=np.float64, na_value=np.nan)
    out: list = [None] * len(arr)
    for i in range(len(arr)):
        v = arr[i]
        if np.isfinite(v):
            out[i] = round(float(v), 4)
    return out


def _clean_list(lst: list) -> list:
    out: list = [None] * len(lst)
    for i, v in enumerate(lst):
        if v is not None:
            try:
                if math.isfinite(v):
                    out[i] = round(float(v), 4)
            except (TypeError, ValueError):
                out[i] = v
    return out


# ── Vectorized Technical Indicators (pandas/numpy) ───────────────────────


def compute_sma_series(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(window=period, min_periods=period).mean()


def compute_ema_series(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def compute_rsi_series(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.iloc[:period] = np.nan
    return rsi


def compute_macd_series(s: pd.Series) -> dict[str, pd.Series]:
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {"macdLine": macd_line, "signalLine": signal_line, "histogram": histogram}


def compute_bollinger_series(
    s: pd.Series, period: int = 20, num_std: float = 2.0
) -> dict[str, pd.Series]:
    middle = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return {"upper": upper, "middle": middle, "lower": lower}


def compute_fibonacci(prices: pd.Series) -> dict[str, float]:
    high = float(prices.max())
    low = float(prices.min())
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


# ── Precomputed-indicator dict type ────────────────────────────────────────
# Strategies receive this dict so nothing is recomputed.
# Keys:  close (pd.Series), high, low, volume (pd.Series),
#        sma20, sma50, sma200 (pd.Series), rsi (pd.Series),
#        macd (dict of pd.Series), bollinger (dict of pd.Series)


def _volatility_vec(close: pd.Series, period: int = 20) -> float:
    ret = close.iloc[-period:].pct_change().dropna()
    if ret.empty:
        return 0.02
    return float(ret.std())


def strategy_trend_following(ind: dict[str, Any]) -> dict[str, Any]:
    close = ind["close"]
    sma20 = ind["sma20"]
    sma50 = ind["sma50"]
    macd_data = ind["macd"]

    current_price = float(close.iloc[-1])
    sma50_val = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    sma20_val = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None

    if sma50_val is None or sma20_val is None:
        return {
            "name": "Time-Series Trend Following", "icon": "📈",
            "description": "Identifies market trends using moving averages and MACD",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
            "reasoning": "Insufficient data to compute moving averages.",
        }

    trend_strength = ((current_price - sma50_val) / sma50_val) * 100.0
    histogram = float(macd_data["histogram"].iloc[-1])

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
            "MACD": f"{float(macd_data['macdLine'].iloc[-1]):.3f}",
            "Trend Strength": f"{trend_strength:.2f}%",
        },
    }


def strategy_multi_factor(ind: dict[str, Any]) -> dict[str, Any]:
    close = ind["close"]
    volumes = ind["volume"]
    n = len(close)
    current_price = float(close.iloc[-1])
    returns_30d = (current_price - float(close.iloc[-30])) / float(close.iloc[-30]) if n >= 30 else 0.0
    momentum_factor = 1 if returns_30d > 0.05 else (-1 if returns_30d < -0.05 else 0)

    avg_volume = float(volumes.mean()) if len(volumes) > 0 else 1.0
    current_vol = float(volumes.iloc[-1]) if len(volumes) > 0 else 1.0
    volume_ratio = current_vol / avg_volume if avg_volume else 1.0
    quality_factor = 1 if 0.8 < volume_ratio < 1.5 else -1

    vol = _volatility_vec(close)
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


def strategy_momentum(ind: dict[str, Any]) -> dict[str, Any]:
    close = ind["close"]
    rsi_s = ind["rsi"]
    n = len(close)
    current_price = float(close.iloc[-1])
    rsi = float(rsi_s.iloc[-1]) if pd.notna(rsi_s.iloc[-1]) else 50.0

    returns_1w = (current_price - float(close.iloc[-5])) / float(close.iloc[-5]) if n >= 5 else 0.0
    returns_1m = (current_price - float(close.iloc[-21])) / float(close.iloc[-21]) if n >= 21 else 0.0
    returns_3m = (current_price - float(close.iloc[-63])) / float(close.iloc[-63]) if n >= 63 else 0.0

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


def strategy_stat_arb(ind: dict[str, Any]) -> dict[str, Any]:
    close = ind["close"]
    sma20 = ind["sma20"]
    mean = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None
    current_price = float(close.iloc[-1])

    if mean is None or mean == 0:
        return {
            "name": "Statistical Arbitrage", "icon": "🎯",
            "description": "Mean reversion and pairs trading opportunities",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
            "reasoning": "Insufficient data to compute statistical bands.",
        }

    recent = close.iloc[-20:]
    std_dev = float(recent.std(ddof=0))

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


def strategy_ml_alpha(ind: dict[str, Any]) -> dict[str, Any]:
    close = ind["close"]
    volumes = ind["volume"]
    rsi_s = ind["rsi"]
    macd_data = ind["macd"]
    rsi = float(rsi_s.iloc[-1]) if pd.notna(rsi_s.iloc[-1]) else 50.0
    vol = _volatility_vec(close)

    avg_vol = float(volumes.iloc[-20:-1].mean()) if len(volumes) >= 20 else (float(volumes.mean()) if len(volumes) > 0 else 1.0)
    current_vol = float(volumes.iloc[-1]) if len(volumes) > 0 else 0.0
    volume_anomaly = current_vol > avg_vol * 1.5

    momentum = (float(close.iloc[-1]) - float(close.iloc[-5])) / float(close.iloc[-5]) if len(close) >= 5 else 0.0

    # Feature scores
    f_mom = momentum * 0.3
    f_rsi = (1 if rsi < 30 else (-1 if rsi > 70 else 0)) * 0.25
    f_macd = (1 if float(macd_data["histogram"].iloc[-1]) > 0 else -1) * 0.2
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

    feature_notes.append(f"MACD histogram is {'positive → bullish' if float(macd_data['histogram'].iloc[-1]) > 0 else 'negative → bearish'}")

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


def strategy_bollinger_squeeze(ind: dict[str, Any]) -> dict[str, Any]:
    """Detects Bollinger Band squeeze (low volatility) and breakout direction."""
    bb = ind["bollinger"]
    close = ind["close"]
    current_price = float(close.iloc[-1])

    upper_val = float(bb["upper"].iloc[-1]) if pd.notna(bb["upper"].iloc[-1]) else None
    lower_val = float(bb["lower"].iloc[-1]) if pd.notna(bb["lower"].iloc[-1]) else None
    middle_val = float(bb["middle"].iloc[-1]) if pd.notna(bb["middle"].iloc[-1]) else None

    if upper_val is None or lower_val is None or middle_val is None or middle_val == 0:
        return {
            "name": "Bollinger Band Squeeze", "icon": "🔄",
            "description": "Detects volatility contraction and imminent breakout direction",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
            "reasoning": "Insufficient data for Bollinger Band calculation.",
        }

    upper = upper_val
    lower = lower_val
    middle = middle_val
    bandwidth = ((upper - lower) / middle) * 100.0

    bw_series = ((bb["upper"] - bb["lower"]) / bb["middle"]).dropna().iloc[-10:] * 100.0
    avg_bw = float(bw_series.mean()) if len(bw_series) > 0 else bandwidth
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


def _compute_stochastic_vec(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
    """Vectorized Stochastic Oscillator %K and %D."""
    if len(close) < k_period:
        return 50.0, 50.0
    highest = high.rolling(window=k_period, min_periods=k_period).max()
    lowest = low.rolling(window=k_period, min_periods=k_period).min()
    denom = highest - lowest
    pct_k_series = ((close - lowest) / denom.replace(0, np.nan)) * 100.0
    pct_k_series = pct_k_series.fillna(50.0)
    pct_d_series = pct_k_series.rolling(window=d_period).mean()
    return float(pct_k_series.iloc[-1]), float(pct_d_series.iloc[-1])


def strategy_stochastic(ind: dict[str, Any]) -> dict[str, Any]:
    """Stochastic Oscillator strategy — overbought/oversold with %K/%D crossovers."""
    pct_k, pct_d = _compute_stochastic_vec(ind["high"], ind["low"], ind["close"])

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


def _compute_adx_vec(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[float, float, float]:
    """Vectorized ADX, +DI, -DI using pandas."""
    if len(close) < period + 1:
        return 25.0, 25.0, 25.0

    h_diff = high.diff()
    l_diff = -low.diff()
    plus_dm = pd.Series(np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0.0), index=close.index)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    atr_val = float(atr.iloc[-1])
    plus_di = (float(plus_di_smooth.iloc[-1]) / atr_val * 100) if atr_val != 0 else 0
    minus_di = (float(minus_di_smooth.iloc[-1]) / atr_val * 100) if atr_val != 0 else 0

    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) != 0 else 0
    return dx, plus_di, minus_di


def strategy_adx(ind: dict[str, Any]) -> dict[str, Any]:
    """ADX Trend Strength — measures how strong the current trend is."""
    adx, plus_di, minus_di = _compute_adx_vec(ind["high"], ind["low"], ind["close"])

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


def strategy_obv(ind: dict[str, Any]) -> dict[str, Any]:
    """On-Balance Volume — confirms price trends via volume flow."""
    close = ind["close"]
    volumes = ind["volume"]
    sma20 = ind["sma20"]

    if len(close) < 21 or len(volumes) < 21:
        return {
            "name": "OBV Volume Trend", "icon": "📊",
            "description": "Confirms price trends by analyzing cumulative volume flow",
            "signal": "NEUTRAL", "confidence": 30.0, "metrics": {},
            "reasoning": "Insufficient data for OBV calculation.",
        }

    # Vectorized OBV
    direction = np.sign(close.diff()).fillna(0)
    obv_series = (direction * volumes).cumsum()

    obv_sma_s = obv_series.rolling(window=20, min_periods=20).mean()
    current_obv = float(obv_series.iloc[-1])
    obv_sma_val = float(obv_sma_s.iloc[-1]) if pd.notna(obv_sma_s.iloc[-1]) else None

    sma20_val = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else float(close.iloc[-1])
    price_above_sma = float(close.iloc[-1]) > sma20_val

    obv_10d_change = float(obv_series.iloc[-1] - obv_series.iloc[-10]) if len(obv_series) >= 10 else 0
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
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict[str, Any]:
    """
    Fetch OHLCV data and compute comprehensive technical analysis
    for a single ticker.  All indicators are computed once (vectorized)
    and shared across strategies — zero redundant computation.
    """
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    # Fetch OHLCV and fundamentals in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        ohlcv_future = pool.submit(fetch_single_ticker_ohlcv, yahoo_ticker, start_date, end_date)
        info_future = pool.submit(fetch_ticker_info, yahoo_ticker)
        df = ohlcv_future.result()
        fundamentals = info_future.result()

    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Compute ALL indicators ONCE (vectorized) ──────────────────────────
    sma20 = compute_sma_series(close, 20)
    sma50 = compute_sma_series(close, 50)
    sma200 = compute_sma_series(close, 200)
    rsi = compute_rsi_series(close)
    macd = compute_macd_series(close)
    bollinger = compute_bollinger_series(close)

    # ── Shared indicator dict for all strategies ──────────────────────────
    ind = {
        "close": close, "high": high, "low": low, "volume": volume,
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "rsi": rsi, "macd": macd, "bollinger": bollinger,
    }

    # ── Trim to display range ─────────────────────────────────────────────
    start_str = start_date.isoformat()
    mask = df.index >= pd.Timestamp(start_str)
    d = df.loc[mask]
    dates = [ts.strftime("%Y-%m-%d") for ts in d.index]

    d_sma50 = sma50.loc[mask]
    d_sma200 = sma200.loc[mask]
    d_rsi = rsi.loc[mask]
    d_macd_line = macd["macdLine"].loc[mask]
    d_signal_line = macd["signalLine"].loc[mask]
    d_histogram = macd["histogram"].loc[mask]
    d_bb_upper = bollinger["upper"].loc[mask]
    d_bb_middle = bollinger["middle"].loc[mask]
    d_bb_lower = bollinger["lower"].loc[mask]

    # ── Fibonacci on display range ────────────────────────────────────────
    fibonacci = compute_fibonacci(d["Close"])

    # ── Current values ────────────────────────────────────────────────────
    current_price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100.0 if prev_close != 0 else 0.0

    d_volume = d["Volume"]
    avg_volume = float(d_volume.iloc[-20:].mean()) if len(d_volume) > 0 else 0
    current_volume = float(d_volume.iloc[-1]) if len(d_volume) > 0 else 0

    latest_rsi_val = rsi.dropna()
    latest_rsi = round(float(latest_rsi_val.iloc[-1]), 2) if len(latest_rsi_val) > 0 else None

    # ── Crossover status ──────────────────────────────────────────────────
    dma50_val = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    dma200_val = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None
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

    # ── Strategies (all receive pre-computed indicators) ──────────────────
    strategies = [
        strategy_trend_following(ind),
        strategy_multi_factor(ind),
        strategy_momentum(ind),
        strategy_stat_arb(ind),
        strategy_bollinger_squeeze(ind),
        strategy_stochastic(ind),
        strategy_adx(ind),
        strategy_obv(ind),
        strategy_ml_alpha(ind),
    ]

    # ── Serialize volume safely ───────────────────────────────────────────
    vol_arr = d_volume.to_numpy()
    safe_vol = [int(v) if np.isfinite(v) else 0 for v in vol_arr]

    # ── Build response ────────────────────────────────────────────────────
    return {
        "ticker": yahoo_ticker.replace("-", "."),
        "companyName": company_name,
        "sector": sector,
        "dateRangeStart": start_date.isoformat(),
        "dateRangeEnd": end_date.isoformat(),
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
            "open": _clean_series(d["Open"]),
            "high": _clean_series(d["High"]),
            "low": _clean_series(d["Low"]),
            "close": _clean_series(d["Close"]),
            "volume": safe_vol,
        },
        "indicators": {
            "sma50": _clean_series(d_sma50),
            "sma200": _clean_series(d_sma200),
            "rsi": _clean_series(d_rsi),
            "macd": {
                "macdLine": _clean_series(d_macd_line),
                "signalLine": _clean_series(d_signal_line),
                "histogram": _clean_series(d_histogram),
            },
            "bollinger": {
                "upper": _clean_series(d_bb_upper),
                "middle": _clean_series(d_bb_middle),
                "lower": _clean_series(d_bb_lower),
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
