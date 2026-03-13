"""
Shared technical indicator computations (single source of truth for formulas).
Used by research.py for single-ticker indicators; rsi_scan.py uses the same
RSI formula (Wilder EWM, alpha=1/period) in vectorized batch form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sma_series(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(window=period, min_periods=period).mean()


def compute_ema_series(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def compute_rsi_series(s: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI: alpha=1/period, RS = avg_gain/avg_loss, RSI = 100 - 100/(1+RS)."""
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
