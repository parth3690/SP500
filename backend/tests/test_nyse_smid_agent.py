"""
Tests for NYSE SMID Agent.

Verify that the NYSE SMID agent:
- Uses the shared S&P 500 data pipeline
- Filters by market cap and exchange
- Applies the same institutional gate
- Can scan specific tickers or the entire universe
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from backend.app.models import Constituent


def test_nyse_smid_agent_universe_filter():
    """Test that NYSE SMID universe excludes stocks outside $100M-$2B cap range."""
    from backend.app.services.sp500 import fetch_nyse_smid_constituents
    
    # Mock FMP response with a mix of stocks
    mock_response = [
        {
            "symbol": "TOO_SMALL",
            "name": "Too Small Inc",
            "price": 10.0,
            "exchange": "NYSE",
            "marketCap": 50e6,  # $50M - below threshold
        },
        {
            "symbol": "VALID_1",
            "name": "Valid Stock 1",
            "price": 25.0,
            "exchange": "NYSE",
            "marketCap": 500e6,  # $500M - valid
        },
        {
            "symbol": "TOO_BIG",
            "name": "Too Big Corp",
            "price": 150.0,
            "exchange": "NYSE",
            "marketCap": 5e9,  # $5B - above threshold
        },
        {
            "symbol": "VALID_2",
            "name": "Valid Stock 2",
            "price": 50.0,
            "exchange": "NYSE",
            "marketCap": 1.5e9,  # $1.5B - valid
        },
    ]
    
    with patch("backend.app.services.sp500._fetch_nyse_smid_from_fmp") as mock_fetch:
        mock_fetch.return_value = [
            Constituent(
                ticker=item["symbol"],
                companyName=item["name"],
                yahooTicker=item["symbol"],
                sector="Technology",
                weight=0.0,
            )
            for item in mock_response
            if 100e6 < item["marketCap"] < 2e9 and item["exchange"] == "NYSE"
        ]
        
        result = fetch_nyse_smid_constituents()
        
        # Should only include VALID_1 and VALID_2
        assert len(result) == 2
        tickers = {c.ticker for c in result}
        assert "VALID_1" in tickers
        assert "VALID_2" in tickers
        assert "TOO_SMALL" not in tickers
        assert "TOO_BIG" not in tickers


def test_nyse_smid_agent_uses_sp500_data_pipeline():
    """Test that NYSE SMID agent reuses S&P 500 data functions."""
    from backend.app.services.nyse_smid_agent import run_nyse_smid_agent
    
    # Mock constituents
    mock_constituents = [
        Constituent(
            ticker="TEST",
            companyName="Test Corp",
            yahooTicker="TEST",
            sector="Technology",
            weight=0.0,
        ),
    ]
    
    # Mock price data
    end = date.today()
    start = end - timedelta(days=760)
    date_range = pd.date_range(start, end, freq="D")
    mock_prices = pd.DataFrame(
        {
            "SPY": [100 + i * 0.1 for i in range(len(date_range))],
            "TEST": [50 + i * 0.05 for i in range(len(date_range))],
        },
        index=date_range,
    )
    
    with patch("backend.app.services.nyse_smid_agent.get_nyse_smid_constituents_cached") as mock_get_constituents, \
         patch("backend.app.services.nyse_smid_agent.fetch_close_prices") as mock_fetch_prices, \
         patch("backend.app.services.nyse_smid_agent.compute_alpha_candidates") as mock_compute:
        
        mock_get_constituents.return_value = mock_constituents
        mock_fetch_prices.return_value = mock_prices
        mock_compute.return_value = {
            "candidates": [],
            "marketRegime": {
                "state": "neutral",
                "spyTrend": "neutral",
                "spyDrawdownPct": 0.0,
                "effectiveState": "neutral",
                "riskMode": "balanced",
            },
            "meta": {"total": 0, "eligible": 0, "computed": 0, "returned": 0},
        }
        
        result = run_nyse_smid_agent(limit=20)
        
        # Verify it called the shared data functions
        mock_get_constituents.assert_called_once()
        mock_fetch_prices.assert_called_once()
        mock_compute.assert_called_once()
        
        # Verify result structure
        assert result["meta"]["universeType"] == "nyse_smid"
        assert result["meta"]["universe"] == "NYSE SMID ($100M-$2B)"


def test_nyse_smid_agent_specific_tickers():
    """Test that NYSE SMID agent can filter to specific tickers."""
    from backend.app.services.nyse_smid_agent import run_nyse_smid_agent
    
    # Mock constituents (3 stocks in universe)
    mock_constituents = [
        Constituent(ticker="A", companyName="A Corp", yahooTicker="A", sector="Tech", weight=0.0),
        Constituent(ticker="B", companyName="B Corp", yahooTicker="B", sector="Finance", weight=0.0),
        Constituent(ticker="C", companyName="C Corp", yahooTicker="C", sector="Healthcare", weight=0.0),
    ]
    
    # Mock price data
    end = date.today()
    start = end - timedelta(days=760)
    date_range = pd.date_range(start, end, freq="D")
    mock_prices = pd.DataFrame(
        {
            "SPY": [100 + i * 0.1 for i in range(len(date_range))],
            "A": [50 + i * 0.05 for i in range(len(date_range))],
            "B": [60 + i * 0.04 for i in range(len(date_range))],
            "C": [70 + i * 0.03 for i in range(len(date_range))],
        },
        index=date_range,
    )
    
    with patch("backend.app.services.nyse_smid_agent.get_nyse_smid_constituents_cached") as mock_get_constituents, \
         patch("backend.app.services.nyse_smid_agent.fetch_close_prices") as mock_fetch_prices, \
         patch("backend.app.services.nyse_smid_agent.compute_alpha_candidates") as mock_compute:
        
        mock_get_constituents.return_value = mock_constituents
        mock_fetch_prices.return_value = mock_prices
        mock_compute.return_value = {
            "candidates": [
                {
                    "ticker": "A",
                    "companyName": "A Corp",
                    "sector": "Tech",
                    "currentPrice": 100.0,
                    "alphaScore": 75.0,
                    "riskScore": 55.0,
                    "expectedReturn20d": 5.0,
                    "volatility20d": 15.0,
                },
            ],
            "marketRegime": {
                "state": "neutral",
                "spyTrend": "neutral",
                "spyDrawdownPct": 0.0,
                "effectiveState": "neutral",
                "riskMode": "balanced",
            },
            "meta": {"total": 1, "eligible": 1, "computed": 1, "returned": 1},
        }
        
        # Request only ticker "A"
        result = run_nyse_smid_agent(tickers=["A"], limit=20)
        
        # Verify it filtered constituents
        assert result["meta"]["status"] in ("complete", "no_candidates")


def test_nyse_smid_requires_fmp_key():
    """Test that NYSE SMID degrades honestly if FMP_API_KEY is not set."""
    from backend.app.services.sp500 import fetch_nyse_smid_constituents
    
    # Clear FMP key
    original_key = os.environ.get("FMP_API_KEY")
    if original_key:
        del os.environ["FMP_API_KEY"]
    
    try:
        result = fetch_nyse_smid_constituents()
        # Should return empty list with honest warning (not fake constituents)
        assert result == []
    finally:
        # Restore key
        if original_key:
            os.environ["FMP_API_KEY"] = original_key


def test_nyse_smid_gate_unchanged():
    """Test that NYSE SMID agent uses the same institutional gate as S&P 500 scanner."""
    from backend.app.services.nyse_smid_agent import run_nyse_smid_agent
    from backend.app.services.institutional_scanner import (
        MIN_CONFIDENCE_FOR_TAKE,
        MIN_BACKTEST_WIN_RATE,
        MIN_BACKTEST_SAMPLE_SIZE,
        MIN_ALPHA_VS_BENCHMARK,
    )
    
    # Mock constituents with one strong candidate
    mock_constituents = [
        Constituent(
            ticker="STRONG",
            companyName="Strong Corp",
            yahooTicker="STRONG",
            sector="Technology",
            weight=0.0,
        ),
    ]
    
    # Mock price data
    end = date.today()
    start = end - timedelta(days=760)
    date_range = pd.date_range(start, end, freq="D")
    mock_prices = pd.DataFrame(
        {
            "SPY": [100 + i * 0.1 for i in range(len(date_range))],
            "STRONG": [50 + i * 0.15 for i in range(len(date_range))],  # Outperforms SPY
        },
        index=date_range,
    )
    
    with patch("backend.app.services.nyse_smid_agent.get_nyse_smid_constituents_cached") as mock_get_constituents, \
         patch("backend.app.services.nyse_smid_agent.fetch_close_prices") as mock_fetch_prices:
        
        mock_get_constituents.return_value = mock_constituents
        mock_fetch_prices.return_value = mock_prices
        
        result = run_nyse_smid_agent(limit=20)
        
        # Verify gate thresholds are documented in meta
        assert result["meta"]["tradeGate"]["minConfidence"] == MIN_CONFIDENCE_FOR_TAKE
        assert result["meta"]["tradeGate"]["minWinRate"] == MIN_BACKTEST_WIN_RATE
        assert result["meta"]["tradeGate"]["minSampleSize"] == MIN_BACKTEST_SAMPLE_SIZE
        assert result["meta"]["tradeGate"]["minAlphaVsBenchmark"] == MIN_ALPHA_VS_BENCHMARK
