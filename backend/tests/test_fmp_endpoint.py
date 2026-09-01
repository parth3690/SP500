"""
Test that FMP endpoints are using the correct paths (no 404s).
"""
from unittest.mock import Mock, patch
import httpx


def test_nyse_smid_uses_company_screener_not_stock_screener():
    """Verify NYSE SMID uses /company-screener (working) not /stock-screener (404)."""
    from backend.app.services.sp500 import _fetch_nyse_smid_from_fmp
    
    with patch("backend.app.services.sp500.httpx.get") as mock_get:
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=[
            {
                "symbol": "TEST",
                "companyName": "Test Corp",
                "marketCap": 500e6,
                "price": 50.0,
                "exchangeShortName": "NYSE",
            }
        ])
        mock_get.return_value = mock_response
        
        # Call the function
        result = _fetch_nyse_smid_from_fmp(100e6, 2e9, 2.0, 1e6)
        
        # Verify it called the correct endpoint
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        
        # Check the URL uses company-screener, not stock-screener
        assert "/company-screener" in call_args[0][0]
        assert "/stock-screener" not in call_args[0][0]
        
        # Verify the result
        assert len(result) == 1
        assert result[0].ticker == "TEST"


def test_nyse_smid_market_cap_filter():
    """Verify market cap band: 100M < cap < 2B."""
    from backend.app.services.sp500 import _fetch_nyse_smid_from_fmp
    
    with patch("backend.app.services.sp500.httpx.get") as mock_get:
        # Mock response with stocks at different cap levels
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=[
            {
                "symbol": "TOO_SMALL",
                "companyName": "Too Small Inc",
                "marketCap": 50e6,  # $50M - below threshold
                "price": 10.0,
                "exchangeShortName": "NYSE",
            },
            {
                "symbol": "VALID",
                "companyName": "Valid Corp",
                "marketCap": 500e6,  # $500M - in range
                "price": 50.0,
                "exchangeShortName": "NYSE",
            },
            {
                "symbol": "TOO_BIG",
                "companyName": "Too Big Corp",
                "marketCap": 5e9,  # $5B - above threshold
                "price": 150.0,
                "exchangeShortName": "NYSE",
            },
        ])
        mock_get.return_value = mock_response
        
        result = _fetch_nyse_smid_from_fmp(100e6, 2e9, 2.0, 1e6)
        
        # Only VALID should pass
        assert len(result) == 1
        assert result[0].ticker == "VALID"


def test_nyse_smid_filters_etfs_and_funds():
    """Verify ETFs/funds/ADRs/preferreds/warrants are excluded."""
    from backend.app.services.sp500 import _fetch_nyse_smid_from_fmp
    
    with patch("backend.app.services.sp500.httpx.get") as mock_get:
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value=[
            {
                "symbol": "FUND",
                "companyName": "Some Fund",
                "marketCap": 500e6,
                "price": 50.0,
                "exchangeShortName": "NYSE",
                "isEtf": False,
                "isFund": True,  # Marked as fund
            },
            {
                "symbol": "ETF",
                "companyName": "Some ETF",
                "marketCap": 500e6,
                "price": 50.0,
                "exchangeShortName": "NYSE",
                "isEtf": True,  # Marked as ETF
                "isFund": False,
            },
            {
                "symbol": "VALID",
                "companyName": "Valid Corp",
                "marketCap": 500e6,
                "price": 50.0,
                "exchangeShortName": "NYSE",
                "isEtf": False,
                "isFund": False,
            },
            {
                "symbol": "HQL",
                "companyName": "Tekla Healthcare Opportunities Fund",  # Real closed-end fund
                "marketCap": 500e6,
                "price": 50.0,
                "exchangeShortName": "NYSE",
            },
        ])
        mock_get.return_value = mock_response
        
        result = _fetch_nyse_smid_from_fmp(100e6, 2e9, 2.0, 1e6)
        
        # Only VALID should pass; FUND and ETF excluded by isEtf/isFund
        # HQL excluded by name containing "fund"
        tickers = {c.ticker for c in result}
        assert "VALID" in tickers
        assert "FUND" not in tickers
        assert "ETF" not in tickers
        assert "HQL" not in tickers
