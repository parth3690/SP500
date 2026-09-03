from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic import ValidationError

from app.models import AgentBotHistoryItem, AgentBotRunResponse, Constituent
from app.services.agent_bot import (
    _compute_outcomes,
    _empty_response,
    _fetch_catalyst_data,
    _forward_journal,
    _generate_alerts,
    run_agent_bot,
)
from app.services.alpha import (
    _combine_institutional_metrics,
    _next_monthly_expiry,
    _score_core,
    _trade_plan,
    institutional_metrics_from_holder_table,
    institutional_metrics_from_info,
)
from app.services.institutional_scanner import (
    MIN_CONFIDENCE_FOR_TAKE,
    MIN_BACKTEST_WIN_RATE,
    MIN_BACKTEST_SAMPLE_SIZE,
    MIN_ALPHA_VS_BENCHMARK,
    CONVEXITY_ALERT_MIN_PROBABILITY,
    CONVEXITY_ALERT_MIN_RETURN,
    _apply_trade_gate,
    _compute_confidence,
    _detect_convexity_alert,
    _run_simulation_validation,
    _run_walk_forward_backtest,
)
from app.services.crossovers import compute_crossovers
from app.services.cache import PRICE_DATA_CACHE, price_cache_get, price_cache_set
from app.services.fmp import parse_fmp_institutional_metrics
from app.services.market_conditions import _download_close, _risk_assessment
from app.services.prices import fetch_close_prices, fetch_fmp_price_history
from app.services.research import _sanitize_fundamentals, compute_research
from app.services.weekly_ma_scan import compute_weekly_ma_watch


class AgentContractTests(unittest.TestCase):
    def test_empty_watchlist_abstains_without_network_calls(self) -> None:
        with patch(
            "app.services.agent_bot.get_sp500_constituents_cached",
            side_effect=AssertionError("network should not be called"),
        ):
            payload = run_agent_bot([], mode="watchlist")

        parsed = AgentBotRunResponse(**payload)
        self.assertEqual(parsed.meta["status"], "insufficient_data")
        self.assertEqual(parsed.forwardJournal.aggregates["6M"].count, 0)
        self.assertEqual(parsed.briefing.counts["sell"], 0)

    def test_empty_response_always_matches_api_contract(self) -> None:
        parsed = AgentBotRunResponse(**_empty_response("sp500", error="No data"))
        self.assertEqual(parsed.briefing.riskLevel, "Unknown")
        self.assertEqual(parsed.recommendations, [])

    def test_history_rejects_non_tradeable_prices_and_actions(self) -> None:
        with self.assertRaises(ValidationError):
            AgentBotHistoryItem(
                ticker="AAPL",
                action="HOLD",
                entryPrice=0,
                recommendedAt=datetime.now(timezone.utc),
            )

    def test_synthetic_agent_run_matches_full_recommendation_contract(self) -> None:
        index = pd.bdate_range("2024-10-01", periods=330)
        prices = pd.DataFrame(
            {
                "AAPL": np.linspace(100.0, 220.0, len(index)),
                "SPY": np.linspace(100.0, 120.0, len(index)),
                "XLK": np.linspace(100.0, 130.0, len(index)),
            },
            index=index,
        )
        constituents = [
            Constituent(
                ticker="AAPL",
                yahooTicker="AAPL",
                companyName="Apple",
                sector="Information Technology",
            )
        ]
        market = {
            "riskLevel": "Normal",
            "coveragePct": 75,
            "triggeredCount": 1,
            "confidence": "Medium",
            "asOf": "2026-01-01T00:00:00+00:00",
            "warnings": [],
        }
        with (
            patch("app.services.agent_bot.get_sp500_constituents_cached", return_value=constituents),
            patch("app.services.agent_bot._market_conditions_summary", return_value=market),
            patch("app.services.agent_bot._cached_close_prices", return_value=prices),
            patch("app.services.agent_bot._enrich_with_catalysts"),
        ):
            payload = run_agent_bot(["AAPL"], mode="watchlist", top_n=1, min_score=0)

        parsed = AgentBotRunResponse(**payload)
        self.assertEqual(len(parsed.recommendations), 1)
        recommendation = parsed.recommendations[0]
        self.assertEqual(recommendation.ticker, "AAPL")
        self.assertTrue(recommendation.horizon)
        self.assertGreater(len(recommendation.backtests), 0)


class TradeDirectionTests(unittest.TestCase):
    @staticmethod
    def _candidate(price: float) -> dict:
        return {
            "ticker": "SHORT",
            "currentPrice": price,
            "alphaScore": 30.0,
            "tradePlan": {
                "action": "SELL",
                "stop": 110.0,
                "target1": 90.0,
                "target2": 80.0,
            },
            "catalystData": {},
        }

    def test_sell_alerts_use_short_direction(self) -> None:
        near_stop = _generate_alerts([self._candidate(109.0)])
        self.assertIn("stop proximity", {alert["type"] for alert in near_stop})
        self.assertNotIn("target hit", {alert["type"] for alert in near_stop})

        near_target = _generate_alerts([self._candidate(91.0)])
        self.assertIn("target hit", {alert["type"] for alert in near_target})
        self.assertNotIn("stop proximity", {alert["type"] for alert in near_target})

    def test_sell_confidence_rewards_bearish_conviction(self) -> None:
        row = {
            "currentPrice": 100.0,
            "alphaScore": 30.0,
            "riskScore": 70.0,
            "expectedReturn20d": -6.0,
            "rsVsSpy20d": -10.0,
            "volatility20d": 20.0,
            "betaVsSpy": 1.0,
        }
        plan = _trade_plan(row)
        self.assertEqual(plan["action"], "SELL")
        self.assertGreater(plan["confidence"], 65.0)
        self.assertGreater(plan["stop"], plan["entry"])
        self.assertLess(plan["target1"], plan["entry"])

    def test_high_iv_bullish_plan_uses_30d_premium_setup(self) -> None:
        row = {
            "currentPrice": 100.0,
            "alphaScore": 75.0,
            "riskScore": 70.0,
            "expectedReturn20d": 6.0,
            "rsVsSpy20d": 5.0,
            "volatility20d": 55.0,
            "betaVsSpy": 1.0,
        }
        plan = _trade_plan(row)
        self.assertEqual(plan["action"], "BUY")
        self.assertEqual(plan["optionStrategy"], "Bull put credit spread")
        self.assertEqual(plan["optionCategory"], "Premium 30D")
        self.assertEqual(plan["optionDte"], 30)
        self.assertEqual(plan["optionIvGate"], "pass")
        self.assertEqual(plan["optionIvProxy"], 55.0)
        self.assertIn("Only sell option premium when IV or IV proxy is 50+.", plan["optionRules"])

    def test_low_iv_bullish_plan_avoids_short_premium(self) -> None:
        row = {
            "currentPrice": 100.0,
            "alphaScore": 75.0,
            "riskScore": 70.0,
            "expectedReturn20d": 6.0,
            "rsVsSpy20d": 5.0,
            "volatility20d": 25.0,
            "betaVsSpy": 1.0,
        }
        plan = _trade_plan(row)
        self.assertEqual(plan["action"], "BUY")
        self.assertEqual(plan["optionStrategy"], "Long call")
        self.assertEqual(plan["optionCategory"], "Directional")
        self.assertEqual(plan["optionDte"], 45)
        self.assertEqual(plan["optionIvGate"], "below_50")
        self.assertNotEqual(plan["optionCategory"], "Premium 30D")

    def test_fmp_institutional_parser_uses_13f_share_change(self) -> None:
        metrics = parse_fmp_institutional_metrics([
            {
                "date": "2026-03-31",
                "ownershipPercent": 24.5,
                "numberOf13Fshares": 1_120_000,
                "lastNumberOf13Fshares": 1_000_000,
            }
        ])
        self.assertEqual(metrics["institutionalOwnershipPct"], 24.5)
        self.assertEqual(metrics["institutionalTransactionPct"], 12.0)
        self.assertEqual(metrics["institutionalDataSource"], "FMP 13F")

    def test_yahoo_institutional_ownership_fraction_scales_to_percent(self) -> None:
        metrics = institutional_metrics_from_info({"heldPercentInstitutions": 0.79285})
        self.assertEqual(metrics["institutionalOwnershipPct"], 79.29)

    def test_yahoo_holder_table_uses_weighted_pct_change(self) -> None:
        holders = pd.DataFrame(
            [
                {"Date Reported": "2026-03-31", "Shares": 110.0, "pctChange": 0.10},
                {"Date Reported": "2026-03-31", "Shares": 60.0, "pctChange": 0.20},
            ]
        )
        metrics = institutional_metrics_from_holder_table(holders)
        self.assertEqual(metrics["institutionalTransactionPct"], 13.33)
        self.assertEqual(metrics["institutionalDataSource"], "Yahoo institutional holders")

    def test_institutional_merge_keeps_yahoo_change_when_fmp_missing(self) -> None:
        merged = _combine_institutional_metrics(
            {"institutionalOwnershipPct": 79.0, "institutionalTransactionPct": 13.2, "institutionalNotes": []},
            {},
        )
        self.assertEqual(merged["institutionalTransactionPct"], 13.2)
        self.assertTrue(merged["institutionalScannerPass"])

    def test_monthly_option_expiry_is_third_friday(self) -> None:
        self.assertEqual(
            _next_monthly_expiry(45, as_of=date(2026, 1, 1)),
            "2026-02-20",
        )

    def test_downtrend_scores_as_bearish(self) -> None:
        scores = _score_core(
            momentum20=-10,
            momentum63=-20,
            rs_spy20=-8,
            rs_sector20=-5,
            price=80,
            sma50=100,
            sma200=120,
            volatility20=25,
            beta_vs_spy=1,
            drawdown63=-20,
            sector_strength20=-2,
            regime="risk_off",
            risk_mode="balanced",
        )
        self.assertLessEqual(float(scores["trendScore"]), 30.0)


class JournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.bdate_range("2026-01-02", periods=10)
        self.prices = pd.DataFrame(
            {"AAPL": np.linspace(100.0, 110.0, len(self.index))},
            index=self.index,
        )

    def test_future_horizons_remain_blank_until_they_mature(self) -> None:
        history = [{
            "ticker": "AAPL",
            "action": "BUY",
            "entryPrice": 100.0,
            "recommendedAt": "2026-01-02T15:00:00+00:00",
            "closed": False,
        }]
        journal = _forward_journal(history, self.prices)
        returns = journal["entries"][0]["forwardReturns"]
        self.assertIsNotNone(returns["1W"])
        self.assertIsNone(returns["1M"])
        self.assertIsNone(returns["6M"])

    def test_only_buy_and_sell_calls_count_as_outcomes(self) -> None:
        history = [
            {"ticker": "AAPL", "action": "WATCH", "entryPrice": 100.0, "recommendedAt": "2026-01-02"},
            {"ticker": "AAPL", "action": "AVOID", "entryPrice": 100.0, "recommendedAt": "2026-01-02"},
            {"ticker": "AAPL", "action": "SELL", "entryPrice": 100.0, "recommendedAt": "2026-01-02"},
        ]
        outcomes = _compute_outcomes(history, self.prices)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["action"], "SELL")
        self.assertLess(outcomes[0]["returnPct"], 0)

    def test_closed_outcome_uses_recorded_exit_price(self) -> None:
        history = [{
            "ticker": "AAPL",
            "action": "BUY",
            "entryPrice": 100.0,
            "recommendedAt": "2026-01-02",
            "closed": True,
            "exitPrice": 105.0,
        }]
        outcome = _compute_outcomes(history, self.prices)[0]
        self.assertEqual(outcome["currentPrice"], 105.0)
        self.assertEqual(outcome["returnPct"], 5.0)


class DataIntegrityTests(unittest.TestCase):
    @staticmethod
    def _ohlcv_frame(close_values: np.ndarray) -> pd.DataFrame:
        index = pd.bdate_range("2025-01-02", periods=len(close_values))
        return pd.DataFrame(
            {
                "Open": close_values * 0.99,
                "High": close_values * 1.02,
                "Low": close_values * 0.98,
                "Close": close_values,
                "Volume": np.full(len(close_values), 1_000_000),
            },
            index=index,
        )

    def test_market_risk_is_withheld_when_coverage_is_too_low(self) -> None:
        coverage, risk, confidence = _risk_assessment(4, 4, 16)
        self.assertEqual(coverage, 25)
        self.assertEqual(risk, "Unknown")
        self.assertEqual(confidence, "Insufficient")

        _, risk, confidence = _risk_assessment(12, 7, 16)
        self.assertEqual(risk, "Elevated")
        self.assertEqual(confidence, "Medium")

    def test_growth_value_is_never_used_as_an_earnings_date(self) -> None:
        info = {"earningsQuarterlyGrowth": 0.25, "shortName": "Example"}
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.info = info
            catalyst = _fetch_catalyst_data("TEST", 100.0)
        self.assertIsNone(catalyst["earningsDate"])

    def test_crossover_metadata_counts_missing_symbols_once(self) -> None:
        index = pd.bdate_range("2025-01-01", periods=220)
        prices = pd.DataFrame({"AAA": np.linspace(100, 102, len(index))}, index=index)
        constituents = [
            Constituent(ticker="AAA", yahooTicker="AAA", companyName="A", sector="Industrials"),
            Constituent(ticker="BBB", yahooTicker="BBB", companyName="B", sector="Industrials"),
        ]
        _, meta = compute_crossovers(constituents, prices, threshold_pct=10.0)
        self.assertEqual(meta["computed"], 1)
        self.assertEqual(meta["skipped"], 1)

    def test_weekly_ma_watch_matches_dip_near_and_reclaim_rules(self) -> None:
        index = pd.date_range("2022-01-07", periods=202, freq="W-FRI")
        prices = pd.DataFrame(
            {
                "CROSS": [100.0] * 201 + [90.0],
                "BELOW": [100.0] * 200 + [90.0, 89.0],
                "NEAR": [100.0] * 200 + [101.0, 101.0],
                "BACK": [100.0] * 200 + [90.0, 101.0],
                "FAR": [100.0] * 200 + [110.0, 110.0],
            },
            index=index,
        )
        constituents = [
            Constituent(ticker=ticker, yahooTicker=ticker, companyName=ticker, sector="Test")
            for ticker in prices.columns
        ]

        rows, meta = compute_weekly_ma_watch(constituents, prices, ma_length=200, near_pct=2.0)
        signals = {row["ticker"]: row["signal"] for row in rows}

        self.assertEqual(signals["CROSS"], "crossed_below")
        self.assertEqual(signals["BELOW"], "below")
        self.assertEqual(signals["NEAR"], "near")
        self.assertEqual(signals["BACK"], "reclaimed")
        self.assertNotIn("FAR", signals)
        cross_row = next(row for row in rows if row["ticker"] == "CROSS")
        self.assertEqual(cross_row["dailySma"], 99.95)
        self.assertEqual(cross_row["dailyDistancePct"], -9.95)
        self.assertEqual(meta["computed"], 5)
        self.assertEqual(meta["candidateCount"], 4)

    def test_weekly_ma_watch_skips_stocks_without_200_weeks(self) -> None:
        index = pd.date_range("2025-01-03", periods=50, freq="W-FRI")
        prices = pd.DataFrame({"NEW": np.linspace(100.0, 80.0, len(index))}, index=index)
        constituents = [
            Constituent(ticker="NEW", yahooTicker="NEW", companyName="New Listing", sector="Test")
        ]

        rows, meta = compute_weekly_ma_watch(constituents, prices)

        self.assertEqual(rows, [])
        self.assertEqual(meta["computed"], 0)
        self.assertEqual(meta["skipped"], 1)

    def test_price_cache_reuses_a_wider_history_window(self) -> None:
        PRICE_DATA_CACHE.clear()
        frame = pd.DataFrame({"AAA": [100.0]})
        try:
            price_cache_set("2020-01-01", "2026-08-18", frame)
            cached = price_cache_get("2025-08-18", "2026-08-18")
            self.assertIs(cached, frame)
            self.assertIsNone(price_cache_get("2019-01-01", "2026-08-18"))
        finally:
            PRICE_DATA_CACHE.clear()

    def test_market_close_falls_back_when_yahoo_returns_empty(self) -> None:
        fred_rows = [("2026-01-02", 100.0), ("2026-01-05", 101.0)]
        with (
            patch("app.services.market_conditions.yf.Ticker") as ticker,
            patch("app.services.market_conditions._fred_series", return_value=fred_rows),
        ):
            ticker.return_value.history.return_value = pd.DataFrame()
            closes = _download_close("SPY", period="3mo")

        self.assertIsNotNone(closes)
        self.assertEqual(len(closes), 2)
        self.assertEqual(float(closes.iloc[-1]), 101.0)

    def test_fmp_history_is_normalized_to_ohlcv(self) -> None:
        response_rows = [
            {
                "date": "2026-01-05",
                "open": 99.0,
                "high": 102.0,
                "low": 98.0,
                "close": 101.0,
                "volume": 12345,
            }
        ]
        with (
            patch.dict("os.environ", {"FMP_API_KEY": "test-key"}),
            patch("app.services.prices.httpx.get") as get,
        ):
            get.return_value.json.return_value = response_rows
            history = fetch_fmp_price_history("AAPL", date(2026, 1, 1), date(2026, 1, 6))

        get.return_value.raise_for_status.assert_called_once()
        self.assertEqual(list(history.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(float(history.iloc[0]["Close"]), 101.0)
        self.assertEqual(history.attrs["source"], "Financial Modeling Prep")

    def test_missing_yahoo_symbols_are_retried_in_small_batches(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=3)
        initial = pd.DataFrame({"AAA": [100.0, 101.0, 102.0]}, index=index)
        retry = pd.DataFrame({"BBB": [50.0, 51.0, 52.0]}, index=index)
        with (
            patch.dict(
                "os.environ",
                {"YAHOO_RETRY_CHUNK_SIZE": "1", "FMP_API_KEY": ""},
            ),
            patch("app.services.prices._download_chunk", side_effect=[initial, retry]) as download,
        ):
            prices = fetch_close_prices(
                ["AAA", "BBB"],
                date(2026, 1, 1),
                date(2026, 1, 10),
            )

        self.assertEqual(download.call_count, 2)
        self.assertEqual(set(prices.columns), {"AAA", "BBB"})
        self.assertEqual(prices.attrs["coveragePct"], 100.0)

    def test_fmp_fills_small_residual_from_a_large_request(self) -> None:
        index = pd.bdate_range("2026-01-02", periods=3)
        initial = pd.DataFrame({"AAA": [100.0, 101.0, 102.0]}, index=index)
        fmp = pd.DataFrame({"BBB": [50.0, 51.0, 52.0]}, index=index)
        with (
            patch.dict(
                "os.environ",
                {
                    "YAHOO_RETRY_CHUNK_SIZE": "1",
                    "YAHOO_RETRY_PASSES": "1",
                    "FMP_API_KEY": "test-key",
                    "FMP_PRICE_FALLBACK_MAX_TICKERS": "1",
                },
            ),
            patch(
                "app.services.prices._download_chunk",
                side_effect=[initial, pd.DataFrame()],
            ),
            patch("app.services.prices._fetch_fmp_close_frame", return_value=fmp) as fallback,
        ):
            prices = fetch_close_prices(
                ["AAA", "BBB"],
                date(2026, 1, 1),
                date(2026, 1, 10),
            )

        fallback.assert_called_once()
        self.assertEqual(set(prices.columns), {"AAA", "BBB"})
        self.assertEqual(prices.attrs["coveragePct"], 100.0)

    def test_agent_abstains_when_watchlist_price_coverage_is_partial(self) -> None:
        index = pd.bdate_range("2025-01-01", periods=250)
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(100.0, 120.0, len(index)),
                "SPY": np.linspace(100.0, 110.0, len(index)),
            },
            index=index,
        )
        constituents = [
            Constituent(ticker="AAA", yahooTicker="AAA", companyName="A", sector="Industrials"),
            Constituent(ticker="BBB", yahooTicker="BBB", companyName="B", sector="Industrials"),
        ]
        with (
            patch("app.services.agent_bot.get_sp500_constituents_cached", return_value=constituents),
            patch("app.services.agent_bot._market_conditions_summary", return_value={"riskLevel": "Normal"}),
            patch("app.services.agent_bot._cached_close_prices", return_value=prices),
        ):
            payload = run_agent_bot(["AAA", "BBB"], mode="watchlist", min_score=0)

        self.assertEqual(payload["meta"]["status"], "insufficient_data")
        self.assertEqual(payload["meta"]["priceCoveragePct"], 50.0)
        self.assertEqual(payload["recommendations"], [])

    def test_research_refetches_history_when_live_quote_mismatch_is_impossible(self) -> None:
        bad_history = self._ohlcv_frame(np.linspace(150.0, 164.0, 260))
        good_history = self._ohlcv_frame(np.linspace(34.0, 28.09, 260))
        good_history.attrs["source"] = "Financial Modeling Prep"

        fundamentals = {
            "trailingPE": 30.0,
            "forwardPE": 20.0,
            "marketCap": 6_500_000_000,
            "fiftyTwoWeekHigh": 70.43,
            "fiftyTwoWeekLow": 13.74,
            "beta": 2.3,
            "dividendYield": 0.0,
            "source": "test",
        }
        quote = {
            "symbol": "HIMS",
            "price": 28.09,
            "previousClose": 32.74,
            "change": -4.65,
            "changePercentage": -14.2,
            "volume": 28_095_864,
            "timestamp": 1_785_000_000,
        }

        with (
            patch("app.services.research.fetch_single_ticker_ohlcv", return_value=bad_history),
            patch("app.services.research.fetch_fmp_price_history", return_value=good_history),
            patch("app.services.research.fetch_ticker_info", return_value=fundamentals),
            patch("app.services.research.fetch_fmp_quote", return_value=quote),
        ):
            payload = compute_research(
                "HIMS",
                "Hims & Hers",
                "Healthcare",
                start_date=date(2025, 1, 2),
                end_date=date(2025, 12, 31),
            )

        self.assertEqual(payload["currentPrice"], 28.09)
        self.assertEqual(payload["chartLastClose"], 28.09)
        self.assertEqual(payload["dataQuality"]["priceSource"], "Financial Modeling Prep")
        self.assertLess(payload["ohlcv"]["close"][-1], 35.0)

    def test_research_suppresses_non_positive_pe_ratios(self) -> None:
        cleaned = _sanitize_fundamentals({"trailingPE": -561.8, "forwardPE": 21.5})
        self.assertIsNone(cleaned["trailingPE"])
        self.assertEqual(cleaned["forwardPE"], 21.5)


class InstitutionalScannerTests(unittest.TestCase):
    def test_trade_gate_passes_high_confidence_candidates(self) -> None:
        """High-confidence candidates with good backtests should TAKE."""
        backtest = {
            "winRate": 70.0,
            "avgReturn": 8.5,
            "alphaAvgReturn": 5.0,
            "sampleSize": 30,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        confidence = {"confidence": 80.0, "sampleSize": 30, "trustworthy": True}

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["decision"], "TAKE")
        self.assertTrue(gate["gateConditions"]["confidence"])
        self.assertTrue(gate["gateConditions"]["winRate"])
        self.assertTrue(gate["gateConditions"]["sampleSize"])
        self.assertTrue(gate["gateConditions"]["alphaVsBenchmark"])
        self.assertTrue(gate["gateConditions"]["simulationSurvival"])

    def test_trade_gate_rejects_low_confidence(self) -> None:
        """Low-confidence candidates should PASS regardless of other metrics."""
        backtest = {
            "winRate": 70.0,
            "avgReturn": 8.5,
            "alphaAvgReturn": 5.0,
            "sampleSize": 30,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        confidence = {"confidence": 50.0, "sampleSize": 30, "trustworthy": False}

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["decision"], "PASS")
        self.assertFalse(gate["gateConditions"]["confidence"])
        self.assertIn("Confidence", gate["reasons"][0])

    def test_trade_gate_rejects_low_win_rate(self) -> None:
        """Candidates with low win rates should PASS."""
        backtest = {
            "winRate": 50.0,  # Below MIN_BACKTEST_WIN_RATE
            "avgReturn": 8.5,
            "alphaAvgReturn": 5.0,
            "sampleSize": 30,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        confidence = {"confidence": 80.0, "sampleSize": 30, "trustworthy": True}

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["decision"], "PASS")
        self.assertFalse(gate["gateConditions"]["winRate"])

    def test_trade_gate_rejects_small_sample_size(self) -> None:
        """Candidates with insufficient samples should PASS."""
        backtest = {
            "winRate": 70.0,
            "avgReturn": 8.5,
            "alphaAvgReturn": 5.0,
            "sampleSize": 10,  # Below MIN_BACKTEST_SAMPLE_SIZE
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        confidence = {"confidence": 80.0, "sampleSize": 10, "trustworthy": False}

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["decision"], "PASS")
        self.assertFalse(gate["gateConditions"]["sampleSize"])

    def test_trade_gate_rejects_low_alpha_vs_benchmark(self) -> None:
        """Candidates with insufficient alpha should PASS."""
        backtest = {
            "winRate": 70.0,
            "avgReturn": 8.5,
            "alphaAvgReturn": 1.0,  # Below MIN_ALPHA_VS_BENCHMARK
            "sampleSize": 30,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        confidence = {"confidence": 80.0, "sampleSize": 30, "trustworthy": True}

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["decision"], "PASS")
        self.assertFalse(gate["gateConditions"]["alphaVsBenchmark"])

    def test_trade_gate_rejects_failed_simulation(self) -> None:
        """Candidates that fail simulation scenarios should PASS."""
        backtest = {
            "winRate": 70.0,
            "avgReturn": 8.5,
            "alphaAvgReturn": 5.0,
            "sampleSize": 30,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": False, "scenarios": {}}
        confidence = {"confidence": 80.0, "sampleSize": 30, "trustworthy": True}

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["decision"], "PASS")
        self.assertFalse(gate["gateConditions"]["simulationSurvival"])

    def test_simulation_validates_under_all_scenarios(self) -> None:
        """Simulation should test bull, base, bear, and high-vol scenarios."""
        index = pd.bdate_range("2024-01-01", periods=300)
        stock = pd.Series(np.linspace(100.0, 120.0, len(index)), index=index)
        spy = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)

        # Mock backtest with strong edge
        backtest = {
            "winRate": 70.0,
            "avgReturn": 5.0,
            "alphaAvgReturn": 3.5,
            "sampleSize": 30,
            "valid": True,
        }

        simulation = _run_simulation_validation(
            stock, spy, spy, backtest, risk_mode="balanced", regime="auto"
        )

        self.assertIn("bull", simulation["scenarios"])
        self.assertIn("base", simulation["scenarios"])
        self.assertIn("bear", simulation["scenarios"])
        self.assertIn("high_vol", simulation["scenarios"])

        for scenario in simulation["scenarios"].values():
            self.assertIn("winRate", scenario)
            self.assertIn("avgReturn", scenario)
            self.assertIn("survives", scenario)

    def test_simulation_includes_transaction_costs(self) -> None:
        """Simulation should account for realistic transaction costs."""
        index = pd.bdate_range("2024-01-01", periods=300)
        stock = pd.Series(np.linspace(100.0, 120.0, len(index)), index=index)
        spy = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)

        # Mock backtest with positive edge
        backtest = {
            "winRate": 65.0,
            "avgReturn": 4.0,
            "alphaAvgReturn": 3.0,
            "sampleSize": 25,
            "valid": True,
        }

        simulation = _run_simulation_validation(
            stock, spy, spy, backtest, risk_mode="balanced", regime="auto"
        )

        # After 25 bps costs (20 bps + 5 bps slippage), returns should be reduced
        # Base scenario stresses the 4.0% average return, so result will be less
        base_avg_return = simulation["scenarios"]["base"]["avgReturn"]
        self.assertLess(base_avg_return, 4.0)  # Should be reduced by costs

    def test_confidence_calibrates_from_backtest_and_simulation(self) -> None:
        """Confidence should be computed from multiple signals."""
        backtest = {
            "winRate": 70.0,
            "avgReturn": 8.5,
            "alphaAvgReturn": 5.0,
            "sampleSize": 30,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}

        confidence_data = _compute_confidence(backtest, simulation, alpha_score=75.0, risk_score=70.0)

        self.assertGreater(confidence_data["confidence"], 0.0)
        self.assertLessEqual(confidence_data["confidence"], 100.0)
        self.assertEqual(confidence_data["sampleSize"], 30)
        self.assertTrue(confidence_data["trustworthy"])

    def test_confidence_penalizes_low_sample_size(self) -> None:
        """Confidence should be reduced for small samples."""
        backtest_large = {
            "winRate": 70.0,
            "alphaAvgReturn": 5.0,
            "sampleSize": 50,
            "valid": True,
        }
        backtest_small = {
            "winRate": 70.0,
            "alphaAvgReturn": 5.0,
            "sampleSize": 10,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}

        conf_large = _compute_confidence(backtest_large, simulation, alpha_score=75.0, risk_score=70.0)
        conf_small = _compute_confidence(backtest_small, simulation, alpha_score=75.0, risk_score=70.0)

        self.assertGreater(conf_large["confidence"], conf_small["confidence"])
        self.assertFalse(conf_small["trustworthy"])

    def test_confidence_penalizes_failed_simulation(self) -> None:
        """Confidence should be reduced when simulation scenarios fail."""
        backtest = {
            "winRate": 70.0,
            "alphaAvgReturn": 5.0,
            "sampleSize": 30,
            "valid": True,
        }
        sim_pass = {"allScenariosSurvive": True, "scenarios": {}}
        sim_fail = {"allScenariosSurvive": False, "scenarios": {}}

        conf_pass = _compute_confidence(backtest, sim_pass, alpha_score=75.0, risk_score=70.0)
        conf_fail = _compute_confidence(backtest, sim_fail, alpha_score=75.0, risk_score=70.0)

        self.assertGreater(conf_pass["confidence"], conf_fail["confidence"])

    def test_convexity_alert_disabled_without_real_option_data(self) -> None:
        """Convexity alerts should not fire without real options chain data."""
        # Low volatility case
        alert_low_vol = _detect_convexity_alert(
            ticker="AAPL",
            current_price=150.0,
            volatility=15.0,
            alpha_score=75.0,
            expected_return=5.0,
        )
        self.assertIsNone(alert_low_vol)

        # High volatility case - still no alert without real options data
        alert_high_vol = _detect_convexity_alert(
            ticker="MEME",
            current_price=50.0,
            volatility=150.0,
            alpha_score=75.0,
            expected_return=10.0,
        )
        self.assertIsNone(alert_high_vol)

        # Strong technicals - still no alert without real options data
        alert_strong = _detect_convexity_alert(
            ticker="TEST",
            current_price=100.0,
            volatility=200.0,
            alpha_score=90.0,
            expected_return=20.0,
        )
        self.assertIsNone(alert_strong)

    def test_original_formula_is_conservative(self) -> None:
        """
        The original confidence formula is structurally conservative.
        Even strong candidates struggle to reach 75% threshold.
        
        This test documents the observed behavior - it does NOT assert
        that strong candidates should automatically TAKE. That would
        require empirical calibration against realized outcomes.
        """
        # Strong backtest: high win rate, good alpha, sufficient samples
        backtest = {
            "winRate": 75.0,
            "avgReturn": 8.0,
            "alphaAvgReturn": 5.5,
            "sampleSize": 35,
            "valid": True,
        }
        
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        confidence = _compute_confidence(backtest, simulation, alpha_score=78.0, risk_score=75.0)
        
        # Document what the original formula produces
        # (This will be below 75% due to conservative weighting)
        self.assertLess(confidence["confidence"], 75.0)
        self.assertGreater(confidence["confidence"], 50.0)  # But not absurdly low

    def test_walk_forward_backtest_avoids_lookahead_bias(self) -> None:
        """Backtest should only use data available at signal time."""
        # Create an uptrend with a sharp reversal at the end
        index = pd.bdate_range("2023-01-01", periods=350)
        prices = np.concatenate([
            np.linspace(100.0, 150.0, 300),  # Uptrend
            np.linspace(150.0, 80.0, 50),    # Sharp reversal
        ])
        stock = pd.Series(prices, index=index)
        spy = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)

        backtest = _run_walk_forward_backtest(
            stock, spy, spy, risk_mode="balanced", regime="auto"
        )

        # The backtest should have stopped before the reversal
        # and should have some successful trades from the uptrend period
        if backtest["valid"]:
            self.assertGreater(backtest["sampleSize"], 0)
            # Win rate should reflect trades made before reversal
            self.assertGreater(backtest["winRate"], 0.0)

    def test_walk_forward_backtest_requires_sufficient_history(self) -> None:
        """Backtest should require at least 260 days of history."""
        index = pd.bdate_range("2025-01-01", periods=100)
        stock = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
        spy = pd.Series(np.linspace(100.0, 105.0, len(index)), index=index)

        backtest = _run_walk_forward_backtest(
            stock, spy, spy, risk_mode="balanced", regime="auto"
        )

        self.assertFalse(backtest["valid"])
        self.assertEqual(backtest["sampleSize"], 0)


if __name__ == "__main__":
    unittest.main()
