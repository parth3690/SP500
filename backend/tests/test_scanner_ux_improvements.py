"""
Tests for institutional scanner UX improvements.

Tests cover:
- Gate transparency (deltas showing actual vs required)
- Watch-tier identification (exactly 1 failure)
- Universe integrity (ETF/fund/ADR exclusion)
- Data quality signals
- Confidence calibration
- Scan persistence
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.models import Constituent
from app.services.institutional_scanner import _apply_trade_gate, _compute_confidence
from app.services.scan_persistence import (
    cleanup_old_runs,
    list_scan_runs,
    load_scan_run,
    save_scan_run,
)
from app.services.sp500 import _is_valid_common_stock


class GateDeltasTests(unittest.TestCase):
    """Tests for gate transparency - showing actual vs required values."""

    def test_gate_deltas_show_actual_and_required_values(self) -> None:
        """Gate deltas should include actual, required, and delta for each condition."""
        backtest = {
            "winRate": 65.0,
            "avgReturn": 7.0,
            "alphaAvgReturn": 4.0,
            "sampleSize": 25,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        confidence = {"confidence": 72.0, "sampleSize": 25, "trustworthy": True}

        gate = _apply_trade_gate(confidence, backtest, simulation)

        # Should have gateDeltas
        self.assertIn("gateDeltas", gate)
        deltas = gate["gateDeltas"]

        # Check confidence delta
        self.assertEqual(deltas["confidence"]["actual"], 72.0)
        self.assertEqual(deltas["confidence"]["required"], 75.0)
        self.assertEqual(deltas["confidence"]["delta"], -3.0)
        self.assertFalse(deltas["confidence"]["pass"])

        # Check winRate delta
        self.assertEqual(deltas["winRate"]["actual"], 65.0)
        self.assertEqual(deltas["winRate"]["required"], 62.0)
        self.assertEqual(deltas["winRate"]["delta"], 3.0)
        self.assertTrue(deltas["winRate"]["pass"])

        # Check sampleSize delta
        self.assertEqual(deltas["sampleSize"]["actual"], 25)
        self.assertEqual(deltas["sampleSize"]["required"], 20)
        self.assertEqual(deltas["sampleSize"]["delta"], 5)
        self.assertTrue(deltas["sampleSize"]["pass"])

        # Check alphaVsBenchmark delta
        self.assertEqual(deltas["alphaVsBenchmark"]["actual"], 4.0)
        self.assertEqual(deltas["alphaVsBenchmark"]["required"], 3.0)
        self.assertEqual(deltas["alphaVsBenchmark"]["delta"], 1.0)
        self.assertTrue(deltas["alphaVsBenchmark"]["pass"])

    def test_gate_deltas_include_failed_scenarios(self) -> None:
        """Gate deltas should list which simulation scenarios failed."""
        backtest = {
            "winRate": 70.0,
            "alphaAvgReturn": 5.0,
            "sampleSize": 30,
            "valid": True,
        }
        simulation = {
            "allScenariosSurvive": False,
            "scenarios": {
                "bull": {"survives": True},
                "base": {"survives": True},
                "bear": {"survives": False},
                "high_vol": {"survives": False},
            },
        }
        confidence = {"confidence": 78.0, "sampleSize": 30, "trustworthy": True}

        gate = _apply_trade_gate(confidence, backtest, simulation)

        sim_delta = gate["gateDeltas"]["simulationSurvival"]
        self.assertFalse(sim_delta["pass"])
        self.assertEqual(set(sim_delta["failedScenarios"]), {"bear", "high_vol"})

    def test_gate_reasons_include_gap_sizes(self) -> None:
        """PASS reasons should show the gap size for transparency."""
        backtest = {
            "winRate": 58.0,  # 4% below threshold
            "alphaAvgReturn": 2.5,  # 0.5% below threshold
            "sampleSize": 15,  # 5 below threshold
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        confidence = {"confidence": 68.0, "sampleSize": 15, "trustworthy": False}  # 7% below

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["decision"], "PASS")
        reasons = " ".join(gate["reasons"])

        # Check that gaps are shown
        self.assertIn("gap: 7.0%", reasons)  # Confidence gap
        self.assertIn("gap: 4.0%", reasons)  # Win rate gap
        self.assertIn("gap: 5", reasons)  # Sample size gap
        self.assertIn("gap: 0.50%", reasons)  # Alpha gap


class WatchTierTests(unittest.TestCase):
    """Tests for watch-tier identification (exactly 1 gate failure)."""

    def test_watch_tier_true_for_single_failure(self) -> None:
        """Candidates failing on exactly one dimension should be watch-tier."""
        # Fail only on confidence
        backtest = {
            "winRate": 70.0,  # Pass
            "alphaAvgReturn": 5.0,  # Pass
            "sampleSize": 30,  # Pass
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}  # Pass
        confidence = {"confidence": 72.0, "sampleSize": 30, "trustworthy": True}  # Fail

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["numFailures"], 1)
        self.assertTrue(gate["watchTier"])
        self.assertEqual(gate["decision"], "PASS")

    def test_watch_tier_false_for_multiple_failures(self) -> None:
        """Candidates failing on multiple dimensions should not be watch-tier."""
        backtest = {
            "winRate": 55.0,  # Fail
            "alphaAvgReturn": 2.0,  # Fail
            "sampleSize": 30,  # Pass
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}  # Pass
        confidence = {"confidence": 68.0, "sampleSize": 30, "trustworthy": True}  # Fail

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["numFailures"], 3)
        self.assertFalse(gate["watchTier"])

    def test_watch_tier_false_for_take(self) -> None:
        """TAKE candidates should not be watch-tier."""
        backtest = {
            "winRate": 75.0,
            "alphaAvgReturn": 5.5,
            "sampleSize": 35,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        confidence = {"confidence": 82.0, "sampleSize": 35, "trustworthy": True}

        gate = _apply_trade_gate(confidence, backtest, simulation)

        self.assertEqual(gate["decision"], "TAKE")
        self.assertEqual(gate["numFailures"], 0)
        self.assertFalse(gate["watchTier"])


class UniverseIntegrityTests(unittest.TestCase):
    """Tests for universe exclusion filters (ETFs, funds, ADRs)."""

    def test_valid_common_stock_passes(self) -> None:
        """Normal common stocks should pass the filter."""
        valid, reason = _is_valid_common_stock("AAPL", "Apple Inc.")
        self.assertTrue(valid)
        self.assertEqual(reason, "")

        valid, reason = _is_valid_common_stock("MSFT", "Microsoft Corporation")
        self.assertTrue(valid)

    def test_etf_is_excluded(self) -> None:
        """ETFs should be excluded."""
        valid, reason = _is_valid_common_stock("SPY", "SPDR S&P 500 ETF Trust")
        self.assertFalse(valid)
        self.assertIn("etf", reason.lower())

        valid, reason = _is_valid_common_stock("QQQ", "Invesco QQQ Trust")
        self.assertFalse(valid)

    def test_closed_end_fund_is_excluded(self) -> None:
        """Closed-end funds should be excluded."""
        valid, reason = _is_valid_common_stock(
            "HQL", "Tekla Life Sciences Investors Closed End Fund"
        )
        self.assertFalse(valid)
        self.assertIn("closed", reason.lower())

    def test_adr_is_excluded(self) -> None:
        """ADRs should be excluded."""
        valid, reason = _is_valid_common_stock("TSM", "Taiwan Semiconductor ADR")
        self.assertFalse(valid)
        self.assertIn("adr", reason.lower())

    def test_preferred_stock_pattern_is_excluded(self) -> None:
        """Preferred stock ticker patterns should be excluded."""
        valid, reason = _is_valid_common_stock("BAC-P", "Bank of America Preferred")
        self.assertFalse(valid)
        self.assertIn("preferred", reason.lower())

        valid, reason = _is_valid_common_stock("JPM.P", "JPMorgan Preferred")
        self.assertFalse(valid)

    def test_warrant_pattern_is_excluded(self) -> None:
        """Warrant ticker patterns should be excluded."""
        valid, reason = _is_valid_common_stock("SPCE-W", "Virgin Galactic Warrant")
        self.assertFalse(valid)
        self.assertIn("warrant", reason.lower())

    def test_trust_is_excluded(self) -> None:
        """Trusts should be excluded."""
        valid, reason = _is_valid_common_stock("VNQ", "Vanguard Real Estate Trust")
        self.assertFalse(valid)
        self.assertIn("trust", reason.lower())


class ConfidenceCalibrationTests(unittest.TestCase):
    """Tests for confidence calibration improvements."""

    def test_strong_candidate_can_reach_take_threshold(self) -> None:
        """A strong candidate should be able to reach 75% confidence."""
        # Strong backtest
        backtest = {
            "winRate": 75.0,
            "avgReturn": 8.0,
            "alphaAvgReturn": 5.5,
            "sampleSize": 35,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}

        # High alpha and risk scores
        confidence = _compute_confidence(backtest, simulation, alpha_score=78.0, risk_score=75.0)

        # Should reach or exceed the 75% threshold
        self.assertGreaterEqual(confidence["confidence"], 75.0)
        self.assertTrue(confidence["trustworthy"])

    def test_calibration_details_included(self) -> None:
        """Confidence should include calibration breakdown."""
        backtest = {
            "winRate": 70.0,
            "alphaAvgReturn": 4.0,
            "sampleSize": 25,
            "valid": True,
        }
        simulation = {"allScenariosSurvive": True, "scenarios": {}}

        confidence = _compute_confidence(backtest, simulation, alpha_score=72.0, risk_score=70.0)

        self.assertIn("calibrationDetails", confidence)
        details = confidence["calibrationDetails"]
        self.assertIn("baseConfidence", details)
        self.assertIn("samplePenalty", details)
        self.assertIn("simulationMultiplier", details)
        self.assertIn("riskAdjustment", details)

    def test_exceptional_performance_bonus(self) -> None:
        """Exceptional candidates should get a confidence bonus."""
        # Exceptional backtest: win rate >= 70% and alpha >= 4%
        backtest_exceptional = {
            "winRate": 73.0,
            "alphaAvgReturn": 5.0,
            "sampleSize": 30,
            "valid": True,
        }
        
        # Good but not exceptional
        backtest_good = {
            "winRate": 68.0,
            "alphaAvgReturn": 3.5,
            "sampleSize": 30,
            "valid": True,
        }
        
        simulation = {"allScenariosSurvive": True, "scenarios": {}}
        
        conf_exceptional = _compute_confidence(
            backtest_exceptional, simulation, alpha_score=75.0, risk_score=70.0
        )
        conf_good = _compute_confidence(
            backtest_good, simulation, alpha_score=75.0, risk_score=70.0
        )
        
        # Exceptional should have higher confidence
        self.assertGreater(
            conf_exceptional["confidence"],
            conf_good["confidence"],
        )


class ScanPersistenceTests(unittest.TestCase):
    """Tests for scan run persistence."""

    def setUp(self) -> None:
        """Set up temporary persistence directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_dir_patch = patch(
            "app.services.scan_persistence.PERSISTENCE_DIR", Path(self.temp_dir)
        )
        self.persistence_dir_patch.start()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.persistence_dir_patch.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_scan_run(self) -> None:
        """Should save and load scan runs correctly."""
        inputs = {
            "universe": "sp500",
            "limit": 20,
            "minScore": 65.0,
            "riskMode": "balanced",
        }
        results = {
            "asOf": "2026-09-03T12:00:00Z",
            "candidates": [{"ticker": "AAPL", "alphaScore": 75.0}],
            "meta": {"status": "complete"},
        }

        run_id = save_scan_run("institutional_sp500", inputs, results)

        # Should be able to load it back
        loaded = load_scan_run(run_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["scanType"], "institutional_sp500")
        self.assertEqual(loaded["inputs"], inputs)
        self.assertEqual(loaded["results"]["candidates"], results["candidates"])

    def test_list_scan_runs(self) -> None:
        """Should list recent scan runs with metadata."""
        # Save a few runs
        for i in range(3):
            inputs = {"universe": "sp500", "limit": 20}
            results = {
                "asOf": "2026-09-03T12:00:00Z",
                "candidates": [],
                "meta": {
                    "status": "complete",
                    "summary": {
                        "takeCount": i,
                        "watchTierCount": 1,
                        "totalScanned": 20,
                    },
                },
            }
            save_scan_run("institutional_sp500", inputs, results)

        runs = list_scan_runs(limit=10)
        self.assertEqual(len(runs), 3)

        # Should include summary data
        first_run = runs[0]  # Most recent
        self.assertEqual(first_run["scanType"], "institutional_sp500")
        self.assertIn("timestamp", first_run)
        self.assertIn("summary", first_run)
        self.assertEqual(first_run["summary"]["takeCount"], 2)

    def test_list_scan_runs_filters_by_type(self) -> None:
        """Should filter scan runs by type."""
        save_scan_run("institutional_sp500", {}, {"candidates": [], "meta": {}})
        save_scan_run("institutional_nyse_smid", {}, {"candidates": [], "meta": {}})

        sp500_runs = list_scan_runs(scan_type="institutional_sp500")
        self.assertEqual(len(sp500_runs), 1)
        self.assertEqual(sp500_runs[0]["scanType"], "institutional_sp500")

    def test_cleanup_old_runs(self) -> None:
        """Should clean up old scan runs."""
        import time

        # Save a run
        run_id = save_scan_run("institutional_sp500", {}, {"candidates": [], "meta": {}})
        run_path = Path(self.temp_dir) / f"{run_id}.json"

        # Age the file by modifying mtime
        old_time = time.time() - (31 * 86400)  # 31 days ago
        run_path.touch()
        import os
        os.utime(run_path, (old_time, old_time))

        # Cleanup should delete it
        deleted = cleanup_old_runs(days_to_keep=30)
        self.assertEqual(deleted, 1)
        self.assertFalse(run_path.exists())


if __name__ == "__main__":
    unittest.main()
