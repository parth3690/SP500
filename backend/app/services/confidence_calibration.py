"""
Empirical confidence calibration for institutional scanner.

Fits isotonic regression mapping from raw confidence scores to predicted win probability,
using walk-forward backtest outcomes. No lookahead: fit on earlier folds, evaluate on later ones.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


def _compute_raw_confidence_features(
    win_rate: float,
    alpha_vs_bench: float,
    alpha_score: float,
    risk_score: float,
    sample_size: int,
    sim_survives: bool,
) -> dict[str, Any]:
    """
    Compute raw confidence features using the ORIGINAL formula.
    Returns both the final score and intermediate components.
    """
    # Original base confidence
    base = win_rate * 0.50 + min(max(alpha_vs_bench * 5.0, 0), 50) + alpha_score * 0.30
    
    # Sample size penalty
    sample_penalty = 0.0
    if sample_size < 20:
        sample_penalty = (20 - sample_size) / 20 * 30
        base -= sample_penalty
    
    # Simulation multiplier
    sim_mult = 0.70 if not sim_survives else 1.0
    base *= sim_mult
    
    # Risk adjustment
    risk_adj = risk_score / 100.0
    base *= risk_adj
    
    raw_confidence = min(max(base, 0.0), 100.0)
    
    return {
        "rawConfidence": raw_confidence,
        "winRate": win_rate,
        "alphaVsBench": alpha_vs_bench,
        "alphaScore": alpha_score,
        "riskScore": risk_score,
        "sampleSize": sample_size,
        "simSurvives": sim_survives,
    }


def fit_and_evaluate_calibration(
    samples: list[dict[str, Any]],
    *,
    n_folds: int = 5,
) -> dict[str, Any]:
    """
    Fit isotonic regression calibration on walk-forward samples.
    
    Args:
        samples: List of dicts with keys:
            - rawConfidence: float (0-100)
            - realized: bool (whether the trade won)
            - timestamp: int (for time-based splitting)
        n_folds: Number of walk-forward folds
    
    Returns:
        dict with:
            - calibrator: fitted IsotonicRegression or None
            - oldMetrics: {brierScore, logLoss, calibrationTable}
            - newMetrics: {brierScore, logLoss, calibrationTable}
            - improved: bool
    """
    if not samples or len(samples) < 50:
        return {
            "calibrator": None,
            "oldMetrics": None,
            "newMetrics": None,
            "improved": False,
            "reason": f"Insufficient samples for calibration (need >=50, got {len(samples)})",
        }
    
    # Sort by timestamp for time-based splitting
    samples = sorted(samples, key=lambda x: x.get("timestamp", 0))
    
    # Collect raw predictions and outcomes
    raw_probs = np.array([s["rawConfidence"] / 100.0 for s in samples])
    outcomes = np.array([1.0 if s["realized"] else 0.0 for s in samples])
    
    # Old metrics (raw confidence, no calibration)
    old_brier = brier_score_loss(outcomes, raw_probs)
    old_logloss = log_loss(outcomes, raw_probs)
    old_table = _calibration_table(raw_probs, outcomes, n_bins=10)
    
    # Walk-forward calibration: fit on first (n_folds-1)/n_folds, evaluate on last 1/n_folds
    fold_size = len(samples) // n_folds
    
    all_calibrated_probs = []
    all_outcomes = []
    
    for fold_idx in range(1, n_folds):
        # Train on folds 0 to fold_idx-1
        train_end = fold_idx * fold_size
        train_raw = raw_probs[:train_end]
        train_outcomes = outcomes[:train_end]
        
        # Test on fold fold_idx
        test_start = train_end
        test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else len(samples)
        test_raw = raw_probs[test_start:test_end]
        test_outcomes = outcomes[test_start:test_end]
        
        if len(train_raw) < 20 or len(test_raw) < 5:
            continue
        
        # Fit isotonic regression on training fold
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(train_raw, train_outcomes)
        
        # Predict on test fold
        calibrated = calibrator.predict(test_raw)
        
        all_calibrated_probs.extend(calibrated)
        all_outcomes.extend(test_outcomes)
    
    if not all_calibrated_probs:
        return {
            "calibrator": None,
            "oldMetrics": {
                "brierScore": round(old_brier, 4),
                "logLoss": round(old_logloss, 4),
                "calibrationTable": old_table,
            },
            "newMetrics": None,
            "improved": False,
            "reason": "Not enough samples per fold for walk-forward evaluation",
        }
    
    # New metrics (calibrated)
    all_calibrated_probs = np.array(all_calibrated_probs)
    all_outcomes = np.array(all_outcomes)
    
    new_brier = brier_score_loss(all_outcomes, all_calibrated_probs)
    new_logloss = log_loss(all_outcomes, all_calibrated_probs)
    new_table = _calibration_table(all_calibrated_probs, all_outcomes, n_bins=10)
    
    # Fit final calibrator on ALL data (for production use if improved)
    final_calibrator = None
    if new_brier < old_brier:
        final_calibrator = IsotonicRegression(out_of_bounds="clip")
        final_calibrator.fit(raw_probs, outcomes)
    
    return {
        "calibrator": final_calibrator,
        "oldMetrics": {
            "brierScore": round(old_brier, 4),
            "logLoss": round(old_logloss, 4),
            "calibrationTable": old_table,
        },
        "newMetrics": {
            "brierScore": round(new_brier, 4),
            "logLoss": round(new_logloss, 4),
            "calibrationTable": new_table,
        },
        "improved": new_brier < old_brier,
        "brierImprovement": round(old_brier - new_brier, 4),
    }


def _calibration_table(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """
    Generate calibration table: predicted probability bucket vs realized win rate.
    
    Returns list of dicts with:
        - bin: str (e.g., "0-10%")
        - predictedMean: float
        - realizedRate: float
        - count: int
    """
    bins = np.linspace(0, 1, n_bins + 1)
    table = []
    
    for i in range(n_bins):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        
        mask = (predicted_probs >= bin_start) & (predicted_probs < bin_end)
        if i == n_bins - 1:  # Include upper bound in last bin
            mask = (predicted_probs >= bin_start) & (predicted_probs <= bin_end)
        
        bin_preds = predicted_probs[mask]
        bin_outcomes = outcomes[mask]
        
        if len(bin_preds) == 0:
            continue
        
        table.append({
            "bin": f"{int(bin_start * 100)}-{int(bin_end * 100)}%",
            "predictedMean": round(float(bin_preds.mean()) * 100, 1),
            "realizedRate": round(float(bin_outcomes.mean()) * 100, 1),
            "count": len(bin_preds),
        })
    
    return table


def apply_calibration(
    raw_confidence: float,
    calibrator: Optional[Any],
) -> float:
    """
    Apply isotonic calibration to raw confidence score.
    
    Args:
        raw_confidence: Raw confidence (0-100)
        calibrator: Fitted IsotonicRegression or None
    
    Returns:
        Calibrated confidence (0-100)
    """
    if calibrator is None:
        return raw_confidence
    
    raw_prob = raw_confidence / 100.0
    calibrated_prob = calibrator.predict([raw_prob])[0]
    return float(calibrated_prob * 100.0)


def generate_calibration_report(
    constituents: list[Any],
    close_prices: pd.DataFrame,
    *,
    min_samples: int = 100,
) -> dict[str, Any]:
    """
    Generate calibration report by running walk-forward backtests on historical data.
    
    This is used ONCE to fit the calibrator, not on every scan.
    """
    from .institutional_scanner import _run_walk_forward_backtest, _run_simulation_validation
    from .alpha import compute_alpha_candidates, SECTOR_ETFS
    
    # Run alpha candidates to get initial set
    alpha_result = compute_alpha_candidates(
        constituents,
        close_prices,
        limit=100,
        min_score=60.0,
        enrich_top=0,
        include_lowest=0,
    )
    
    if not alpha_result["candidates"]:
        return {
            "status": "no_candidates",
            "samples": [],
            "calibration": None,
        }
    
    spy = close_prices["SPY"].dropna() if "SPY" in close_prices.columns else pd.Series()
    
    samples = []
    
    for candidate in alpha_result["candidates"][:50]:  # Limit to top 50 for performance
        ticker = candidate["ticker"]
        yahoo_ticker = None
        
        for c in constituents:
            if c.ticker == ticker:
                yahoo_ticker = c.yahooTicker
                break
        
        if not yahoo_ticker or yahoo_ticker not in close_prices.columns:
            continue
        
        stock = close_prices[yahoo_ticker].dropna()
        sector_etf = None
        for c in constituents:
            if c.ticker == ticker:
                sector_etf = SECTOR_ETFS.get(c.sector, "SPY")
                break
        
        sector_series = (
            close_prices[sector_etf].dropna()
            if sector_etf and sector_etf in close_prices.columns
            else spy
        )
        
        # Run backtest
        backtest = _run_walk_forward_backtest(
            stock, spy, sector_series, risk_mode="balanced", regime="auto"
        )
        
        if not backtest["valid"] or backtest["sampleSize"] == 0:
            continue
        
        # Run simulation
        simulation = _run_simulation_validation(
            stock, spy, sector_series, backtest, risk_mode="balanced", regime="auto"
        )
        
        # Compute raw confidence
        features = _compute_raw_confidence_features(
            win_rate=backtest["winRate"],
            alpha_vs_bench=backtest["alphaAvgReturn"],
            alpha_score=candidate["alphaScore"],
            risk_score=candidate["riskScore"],
            sample_size=backtest["sampleSize"],
            sim_survives=simulation["allScenariosSurvive"],
        )
        
        # Use backtest win rate as realized outcome
        # (In production, you'd use actual forward returns, but this is a proxy)
        samples.append({
            "ticker": ticker,
            "rawConfidence": features["rawConfidence"],
            "realized": backtest["winRate"] >= 62.0,  # Proxy: did it beat the gate threshold?
            "timestamp": len(samples),  # Time ordering
        })
    
    if len(samples) < min_samples:
        return {
            "status": "insufficient_samples",
            "samples": samples,
            "calibration": None,
            "reason": f"Need {min_samples} samples, got {len(samples)}",
        }
    
    # Fit calibration
    calibration = fit_and_evaluate_calibration(samples)
    
    return {
        "status": "complete",
        "samples": samples,
        "calibration": calibration,
    }
