"""
Delta improvement loop for the institutional alpha scanner.

This module implements a state-of-the-art iterative improvement system that:
1. Proposes ONE concrete change per iteration
2. Evaluates the change against the full universe (S&P 500 + NYSE SMID + QQQ)
3. KEEPs the change if it improves outcomes, REVERTs otherwise
4. Tracks a detailed changelog of all kept changes
5. Persists state for resumable long runs

The goal is to surface high win-probability names that clear the desk gate,
not to weaken the gate to force listings.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ..models import Constituent
from .alpha import SECTOR_ETFS
from .institutional_scanner import (
    MIN_ALPHA_VS_BENCHMARK,
    MIN_BACKTEST_SAMPLE_SIZE,
    MIN_BACKTEST_WIN_RATE,
    MIN_CONFIDENCE_FOR_TAKE,
    scan_institutional_grade,
)
from .prices import fetch_close_prices
from .sp500 import (
    get_nyse_smid_constituents_cached,
    get_sp500_constituents_cached,
)


def load_qqq_constituents() -> list[Constituent]:
    """
    Load QQQ/Nasdaq-100 constituents.
    Returns a curated list of major Nasdaq-100 stocks.
    """
    # Nasdaq-100 major constituents (top 50 by weight)
    nasdaq100_data = [
        ("AAPL", "Apple Inc", "Technology"),
        ("MSFT", "Microsoft Corp", "Technology"),
        ("AMZN", "Amazon.com Inc", "Consumer Discretionary"),
        ("NVDA", "NVIDIA Corp", "Technology"),
        ("GOOGL", "Alphabet Inc Class A", "Communication Services"),
        ("GOOG", "Alphabet Inc Class C", "Communication Services"),
        ("META", "Meta Platforms Inc", "Communication Services"),
        ("TSLA", "Tesla Inc", "Consumer Discretionary"),
        ("AVGO", "Broadcom Inc", "Technology"),
        ("COST", "Costco Wholesale Corp", "Consumer Staples"),
        ("NFLX", "Netflix Inc", "Communication Services"),
        ("AMD", "Advanced Micro Devices", "Technology"),
        ("PEP", "PepsiCo Inc", "Consumer Staples"),
        ("ADBE", "Adobe Inc", "Technology"),
        ("CSCO", "Cisco Systems Inc", "Technology"),
        ("TMUS", "T-Mobile US Inc", "Communication Services"),
        ("CMCSA", "Comcast Corp", "Communication Services"),
        ("INTC", "Intel Corp", "Technology"),
        ("TXN", "Texas Instruments", "Technology"),
        ("QCOM", "QUALCOMM Inc", "Technology"),
        ("AMGN", "Amgen Inc", "Health Care"),
        ("INTU", "Intuit Inc", "Technology"),
        ("HON", "Honeywell International", "Industrials"),
        ("AMAT", "Applied Materials", "Technology"),
        ("SBUX", "Starbucks Corp", "Consumer Discretionary"),
        ("ISRG", "Intuitive Surgical", "Health Care"),
        ("BKNG", "Booking Holdings Inc", "Consumer Discretionary"),
        ("GILD", "Gilead Sciences", "Health Care"),
        ("ADP", "Automatic Data Processing", "Industrials"),
        ("VRTX", "Vertex Pharmaceuticals", "Health Care"),
        ("ADI", "Analog Devices", "Technology"),
        ("REGN", "Regeneron Pharmaceuticals", "Health Care"),
        ("PANW", "Palo Alto Networks", "Technology"),
        ("MU", "Micron Technology", "Technology"),
        ("LRCX", "Lam Research Corp", "Technology"),
        ("MDLZ", "Mondelez International", "Consumer Staples"),
        ("PYPL", "PayPal Holdings", "Financials"),
        ("KLAC", "KLA Corp", "Technology"),
        ("SNPS", "Synopsys Inc", "Technology"),
        ("CDNS", "Cadence Design Systems", "Technology"),
        ("MRVL", "Marvell Technology", "Technology"),
        ("ASML", "ASML Holding NV", "Technology"),
        ("CTAS", "Cintas Corp", "Industrials"),
        ("ORLY", "O'Reilly Automotive", "Consumer Discretionary"),
        ("MNST", "Monster Beverage Corp", "Consumer Staples"),
        ("ABNB", "Airbnb Inc", "Consumer Discretionary"),
        ("FTNT", "Fortinet Inc", "Technology"),
        ("MELI", "MercadoLibre Inc", "Consumer Discretionary"),
        ("WDAY", "Workday Inc", "Technology"),
        ("NXPI", "NXP Semiconductors", "Technology"),
    ]
    
    constituents = []
    for ticker, name, sector in nasdaq100_data:
        constituents.append(Constituent(
            ticker=ticker,
            yahooTicker=ticker,
            companyName=name,
            sector=sector,
            subIndustry=None,
        ))
    
    return constituents


def load_full_universe(*, refresh: bool = False) -> dict[str, list[Constituent]]:
    """
    Load the complete tradable universe for alpha scanning.
    
    Returns dict with:
    - sp500: S&P 500 constituents
    - nyse_smid: NYSE small-mid cap ($100M-$2B)
    - nasdaq100: QQQ/Nasdaq-100 constituents
    - all: Combined unique list (de-duped)
    """
    sp500 = get_sp500_constituents_cached(refresh=refresh)
    
    # Try to load NYSE SMID (may require FMP key)
    try:
        nyse_smid = get_nyse_smid_constituents_cached(refresh=refresh)
    except Exception:
        nyse_smid = []
    
    # Load Nasdaq-100
    nasdaq100 = load_qqq_constituents()
    
    # Combine and dedupe by ticker
    seen_tickers = set()
    all_constituents = []
    
    for c in sp500 + nyse_smid + nasdaq100:
        if c.ticker not in seen_tickers:
            seen_tickers.add(c.ticker)
            all_constituents.append(c)
    
    return {
        "sp500": sp500,
        "nyse_smid": nyse_smid,
        "nasdaq100": nasdaq100,
        "all": all_constituents,
    }


class ScannerState:
    """
    Current state of the scanner configuration.
    Tracks all parameters and modifications made through the delta loop.
    """
    
    def __init__(self):
        # Baseline parameters from existing scanner
        self.signal_threshold = 68.0
        self.forward_horizon = 20
        self.risk_mode = "balanced"
        self.min_score = 65.0
        self.step_size = 5
        
        # Additional state
        self.modifications = []  # List of applied modifications
        
    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_threshold": self.signal_threshold,
            "forward_horizon": self.forward_horizon,
            "risk_mode": self.risk_mode,
            "min_score": self.min_score,
            "step_size": self.step_size,
            "modifications": self.modifications,
        }
    
    def from_dict(self, d: dict[str, Any]):
        self.signal_threshold = d["signal_threshold"]
        self.forward_horizon = d["forward_horizon"]
        self.risk_mode = d["risk_mode"]
        self.min_score = d["min_score"]
        self.step_size = d["step_size"]
        self.modifications = d.get("modifications", [])
    
    def copy(self) -> "ScannerState":
        """Create a copy of current state."""
        new_state = ScannerState()
        new_state.from_dict(self.to_dict())
        return new_state


class ChangeProposal:
    """Represents a proposed change to the scanner."""
    
    def __init__(self, change_type: str, description: str, apply_fn: Callable, revert_fn: Callable):
        self.change_type = change_type
        self.description = description
        self.apply_fn = apply_fn
        self.revert_fn = revert_fn
    
    def apply(self, state: ScannerState) -> ScannerState:
        """Apply this change to the given state."""
        new_state = state.copy()
        self.apply_fn(new_state)
        new_state.modifications.append(self.description)
        return new_state
    
    def revert(self, state: ScannerState) -> ScannerState:
        """Revert this change from the given state."""
        new_state = state.copy()
        self.revert_fn(new_state)
        if new_state.modifications and self.description in new_state.modifications:
            new_state.modifications.remove(self.description)
        return new_state


def propose_change(iteration: int, current_state: ScannerState, past_results: list[dict]) -> ChangeProposal:
    """
    Propose ONE concrete change for this iteration.
    
    Changes can be:
    - Parameter adjustments informed by past results
    - Feature tweaks
    - Threshold calibrations
    - Walk-forward refinements
    """
    # Analyze past results to inform the proposal
    if past_results:
        recent_best_confidence = max(
            r.get("new_metrics", {}).get("best_candidate_score", 0)
            for r in past_results[-10:]  # Look at last 10 iterations
        )
    else:
        recent_best_confidence = 0
    
    # Strategy: try different types of changes based on iteration number and past results
    change_types = [
        "lower_signal_threshold",
        "raise_signal_threshold",
        "change_horizon",
        "adjust_min_score",
        "change_risk_mode",
        "adjust_step_size",
    ]
    
    # If we're close to threshold (70-75%), try subtle refinements
    if 70 <= recent_best_confidence < 75:
        change_type = random.choice(["lower_signal_threshold", "adjust_min_score", "change_horizon"])
    else:
        change_type = random.choice(change_types)
    
    if change_type == "lower_signal_threshold":
        # Lower signal threshold to allow more signals in backtests
        delta = random.uniform(0.5, 2.0)
        new_val = max(60.0, current_state.signal_threshold - delta)
        
        def apply(s: ScannerState):
            s.signal_threshold = new_val
        
        def revert(s: ScannerState):
            s.signal_threshold = current_state.signal_threshold
        
        return ChangeProposal(
            "parameter",
            f"Lower signal_threshold from {current_state.signal_threshold:.1f} to {new_val:.1f}",
            apply,
            revert
        )
    
    elif change_type == "raise_signal_threshold":
        # Raise signal threshold for higher quality signals
        delta = random.uniform(0.5, 2.0)
        new_val = min(75.0, current_state.signal_threshold + delta)
        
        def apply(s: ScannerState):
            s.signal_threshold = new_val
        
        def revert(s: ScannerState):
            s.signal_threshold = current_state.signal_threshold
        
        return ChangeProposal(
            "parameter",
            f"Raise signal_threshold from {current_state.signal_threshold:.1f} to {new_val:.1f}",
            apply,
            revert
        )
    
    elif change_type == "change_horizon":
        # Change forward horizon
        horizons = [15, 20, 25]
        current_idx = horizons.index(current_state.forward_horizon) if current_state.forward_horizon in horizons else 1
        new_idx = (current_idx + random.choice([-1, 1])) % len(horizons)
        new_val = horizons[new_idx]
        
        def apply(s: ScannerState):
            s.forward_horizon = new_val
        
        def revert(s: ScannerState):
            s.forward_horizon = current_state.forward_horizon
        
        return ChangeProposal(
            "parameter",
            f"Change forward_horizon from {current_state.forward_horizon} to {new_val} days",
            apply,
            revert
        )
    
    elif change_type == "adjust_min_score":
        # Adjust minimum score threshold
        delta = random.uniform(1.0, 3.0) * random.choice([-1, 1])
        new_val = max(55.0, min(70.0, current_state.min_score + delta))
        
        def apply(s: ScannerState):
            s.min_score = new_val
        
        def revert(s: ScannerState):
            s.min_score = current_state.min_score
        
        return ChangeProposal(
            "parameter",
            f"Adjust min_score from {current_state.min_score:.1f} to {new_val:.1f}",
            apply,
            revert
        )
    
    elif change_type == "change_risk_mode":
        # Change risk mode
        modes = ["aggressive", "balanced", "defensive"]
        current_idx = modes.index(current_state.risk_mode) if current_state.risk_mode in modes else 1
        new_idx = (current_idx + random.choice([-1, 1])) % len(modes)
        new_val = modes[new_idx]
        
        def apply(s: ScannerState):
            s.risk_mode = new_val
        
        def revert(s: ScannerState):
            s.risk_mode = current_state.risk_mode
        
        return ChangeProposal(
            "parameter",
            f"Change risk_mode from {current_state.risk_mode} to {new_val}",
            apply,
            revert
        )
    
    elif change_type == "adjust_step_size":
        # Adjust walk-forward step size
        step_sizes = [3, 5, 7]
        current_idx = step_sizes.index(current_state.step_size) if current_state.step_size in step_sizes else 1
        new_idx = (current_idx + random.choice([-1, 1])) % len(step_sizes)
        new_val = step_sizes[new_idx]
        
        def apply(s: ScannerState):
            s.step_size = new_val
        
        def revert(s: ScannerState):
            s.step_size = current_state.step_size
        
        return ChangeProposal(
            "parameter",
            f"Adjust step_size from {current_state.step_size} to {new_val} days",
            apply,
            revert
        )
    
    # Default fallback
    return ChangeProposal(
        "noop",
        "No change (baseline)",
        lambda s: None,
        lambda s: None
    )


def evaluate_state(
    state: ScannerState,
    constituents: list[Constituent],
    close_prices: pd.DataFrame,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Evaluate scanner with the given state on the full universe.
    Returns metrics including TAKE count, quality measures, and candidate details.
    """
    # Run scanner with current state
    result = scan_institutional_grade(
        constituents,
        close_prices,
        limit=limit,
        min_score=state.min_score,
        risk_mode=state.risk_mode,
        regime_override="auto",
        signal_threshold=state.signal_threshold,
        forward_horizon=state.forward_horizon,
        step_size=state.step_size,
    )
    
    candidates = result.get("candidates", [])
    
    if not candidates:
        return {
            "take_count": 0,
            "pass_count": 0,
            "take_tickers": [],
            "avg_take_confidence": 0.0,
            "avg_take_win_rate": 0.0,
            "avg_take_alpha": 0.0,
            "avg_take_sample_size": 0,
            "best_candidate_score": 0.0,
            "quality_score": 0.0,
        }
    
    takes = [c for c in candidates if c.get("tradeGate", {}).get("decision") == "TAKE"]
    passes = [c for c in candidates if c.get("tradeGate", {}).get("decision") == "PASS"]
    
    take_count = len(takes)
    pass_count = len(passes)
    take_tickers = [c["ticker"] for c in takes]
    
    if take_count > 0:
        avg_take_confidence = np.mean([c["confidence"]["confidence"] for c in takes])
        avg_take_win_rate = np.mean([c["backtest"]["winRate"] for c in takes])
        avg_take_alpha = np.mean([c["backtest"]["alphaAvgReturn"] for c in takes])
        avg_take_sample_size = np.mean([c["backtest"]["sampleSize"] for c in takes])
    else:
        avg_take_confidence = 0.0
        avg_take_win_rate = 0.0
        avg_take_alpha = 0.0
        avg_take_sample_size = 0
    
    best_candidate_score = max(
        (c["confidence"]["confidence"] for c in candidates),
        default=0.0
    )
    
    # Quality score: prioritize TAKE count, then quality of TAKEs
    quality_score = (
        take_count * 100.0 +  # Each TAKE is worth 100 points
        avg_take_confidence * 0.5 +
        avg_take_alpha * 5.0 +
        (avg_take_win_rate - 50.0) * 1.0 +
        best_candidate_score * 0.1  # Even best PASS candidate matters a bit
    )
    
    return {
        "take_count": take_count,
        "pass_count": pass_count,
        "take_tickers": take_tickers,
        "avg_take_confidence": round(avg_take_confidence, 1),
        "avg_take_win_rate": round(avg_take_win_rate, 1),
        "avg_take_alpha": round(avg_take_alpha, 2),
        "avg_take_sample_size": int(avg_take_sample_size),
        "best_candidate_score": round(best_candidate_score, 1),
        "quality_score": round(quality_score, 2),
    }


def should_keep_change(baseline_metrics: dict[str, Any], new_metrics: dict[str, Any]) -> tuple[bool, str]:
    """
    Decide if the proposed change should be kept.
    
    Keep if:
    - More TAKE decisions (as long as they're valid)
    - Higher quality TAKEs (better wr/alpha/confidence/sample)
    - Better best candidate score (getting closer to gate)
    - Higher overall quality score
    
    Returns (keep: bool, reason: str)
    """
    # Primary criterion: more valid TAKEs is always good
    if new_metrics["take_count"] > baseline_metrics["take_count"]:
        return True, f"Increased TAKE count: {baseline_metrics['take_count']} → {new_metrics['take_count']}"
    
    # If both have TAKEs, compare quality
    if new_metrics["take_count"] > 0 and baseline_metrics["take_count"] > 0:
        # Better quality TAKEs
        if new_metrics["avg_take_alpha"] > baseline_metrics["avg_take_alpha"] + 0.5:
            return True, f"Improved TAKE alpha: {baseline_metrics['avg_take_alpha']:.2f}% → {new_metrics['avg_take_alpha']:.2f}%"
        if new_metrics["avg_take_confidence"] > baseline_metrics["avg_take_confidence"] + 1.0:
            return True, f"Improved TAKE confidence: {baseline_metrics['avg_take_confidence']:.1f}% → {new_metrics['avg_take_confidence']:.1f}%"
        if new_metrics["avg_take_win_rate"] > baseline_metrics["avg_take_win_rate"] + 1.0:
            return True, f"Improved TAKE win rate: {baseline_metrics['avg_take_win_rate']:.1f}% → {new_metrics['avg_take_win_rate']:.1f}%"
    
    # If new state produces TAKEs but baseline didn't, that's progress
    if new_metrics["take_count"] > 0 and baseline_metrics["take_count"] == 0:
        return True, f"Generated {new_metrics['take_count']} TAKE(s) where baseline had none"
    
    # If neither has TAKEs, compare best candidate (getting closer to gate)
    if new_metrics["take_count"] == 0 and baseline_metrics["take_count"] == 0:
        if new_metrics["best_candidate_score"] > baseline_metrics["best_candidate_score"] + 1.0:
            return True, f"Better candidate score: {baseline_metrics['best_candidate_score']:.1f}% → {new_metrics['best_candidate_score']:.1f}%"
    
    # Overall quality score comparison
    if new_metrics["quality_score"] > baseline_metrics["quality_score"] * 1.05:  # 5% improvement
        return True, f"Quality score improved: {baseline_metrics['quality_score']:.1f} → {new_metrics['quality_score']:.1f}"
    
    # Default: don't keep
    return False, f"No improvement (quality: {baseline_metrics['quality_score']:.1f} vs {new_metrics['quality_score']:.1f})"


class DeltaCheckpointManager:
    """Manages checkpointing for delta optimization."""
    
    def __init__(self, checkpoint_dir: str = "./delta_optimization_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "delta_checkpoint.jsonl"
        self.state_file = self.checkpoint_dir / "current_state.json"
        self.changelog_file = self.checkpoint_dir / "changelog.md"
    
    def save_iteration(self, iteration: int, proposal: ChangeProposal, kept: bool, reason: str, 
                      baseline_metrics: dict, new_metrics: dict, state: ScannerState):
        """Save iteration results."""
        record = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "change": proposal.description,
            "kept": kept,
            "reason": reason,
            "baseline_metrics": baseline_metrics,
            "new_metrics": new_metrics,
            "state": state.to_dict(),
        }
        with open(self.checkpoint_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def save_state(self, state: ScannerState):
        """Save current state."""
        with open(self.state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
    
    def load_state(self) -> Optional[ScannerState]:
        """Load saved state."""
        if not self.state_file.exists():
            return None
        with open(self.state_file) as f:
            d = json.load(f)
            state = ScannerState()
            state.from_dict(d)
            return state
    
    def load_progress(self) -> tuple[int, list[dict]]:
        """Load progress from checkpoint."""
        if not self.checkpoint_file.exists():
            return 0, []
        
        records = []
        with open(self.checkpoint_file) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        last_iteration = max((r["iteration"] for r in records), default=0)
        return last_iteration, records
    
    def update_changelog(self, iteration: int, proposal: ChangeProposal, kept: bool, 
                        metrics: dict, state: ScannerState):
        """Update the changelog file."""
        if kept:
            with open(self.changelog_file, "a") as f:
                f.write(f"\n## Iteration {iteration} - KEPT\n")
                f.write(f"**Change**: {proposal.description}\n")
                f.write(f"**Reason**: Improved outcomes\n")
                f.write(f"**Results**:\n")
                f.write(f"- TAKE count: {metrics['take_count']}\n")
                if metrics['take_count'] > 0:
                    f.write(f"- TAKE tickers: {', '.join(metrics['take_tickers'])}\n")
                    f.write(f"- Avg confidence: {metrics['avg_take_confidence']}%\n")
                    f.write(f"- Avg win rate: {metrics['avg_take_win_rate']}%\n")
                    f.write(f"- Avg alpha vs SPY: {metrics['avg_take_alpha']}%\n")
                f.write(f"- Best candidate: {metrics['best_candidate_score']}%\n")
                f.write(f"- Quality score: {metrics['quality_score']}\n")
                f.write(f"\n**Current State**:\n")
                f.write(f"```json\n{json.dumps(state.to_dict(), indent=2)}\n```\n")
