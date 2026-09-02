"""
Self-improving parameter optimization loop for the institutional alpha scanner.

This module iteratively searches for better scanner configurations by:
1. Sampling parameter combinations from a defined search space
2. Running walk-forward backtests and simulations for each configuration
3. Evaluating results based on TAKE counts and quality metrics
4. Persisting checkpoints to enable resumption
5. Tracking the best configurations found

The loop optimizes parameters that affect the scanner's signal generation and gate,
while preserving the existing walk-forward backtest and simulation validation framework.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..models import Constituent
from .institutional_scanner import (
    MIN_ALPHA_VS_BENCHMARK,
    MIN_BACKTEST_SAMPLE_SIZE,
    MIN_BACKTEST_WIN_RATE,
    MIN_CONFIDENCE_FOR_TAKE,
    scan_institutional_grade,
)


class ParameterSpace:
    """
    Defines the search space for scanner parameters.
    
    Parameters to optimize:
    - signal_threshold: Alpha score threshold for BUY signals in walk-forward (currently hardcoded at 68.0)
    - forward_horizon: Days to measure forward returns (currently 20)
    - risk_mode: aggressive, balanced, or defensive
    - min_score: Minimum alpha score for candidates (default 65.0)
    - step_size: Walk-forward step size in days (currently 5)
    """
    
    def __init__(self):
        self.ranges = {
            "signal_threshold": (60.0, 75.0),  # Alpha score threshold for signal
            "forward_horizon": [15, 20, 25],    # Forward return horizon in days
            "risk_mode": ["aggressive", "balanced", "defensive"],
            "min_score": (55.0, 70.0),          # Minimum candidate score
            "step_size": [3, 5, 7],             # Walk-forward step size
        }
    
    def sample(self, method: str = "random") -> dict[str, Any]:
        """Sample a parameter configuration from the search space."""
        if method == "random":
            return {
                "signal_threshold": random.uniform(*self.ranges["signal_threshold"]),
                "forward_horizon": random.choice(self.ranges["forward_horizon"]),
                "risk_mode": random.choice(self.ranges["risk_mode"]),
                "min_score": random.uniform(*self.ranges["min_score"]),
                "step_size": random.choice(self.ranges["step_size"]),
            }
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def default(self) -> dict[str, Any]:
        """Return the default (current) parameter configuration."""
        return {
            "signal_threshold": 68.0,
            "forward_horizon": 20,
            "risk_mode": "balanced",
            "min_score": 65.0,
            "step_size": 5,
        }


class CheckpointManager:
    """Manages persistence of optimization progress."""
    
    def __init__(self, checkpoint_dir: str = "./optimization_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.jsonl"
        self.best_file = self.checkpoint_dir / "best_configs.json"
    
    def save_iteration(self, iteration: int, params: dict[str, Any], results: dict[str, Any]):
        """Append iteration results to checkpoint file."""
        record = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "results": results,
        }
        with open(self.checkpoint_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def save_best_configs(self, best_configs: list[dict[str, Any]]):
        """Save the best configurations found so far."""
        with open(self.best_file, "w") as f:
            json.dump(best_configs, f, indent=2)
    
    def load_progress(self) -> tuple[int, list[dict[str, Any]]]:
        """
        Load progress from checkpoint file.
        Returns (last_iteration, all_records).
        """
        if not self.checkpoint_file.exists():
            return 0, []
        
        records = []
        with open(self.checkpoint_file) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        last_iteration = max((r["iteration"] for r in records), default=0)
        return last_iteration, records
    
    def load_best_configs(self) -> list[dict[str, Any]]:
        """Load the best configurations from file."""
        if not self.best_file.exists():
            return []
        with open(self.best_file) as f:
            return json.load(f)


def evaluate_scan_results(scan_result: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate scanner results to produce optimization metrics.
    
    Returns metrics including:
    - take_count: Number of TAKE decisions
    - pass_count: Number of PASS decisions
    - avg_take_confidence: Average confidence of TAKE decisions
    - avg_take_win_rate: Average win rate of TAKE decisions
    - avg_take_alpha: Average alpha vs benchmark of TAKE decisions
    - avg_take_sample_size: Average backtest sample size of TAKE decisions
    - best_candidate_score: Highest confidence among all candidates
    """
    candidates = scan_result.get("candidates", [])
    
    if not candidates:
        return {
            "take_count": 0,
            "pass_count": 0,
            "total_candidates": 0,
            "avg_take_confidence": 0.0,
            "avg_take_win_rate": 0.0,
            "avg_take_alpha": 0.0,
            "avg_take_sample_size": 0,
            "best_candidate_score": 0.0,
            "fitness": 0.0,
        }
    
    takes = [c for c in candidates if c.get("tradeGate", {}).get("decision") == "TAKE"]
    passes = [c for c in candidates if c.get("tradeGate", {}).get("decision") == "PASS"]
    
    take_count = len(takes)
    pass_count = len(passes)
    
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
    
    # Fitness function: prioritize TAKE count, then quality of TAKE candidates
    # Fitness = take_count * 10 + avg_take_confidence + avg_take_alpha * 2
    fitness = (
        take_count * 10.0 +
        avg_take_confidence * 0.5 +
        avg_take_alpha * 2.0 +
        (avg_take_win_rate - 50.0) * 0.3
    )
    
    return {
        "take_count": take_count,
        "pass_count": pass_count,
        "total_candidates": len(candidates),
        "avg_take_confidence": round(avg_take_confidence, 2),
        "avg_take_win_rate": round(avg_take_win_rate, 2),
        "avg_take_alpha": round(avg_take_alpha, 2),
        "avg_take_sample_size": int(avg_take_sample_size),
        "best_candidate_score": round(best_candidate_score, 2),
        "fitness": round(fitness, 2),
    }


def run_scan_with_params(
    constituents: list[Constituent],
    close_prices: pd.DataFrame,
    params: dict[str, Any],
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Run the institutional scanner with the given parameters.
    """
    # Run scanner with all optimization parameters
    result = scan_institutional_grade(
        constituents,
        close_prices,
        limit=limit,
        min_score=params["min_score"],
        risk_mode=params["risk_mode"],
        regime_override="auto",
        signal_threshold=params["signal_threshold"],
        forward_horizon=params["forward_horizon"],
        step_size=params["step_size"],
    )
    
    return result


def optimize_parameters(
    constituents: list[Constituent],
    close_prices: pd.DataFrame,
    *,
    iterations: int = 100,
    limit: int = 20,
    checkpoint_dir: str = "./optimization_checkpoints",
    resume: bool = True,
    universe_size: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run the self-improving optimization loop.
    
    Args:
        constituents: List of stock constituents to scan
        close_prices: Historical price data
        iterations: Number of optimization iterations to run
        limit: Number of candidates to return per scan
        checkpoint_dir: Directory to store checkpoints
        resume: Whether to resume from existing checkpoint
        universe_size: If set, limit to a subset of constituents (for smoke tests)
    
    Returns:
        Summary of optimization results including best configurations found
    """
    # Initialize
    param_space = ParameterSpace()
    checkpoint_mgr = CheckpointManager(checkpoint_dir)
    
    # Optionally limit universe for smoke tests
    if universe_size is not None and universe_size < len(constituents):
        constituents = random.sample(constituents, universe_size)
        print(f"Limited universe to {universe_size} stocks for testing")
    
    # Load progress if resuming
    start_iteration = 1
    if resume:
        last_iteration, past_records = checkpoint_mgr.load_progress()
        if last_iteration > 0:
            start_iteration = last_iteration + 1
            print(f"Resuming from iteration {start_iteration}")
    
    # Track best configurations
    best_configs: list[dict[str, Any]] = []
    best_fitness = -float("inf")
    
    # Run optimization loop
    for iteration in range(start_iteration, start_iteration + iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{start_iteration + iterations - 1}")
        print(f"{'='*60}")
        
        # Sample parameters (include baseline every 10th iteration)
        if iteration % 10 == 1:
            params = param_space.default()
            print("Using DEFAULT parameters (baseline)")
        else:
            params = param_space.sample()
            print("Sampled RANDOM parameters")
        
        print(f"Parameters: {json.dumps(params, indent=2)}")
        
        # Run scanner with these parameters
        try:
            scan_result = run_scan_with_params(
                constituents,
                close_prices,
                params,
                limit=limit,
            )
            
            # Evaluate results
            metrics = evaluate_scan_results(scan_result)
            print(f"\nResults:")
            print(f"  TAKE count: {metrics['take_count']}")
            print(f"  PASS count: {metrics['pass_count']}")
            print(f"  Total candidates: {metrics['total_candidates']}")
            if metrics['take_count'] > 0:
                print(f"  Avg TAKE confidence: {metrics['avg_take_confidence']:.1f}%")
                print(f"  Avg TAKE win rate: {metrics['avg_take_win_rate']:.1f}%")
                print(f"  Avg TAKE alpha: {metrics['avg_take_alpha']:.2f}%")
                print(f"  Avg TAKE sample size: {metrics['avg_take_sample_size']}")
            print(f"  Fitness: {metrics['fitness']:.2f}")
            
            # Track best configuration
            if metrics["fitness"] > best_fitness:
                best_fitness = metrics["fitness"]
                best_configs.append({
                    "iteration": iteration,
                    "params": params,
                    "metrics": metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                # Keep only top 10 best configs
                best_configs = sorted(
                    best_configs,
                    key=lambda x: x["metrics"]["fitness"],
                    reverse=True
                )[:10]
                print(f"\n🎯 NEW BEST configuration! Fitness: {best_fitness:.2f}")
                checkpoint_mgr.save_best_configs(best_configs)
            
            # Save checkpoint
            checkpoint_mgr.save_iteration(iteration, params, metrics)
            
        except Exception as e:
            print(f"ERROR in iteration {iteration}: {e}")
            # Save error to checkpoint
            checkpoint_mgr.save_iteration(
                iteration,
                params,
                {"error": str(e), "fitness": -float("inf")},
            )
            continue
    
    # Final summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {iterations}")
    print(f"Best fitness achieved: {best_fitness:.2f}")
    print(f"\nTop 3 configurations:")
    for i, config in enumerate(best_configs[:3], 1):
        print(f"\n{i}. Iteration {config['iteration']} (fitness: {config['metrics']['fitness']:.2f})")
        print(f"   Parameters: {json.dumps(config['params'], indent=6)}")
        print(f"   TAKE count: {config['metrics']['take_count']}")
        print(f"   Avg TAKE confidence: {config['metrics']['avg_take_confidence']:.1f}%")
        print(f"   Avg TAKE win rate: {config['metrics']['avg_take_win_rate']:.1f}%")
        print(f"   Avg TAKE alpha: {config['metrics']['avg_take_alpha']:.2f}%")
    
    return {
        "iterations": iterations,
        "best_fitness": best_fitness,
        "best_configs": best_configs,
        "checkpoint_dir": checkpoint_dir,
    }
