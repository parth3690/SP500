#!/usr/bin/env python3
"""
CLI script to run the delta improvement loop for the institutional alpha scanner.

This implements a state-of-the-art iterative improvement system:
- Each iteration proposes ONE concrete change
- Evaluates on full universe (S&P 500 + NYSE SMID + QQQ/Nasdaq-100)
- KEEPs if improvement, REVERTs otherwise
- Tracks detailed changelog of all kept changes

Usage:
    # Run 10,000 iterations
    python3 backend/run_delta_optimization.py --iterations 10000
    
    # Run smoke test with 10 iterations on small universe
    python3 backend/run_delta_optimization.py --smoke-test
    
    # Resume from checkpoint
    python3 backend/run_delta_optimization.py --iterations 10000 --resume
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.app.services.delta_optimizer import (
    DeltaCheckpointManager,
    ScannerState,
    evaluate_state,
    load_full_universe,
    propose_change,
    should_keep_change,
)
from backend.app.services.prices import fetch_close_prices
from backend.app.services.alpha import alpha_universe_tickers


def run_delta_optimization(
    *,
    iterations: int = 100,
    limit: int = 20,
    checkpoint_dir: str = "./delta_optimization_checkpoints",
    resume: bool = True,
    universe_size: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run the delta improvement optimization loop.
    
    Args:
        iterations: Number of optimization iterations to run
        limit: Number of candidates to return per scan
        checkpoint_dir: Directory to store checkpoints
        resume: Whether to resume from existing checkpoint
        universe_size: If set, limit to subset of universe (for smoke tests)
    
    Returns:
        Summary of optimization results
    """
    # Initialize checkpoint manager
    checkpoint_mgr = DeltaCheckpointManager(checkpoint_dir)
    
    # Load or initialize state
    start_iteration = 1
    if resume:
        loaded_state = checkpoint_mgr.load_state()
        if loaded_state:
            current_state = loaded_state
            last_iteration, past_records = checkpoint_mgr.load_progress()
            start_iteration = last_iteration + 1
            print(f"✅ Resumed from iteration {last_iteration}")
            print(f"Current state: {current_state.to_dict()}")
        else:
            current_state = ScannerState()
            past_records = []
            print("Starting fresh - no checkpoint found")
    else:
        current_state = ScannerState()
        past_records = []
        print("Starting fresh optimization")
    
    # Load universe
    print("\n📊 Loading tradable universe...")
    universe = load_full_universe(refresh=False)
    
    print(f"  S&P 500: {len(universe['sp500'])} constituents")
    print(f"  NYSE SMID: {len(universe['nyse_smid'])} constituents")
    print(f"  Nasdaq-100: {len(universe['nasdaq100'])} constituents")
    print(f"  Total unique: {len(universe['all'])} constituents")
    
    # Use appropriate universe
    if universe_size is not None and universe_size < len(universe['all']):
        import random
        constituents = random.sample(universe['all'], universe_size)
        print(f"\n🧪 SMOKE TEST: Limited to {universe_size} stocks")
    else:
        constituents = universe['all']
        print(f"\n🎯 Full universe: {len(constituents)} stocks")
    
    # Load price data
    print("\n📈 Fetching price data...")
    tickers = alpha_universe_tickers(constituents)
    end_date = date.today()
    start_date = end_date - timedelta(days=760)
    close_prices = fetch_close_prices(tickers, start_date, end_date)
    print(f"  Loaded {len(close_prices.columns)} tickers")
    print(f"  Date range: {close_prices.index.min()} to {close_prices.index.max()}")
    print(f"  Total days: {len(close_prices)}")
    
    # Evaluate baseline
    print("\n🎯 Evaluating baseline state...")
    baseline_metrics = evaluate_state(current_state, constituents, close_prices, limit=limit)
    print(f"  TAKE count: {baseline_metrics['take_count']}")
    print(f"  PASS count: {baseline_metrics['pass_count']}")
    print(f"  Best candidate: {baseline_metrics['best_candidate_score']}%")
    print(f"  Quality score: {baseline_metrics['quality_score']}")
    
    # Run optimization loop
    print(f"\n{'='*60}")
    print(f"STARTING DELTA OPTIMIZATION LOOP")
    print(f"{'='*60}\n")
    
    kept_count = 0
    reverted_count = 0
    
    for iteration in range(start_iteration, start_iteration + iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{start_iteration + iterations - 1}")
        print(f"{'='*60}")
        
        # Propose change
        proposal = propose_change(iteration, current_state, past_records)
        print(f"\n📝 Proposed: {proposal.description}")
        
        # Apply change and evaluate
        test_state = proposal.apply(current_state)
        print(f"\n⚙️  Testing change...")
        new_metrics = evaluate_state(test_state, constituents, close_prices, limit=limit)
        
        # Decide keep/revert
        keep, reason = should_keep_change(baseline_metrics, new_metrics)
        
        if keep:
            # KEEP the change
            current_state = test_state
            baseline_metrics = new_metrics
            kept_count += 1
            print(f"\n✅ KEPT - {reason}")
            print(f"  TAKE count: {new_metrics['take_count']}")
            if new_metrics['take_count'] > 0:
                print(f"  TAKE tickers: {', '.join(new_metrics['take_tickers'])}")
                print(f"  Avg confidence: {new_metrics['avg_take_confidence']}%")
                print(f"  Avg win rate: {new_metrics['avg_take_win_rate']}%")
                print(f"  Avg alpha: {new_metrics['avg_take_alpha']}%")
            print(f"  Best candidate: {new_metrics['best_candidate_score']}%")
            print(f"  Quality score: {new_metrics['quality_score']}")
            
            # Update changelog
            checkpoint_mgr.update_changelog(iteration, proposal, True, new_metrics, current_state)
        else:
            # REVERT the change
            reverted_count += 1
            print(f"\n❌ REVERTED - {reason}")
            print(f"  Baseline still better")
        
        # Save checkpoint
        checkpoint_mgr.save_iteration(
            iteration, proposal, keep, reason,
            baseline_metrics, new_metrics, current_state
        )
        checkpoint_mgr.save_state(current_state)
    
    # Final summary
    print(f"\n{'='*60}")
    print("DELTA OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {iterations}")
    print(f"Changes KEPT: {kept_count}")
    print(f"Changes REVERTED: {reverted_count}")
    print(f"\nFinal state:")
    print(f"  TAKE count: {baseline_metrics['take_count']}")
    if baseline_metrics['take_count'] > 0:
        print(f"  TAKE tickers: {', '.join(baseline_metrics['take_tickers'])}")
        print(f"  Avg confidence: {baseline_metrics['avg_take_confidence']}%")
        print(f"  Avg win rate: {baseline_metrics['avg_take_win_rate']}%")
        print(f"  Avg alpha: {baseline_metrics['avg_take_alpha']}%")
    print(f"  Best candidate: {baseline_metrics['best_candidate_score']}%")
    print(f"  Quality score: {baseline_metrics['quality_score']}")
    print(f"\nCheckpoint dir: {checkpoint_dir}")
    print(f"Changelog: {Path(checkpoint_dir) / 'changelog.md'}")
    
    return {
        "iterations": iterations,
        "kept_count": kept_count,
        "reverted_count": reverted_count,
        "final_metrics": baseline_metrics,
        "final_state": current_state.to_dict(),
        "checkpoint_dir": checkpoint_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run delta improvement loop for institutional alpha scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of optimization iterations to run (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of candidates to return per scan (default: 20)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./delta_optimization_checkpoints",
        help="Directory to store checkpoints (default: ./delta_optimization_checkpoints)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing checkpoint if available (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start fresh, ignore existing checkpoints",
    )
    parser.add_argument(
        "--universe-size",
        type=int,
        default=None,
        help="Limit to N stocks for testing (default: use full universe)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test (10 iterations, 50 stocks)",
    )
    
    args = parser.parse_args()
    
    # Override for smoke test
    if args.smoke_test:
        args.iterations = 10
        args.universe_size = 50
        print("\n🧪 SMOKE TEST MODE: 10 iterations, 50 stocks")
    
    print("\n" + "="*60)
    print("DELTA IMPROVEMENT OPTIMIZATION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iterations: {args.iterations}")
    print(f"Resume: {args.resume}")
    print(f"Universe size: {args.universe_size or 'Full (S&P 500 + NYSE SMID + Nasdaq-100)'}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("="*60 + "\n")
    
    # Run optimization
    result = run_delta_optimization(
        iterations=args.iterations,
        limit=args.limit,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        universe_size=args.universe_size,
    )
    
    return 0


if __name__ == "__main__":
    # Import needs to be here to avoid circular import
    from typing import Any, Optional
    sys.exit(main())
