#!/usr/bin/env python3
"""
CLI script to run the self-improving optimization loop for the institutional alpha scanner.

Usage:
    # Run 10,000 iterations
    python -m backend.run_optimization --iterations 10000
    
    # Run a smoke test with 10 iterations on 20 stocks
    python -m backend.run_optimization --iterations 10 --universe-size 20 --smoke-test
    
    # Resume from checkpoint
    python -m backend.run_optimization --iterations 5000 --resume
    
    # Start fresh (ignore existing checkpoints)
    python -m backend.run_optimization --iterations 100 --no-resume
"""

import argparse
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Add the workspace root to sys.path so we can import backend.app modules
workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.app.models import Constituent
from backend.app.services.alpha import alpha_universe_tickers
from backend.app.services.alpha_optimizer import optimize_parameters
from backend.app.services.prices import fetch_close_prices
from backend.app.services.sp500 import get_sp500_constituents_cached


def main():
    parser = argparse.ArgumentParser(
        description="Run self-improving optimization loop for institutional alpha scanner",
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
        default="./optimization_checkpoints",
        help="Directory to store checkpoints (default: ./optimization_checkpoints)",
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
        help="Limit to N stocks for testing (default: use all S&P 500)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test (10 iterations, 20 stocks)",
    )
    
    args = parser.parse_args()
    
    # Override for smoke test
    if args.smoke_test:
        args.iterations = 10
        args.universe_size = 20
        print("\n🧪 SMOKE TEST MODE: 10 iterations, 20 stocks")
    
    print("\n" + "="*60)
    print("INSTITUTIONAL ALPHA SCANNER OPTIMIZATION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Iterations: {args.iterations}")
    print(f"Resume: {args.resume}")
    print(f"Universe size: {args.universe_size or 'Full S&P 500'}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("="*60 + "\n")
    
    # Load S&P 500 constituents
    print("Loading S&P 500 constituents...")
    constituents = get_sp500_constituents_cached()
    print(f"Loaded {len(constituents)} constituents")
    
    # Load price data
    print("\nFetching price data...")
    tickers = alpha_universe_tickers(constituents)
    end_date = date.today()
    start_date = end_date - timedelta(days=760)  # Need ~3 years for walk-forward backtest
    close_prices = fetch_close_prices(tickers, start_date, end_date)
    print(f"Loaded price data for {len(close_prices.columns)} tickers")
    print(f"Price data range: {close_prices.index.min()} to {close_prices.index.max()}")
    print(f"Total days: {len(close_prices)}")
    
    # Run optimization
    print("\nStarting optimization loop...\n")
    result = optimize_parameters(
        constituents,
        close_prices,
        iterations=args.iterations,
        limit=args.limit,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        universe_size=args.universe_size,
    )
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total iterations: {result['iterations']}")
    print(f"Best fitness: {result['best_fitness']:.2f}")
    print(f"Checkpoints saved to: {result['checkpoint_dir']}")
    print("\nBest configurations saved to:")
    print(f"  {Path(result['checkpoint_dir']) / 'best_configs.json'}")
    print("\nAll iteration results saved to:")
    print(f"  {Path(result['checkpoint_dir']) / 'checkpoint.jsonl'}")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
