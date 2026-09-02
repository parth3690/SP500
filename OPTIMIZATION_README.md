# Self-Improving Alpha Scanner Optimization

This document describes the self-improving parameter optimization loop for the institutional alpha scanner.

## Overview

The optimization system iteratively searches for better scanner configurations by:

1. **Sampling** parameter combinations from a defined search space
2. **Running** walk-forward backtests and simulations for each configuration
3. **Evaluating** results based on TAKE counts and quality metrics
4. **Persisting** checkpoints to enable resumption of long runs
5. **Tracking** the best configurations found

The loop optimizes parameters that affect the scanner's signal generation and trade gate, while preserving the existing walk-forward backtest and simulation validation framework.

## Parameters Being Optimized

The optimization searches over the following parameter space:

| Parameter | Range/Options | Description |
|-----------|--------------|-------------|
| `signal_threshold` | 60.0 - 75.0 | Alpha score threshold for BUY signals in walk-forward backtest |
| `forward_horizon` | 15, 20, 25 days | Forward return horizon for measuring backtest performance |
| `risk_mode` | aggressive, balanced, defensive | Risk scoring profile used in alpha calculation |
| `min_score` | 55.0 - 70.0 | Minimum alpha score for candidates to be considered |
| `step_size` | 3, 5, 7 days | Walk-forward step size between backtest samples |

## Fitness Function

Each configuration is evaluated using a fitness function that prioritizes:

1. **TAKE count**: Number of candidates that pass the hard trade gate (10x weight)
2. **Average TAKE confidence**: Quality of TAKE decisions (0.5x weight)
3. **Average TAKE alpha**: Alpha vs SPY benchmark (2x weight)
4. **Average TAKE win rate**: Backtest win rate above 50% (0.3x weight)

```
Fitness = take_count * 10.0 
        + avg_take_confidence * 0.5 
        + avg_take_alpha * 2.0 
        + (avg_take_win_rate - 50.0) * 0.3
```

## Usage

### Quick Start - Smoke Test

Run a quick 10-iteration smoke test on 20 stocks to verify the system works:

```bash
cd /workspace
python3 backend/run_optimization.py --smoke-test
```

Expected output:
- Loads S&P 500 constituents and price data
- Runs 10 iterations with different parameter combinations
- Saves checkpoints to `./optimization_checkpoints/`
- Takes ~1-2 minutes

### Production Run - 100 Iterations

Run 100 iterations on the full S&P 500 universe:

```bash
cd /workspace
python3 backend/run_optimization.py --iterations 100
```

Expected duration: ~3-4 hours for 100 iterations on full S&P 500

### Production Run - 10,000 Iterations

To run the full 10,000 iterations as requested:

```bash
cd /workspace
python3 backend/run_optimization.py --iterations 10000
```

**Important**: This will take approximately 390 hours (~16 days) to complete on the full S&P 500 universe. The run can be:
- **Interrupted** at any time (Ctrl+C)
- **Resumed** later using `--resume` (default behavior)
- **Monitored** by checking the log files

### Resume from Checkpoint

The optimization automatically saves progress after each iteration. To resume:

```bash
cd /workspace
python3 backend/run_optimization.py --iterations 10000 --resume
```

This will continue from the last completed iteration.

### Start Fresh (Ignore Checkpoints)

To start a new optimization run from scratch:

```bash
cd /workspace
python3 backend/run_optimization.py --iterations 100 --no-resume
```

### Custom Universe Size

To run on a subset of stocks (useful for faster testing):

```bash
cd /workspace
python3 backend/run_optimization.py --iterations 500 --universe-size 100
```

This runs 500 iterations on a random sample of 100 stocks.

## Command-Line Options

```
--iterations N          Number of optimization iterations to run (default: 100)
--limit N               Number of candidates to return per scan (default: 20)
--checkpoint-dir DIR    Directory to store checkpoints (default: ./optimization_checkpoints)
--resume                Resume from existing checkpoint if available (default: True)
--no-resume             Start fresh, ignore existing checkpoints
--universe-size N       Limit to N stocks for testing (default: use all S&P 500)
--smoke-test            Run quick smoke test (10 iterations, 20 stocks)
```

## Output Files

The optimization creates the following files:

### `optimization_checkpoints/checkpoint.jsonl`

JSONL file with one entry per iteration containing:
- Iteration number
- Timestamp
- Parameter configuration tested
- Results (TAKE count, PASS count, fitness, metrics)

Example entry:
```json
{
  "iteration": 42,
  "timestamp": "2026-09-02T21:08:26.405587+00:00",
  "params": {
    "signal_threshold": 64.5,
    "forward_horizon": 20,
    "risk_mode": "balanced",
    "min_score": 62.0,
    "step_size": 5
  },
  "results": {
    "take_count": 3,
    "pass_count": 17,
    "total_candidates": 20,
    "avg_take_confidence": 78.2,
    "avg_take_win_rate": 64.5,
    "avg_take_alpha": 4.8,
    "avg_take_sample_size": 28,
    "best_candidate_score": 82.1,
    "fitness": 39.77
  }
}
```

### `optimization_checkpoints/best_configs.json`

JSON file with the top 10 best configurations found so far, sorted by fitness. Updated after each new best configuration is found.

## Trade Gate Thresholds (Unchanged)

The optimization preserves the existing hard trade gate requirements:

- **Confidence** ≥ 75%
- **Win Rate** ≥ 62%
- **Sample Size** ≥ 20
- **Alpha vs SPY** ≥ 3%
- **All simulation scenarios survive** (bull, base, bear, high-vol with transaction costs)

## Monitoring Progress

To monitor a running optimization:

```bash
# View last 50 lines of output
tail -50 optimization_run.log

# Follow live output
tail -f optimization_run.log

# Count completed iterations
grep "^Iteration" optimization_checkpoints/checkpoint.jsonl | wc -l

# View best configurations found so far
cat optimization_checkpoints/best_configs.json | python3 -m json.tool
```

## Expected Behavior

### Early Iterations (1-100)
- Mostly PASS decisions due to stringent gate conditions
- Establishing baseline performance
- Exploring parameter space randomly

### Mid Iterations (100-1000)
- May find occasional TAKE candidates
- Best configurations emerge
- Fitness scores improve gradually

### Late Iterations (1000-10000)
- Continued refinement
- Rare but high-quality TAKE configurations
- Diminishing returns as search space is explored

### No TAKE Results
If no configurations produce TAKE decisions after many iterations, this indicates:
- The current market regime is not conducive to passing the gate
- The gate thresholds are correctly stringent (protecting against false signals)
- The scanner is functioning as designed (PASS is the default safe state)

This is **not a failure** - it means the scanner is correctly withholding recommendations when edge is not sufficient.

## Technical Details

### Walk-Forward Backtest
- Uses only historical data at each point (no lookahead bias)
- Steps forward through time in configurable increments
- Measures 15/20/25-day forward returns
- Generates BUY signals when alpha score ≥ threshold and expected return > 0
- Computes win rate, alpha vs SPY, max drawdown, sample size

### Simulation Validation
- Stresses the measured edge under 4 scenarios: bull, base, bear, high-vol
- Applies 25 bps transaction costs (20 bps round-trip + 5 bps slippage)
- Base case must show positive alpha after costs to pass gate
- Bear/high-vol scenarios can degrade but shouldn't blow up

### Checkpoint Format
- JSONL (JSON Lines) format for append-only writes
- Each iteration writes atomically
- Can be resumed from any point
- Safe for long-running processes

## Troubleshooting

### "ModuleNotFoundError" when running
```bash
cd /workspace/backend
python3 -m pip install -r requirements.txt
```

### Out of memory errors
Reduce the universe size:
```bash
python3 backend/run_optimization.py --iterations 1000 --universe-size 250
```

### Slow progress
This is expected. Each iteration:
- Runs full walk-forward backtests on many stocks
- Computes simulations with 1000 Monte Carlo samples per scenario
- On full S&P 500, expect ~2-3 minutes per iteration

### All PASS, no TAKE
This is correct behavior when market conditions don't meet the gate requirements. The scanner is designed to only emit TAKE when confidence is high.

## Next Steps

After optimization completes:

1. Review `best_configs.json` for top configurations
2. Analyze which parameters correlate with better fitness
3. Potentially adjust parameter ranges for further refinement
4. Test the best configuration in live scanning
5. Monitor real-world performance

## Implementation Files

- `backend/app/services/alpha_optimizer.py` - Core optimization engine
- `backend/app/services/institutional_scanner.py` - Modified to accept parameters
- `backend/run_optimization.py` - CLI script for running optimization
