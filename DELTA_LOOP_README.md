# Delta Improvement Loop for State-of-the-Art Alpha Scanner

This document describes the delta improvement optimization system - a fundamental upgrade from the previous random parameter search.

## What Changed from the Previous Loop

### Previous System (Iterations 1-233)
- **Random parameter search** across 5 scanner parameters
- **S&P 500 only** universe
- Fitness-based evaluation but no keep/revert logic
- All parameter combinations logged equally
- Result: 0 TAKE, best confidence 71.4%

### New System (Delta Loop)
- **Iterative improvement** with keep/revert logic
- **Full universe**: S&P 500 + NYSE SMID + Nasdaq-100 (QQQ)
- Each iteration proposes **ONE concrete change**
- **KEEP** only if it improves outcomes, otherwise **REVERT**
- Detailed changelog tracks all kept changes
- Goal: Find configurations that produce genuine high win-prob names

## Universe Expansion

The scanner now covers the complete tradable universe:

| Source | Description | Count | Status |
|--------|-------------|-------|--------|
| **S&P 500** | Large cap US equities | 503 | ✅ Active |
| **NYSE SMID** | NYSE $100M-$2B market cap | Variable | ✅ Fixed (FMP required) |
| **Nasdaq-100** | QQQ top holdings | 50 | ✅ Active |
| **Total** | Deduplicated universe | ~550 | ✅ Active |

**Note**: NYSE SMID requires FMP_API_KEY. Without it, scanner still covers S&P 500 + Nasdaq-100 (>500 stocks).

## Delta Loop Architecture

### 1. State Management

The system maintains a `ScannerState` with:
- `signal_threshold`: Alpha score threshold for BUY signals (60-75)
- `forward_horizon`: Forward return measurement period (15/20/25 days)
- `risk_mode`: Risk scoring profile (aggressive/balanced/defensive)
- `min_score`: Minimum alpha score for candidates (55-70)
- `step_size`: Walk-forward step size (3/5/7 days)
- `modifications`: List of all applied changes (changelog)

### 2. Change Proposal

Each iteration proposes ONE change:
- **Informed by past results**: Analyzes last 10 iterations
- **Adaptive strategy**: 
  - If close to threshold (70-75%), try subtle refinements
  - Otherwise, explore broadly across all parameters
- **Types of changes**:
  - Lower/raise signal threshold
  - Change forward horizon
  - Adjust min score
  - Change risk mode
  - Adjust step size

### 3. Evaluation

For each proposed change:
1. Apply change to create test state
2. Run full scanner on entire universe
3. Measure:
   - TAKE count
   - TAKE quality (confidence, win rate, alpha, sample size)
   - Best candidate score
   - Overall quality score

### 4. Keep/Revert Decision

**KEEP if:**
- More TAKE decisions that pass the gate
- Higher TAKE quality (better wr/alpha/confidence)
- Best candidate score improves by ≥2.0 percentage points
- Quality score improves by >2%

**REVERT if:**
- No meaningful improvement on any metric
- Change makes things worse

### 5. Checkpoint & Changelog

After each iteration:
- Save to `delta_checkpoint.jsonl` (full iteration data)
- Update `current_state.json` (resumable state)
- Append to `changelog.md` (kept changes only)

## Quality Score Formula

```python
quality_score = (
    take_count * 100.0 +           # Each TAKE worth 100 points
    avg_take_confidence * 0.5 +
    avg_take_alpha * 5.0 +
    (avg_take_win_rate - 50.0) * 1.0 +
    best_candidate_score * 0.1     # Even best PASS matters
)
```

Prioritizes:
1. **TAKE count** (10x weight) - actual listings that clear gate
2. **Alpha vs SPY** (5x weight) - outperformance vs benchmark
3. **Win rate** (1x weight) - reliability of signal
4. **Confidence** (0.5x weight) - calibrated certainty
5. **Best candidate** (0.1x weight) - progress toward threshold

## Desk Gate (Unchanged)

The hard gate requirements are **preserved** (not weakened):

- ✅ Confidence ≥ 75%
- ✅ Win rate ≥ 62%
- ✅ Sample size ≥ 20
- ✅ Alpha vs SPY ≥ 3%
- ✅ All simulations survive (bull/base/bear/high-vol with 25 bps costs)

Default is **PASS**. Empty book is valid.

## Usage

### Smoke Test (10 iterations, 50 stocks)
```bash
cd /workspace
python3 backend/run_delta_optimization.py --smoke-test
```

Expected: ~30 seconds, validates keep/revert logic

### Production Run (100 iterations, full universe)
```bash
cd /workspace
python3 backend/run_delta_optimization.py --iterations 100
```

Expected: ~45 minutes for 100 iterations

### Full 10,000 Iterations
```bash
cd /workspace
python3 backend/run_delta_optimization.py --iterations 10000 --resume
```

Expected: ~75 hours for full run, can be interrupted/resumed

### Command-Line Options

```
--iterations N          Number of iterations to run (default: 100)
--limit N               Candidates per scan (default: 20)
--checkpoint-dir DIR    Checkpoint directory (default: ./delta_optimization_checkpoints)
--resume                Resume from checkpoint (default: True)
--no-resume             Start fresh
--universe-size N       Limit universe for testing (default: full)
--smoke-test            Quick test: 10 iterations, 50 stocks
```

## Output Files

### `delta_checkpoint.jsonl`
JSONL file with one entry per iteration:
```json
{
  "iteration": 42,
  "timestamp": "2026-09-03T01:50:00Z",
  "change": "Lower signal_threshold from 68.0 to 66.5",
  "kept": true,
  "reason": "Better candidate score: 57.9% → 60.2%",
  "baseline_metrics": {...},
  "new_metrics": {...},
  "state": {...}
}
```

### `current_state.json`
Current scanner state (for resumption):
```json
{
  "signal_threshold": 66.5,
  "forward_horizon": 20,
  "risk_mode": "balanced",
  "min_score": 65.0,
  "step_size": 5,
  "modifications": [
    "Lower signal_threshold from 68.0 to 66.5"
  ]
}
```

### `changelog.md`
Human-readable changelog of **kept changes only**:
```markdown
## Iteration 42 - KEPT
**Change**: Lower signal_threshold from 68.0 to 66.5
**Reason**: Improved outcomes
**Results**:
- TAKE count: 0
- Best candidate: 60.2%
- Quality score: -42.1
```

## Monitoring Progress

```bash
# View last 30 lines
tail -30 delta_optimization_run.log

# Count kept vs reverted
grep "KEPT" delta_optimization_checkpoints/delta_checkpoint.jsonl | wc -l
grep "REVERTED" delta_optimization_checkpoints/delta_checkpoint.jsonl | wc -l

# View changelog
cat delta_optimization_checkpoints/changelog.md

# Check current iteration
tail -1 delta_optimization_checkpoints/delta_checkpoint.jsonl | python3 -c "import sys, json; print(f\"Iteration {json.load(sys.stdin)['iteration']}\")"
```

## Expected Behavior

### Early Iterations (1-100)
- Most changes REVERTED (baseline is hard to beat)
- Occasional KEEPs when a change improves candidate quality
- Best candidate score gradually improving
- No TAKE yet (gate is stringent)

### Mid Iterations (100-1000)
- More KEEPs as state improves
- Best candidate score approaching 75% threshold
- May see first TAKEs if configurations align
- Quality score increasing

### Late Iterations (1000-10000)
- Refined state producing consistent quality
- TAKEs appearing if high win-prob names exist
- Plateau as optimal configuration is found
- Keep/revert ratio stabilizes

## Differences from Random Search

| Aspect | Random Search | Delta Loop |
|--------|---------------|------------|
| **Strategy** | Try random combinations | Iterative improvement |
| **Memory** | No state carried forward | Builds on best state |
| **Changes** | Full parameter set each time | ONE change per iteration |
| **Learning** | No adaptation | Informed by past results |
| **Changelog** | All equally weighted | Only improvements tracked |
| **Resume** | Restart from last iteration | Resume with accumulated improvements |

## Why This is Better

1. **Cumulative Improvement**: Each kept change builds on the last
2. **Controlled Exploration**: One variable at a time shows clear cause/effect
3. **Adaptive**: Learns from past results to guide proposals
4. **Transparent**: Changelog shows exactly what changed and why
5. **Efficient**: Doesn't re-test known bad configurations
6. **State-of-the-Art**: Industry standard approach for iterative optimization

## Technical Notes

### NYSE SMID Loading

The `nyse_smid_agent.py` had an import error (SECTOR_ETFS from wrong module). This has been fixed:

```python
# Before (broken):
from .sp500 import SECTOR_ETFS, ...

# After (fixed):
from .alpha import SECTOR_ETFS
from .sp500 import get_nyse_smid_constituents_cached, ...
```

NYSE SMID requires FMP_API_KEY in environment. Without it:
- Scanner still works with S&P 500 + Nasdaq-100
- NYSE SMID shows 0 constituents (expected)
- No impact on optimization loop

### Nasdaq-100 Coverage

50 major Nasdaq-100 constituents hardcoded in `delta_optimizer.py`:
- AAPL, MSFT, AMZN, NVDA, GOOGL, META, TSLA, etc.
- Covers ~80% of QQQ weight
- Can be expanded to full 100 if needed

### Benchmarks

SPY and QQQ are in the price universe as references for alpha calculation, **not** as TAKE candidates. ETFs are filtered out from alpha listings.

## Success Criteria

✅ **System Working** if:
- Keep/revert logic functions correctly
- Best candidate score improves over iterations
- Changelog tracks only kept changes
- Universe includes S&P 500 + Nasdaq-100 (+ NYSE SMID if FMP key available)

✅ **High Win-Prob Names Found** if:
- TAKE count > 0 with valid tickers
- TAKEs clear all gate conditions
- Confidence ≥75%, wr ≥62%, n ≥20, alpha ≥3%

✅ **Empty Book is Valid** if:
- No names clear gate
- Book stays empty (correct)
- Scanner not weakened to force listings

## Implementation Files

- `backend/app/services/delta_optimizer.py`: Core delta loop engine
- `backend/run_delta_optimization.py`: CLI script
- `backend/app/services/nyse_smid_agent.py`: Fixed import error
- `delta_optimization_checkpoints/`: Output directory

## Next Steps After 10k Iterations

1. Review changelog for successful modifications
2. Analyze parameter patterns that produced TAKEs
3. Consider manual tuning of ranges based on findings
4. Test best state in live scanning
5. Monitor real-world performance

---

**This is a state-of-the-art iterative improvement system designed to find genuine high win-probability configurations without weakening the desk gate.**
