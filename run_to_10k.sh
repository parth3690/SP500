#!/bin/bash
#
# Script to run the full 10,000 iteration optimization continuously.
# This script can be run in a screen/tmux session or as a background job.
#
# Usage:
#   ./run_to_10k.sh
#
# Or to run in background and log to file:
#   nohup ./run_to_10k.sh > optimization_full_run.log 2>&1 &
#

set -e  # Exit on error

WORKSPACE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$WORKSPACE_DIR"

echo "=========================================="
echo "Starting 10,000 Iteration Optimization"
echo "=========================================="
echo "Start time: $(date)"
echo "Working directory: $WORKSPACE_DIR"
echo ""

# Run optimization with resume enabled
python3 backend/run_optimization.py --iterations 10000 --resume

echo ""
echo "=========================================="
echo "Optimization Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - optimization_checkpoints/checkpoint.jsonl"
echo "  - optimization_checkpoints/best_configs.json"
echo ""
