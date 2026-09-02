#!/usr/bin/env python3
"""
Analyze optimization checkpoint results and generate summary statistics.

Usage:
    python3 analyze_results.py [checkpoint_file]
    
Default checkpoint file: ./optimization_checkpoints/checkpoint.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_checkpoints(checkpoint_file: Path) -> list[dict[str, Any]]:
    """Load all checkpoint records from JSONL file."""
    records = []
    with open(checkpoint_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def analyze_checkpoints(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze checkpoint records and compute statistics."""
    if not records:
        return {"error": "No checkpoint records found"}
    
    # Basic stats
    total_iterations = len(records)
    last_iteration = records[-1]["iteration"]
    
    # TAKE/PASS counts
    take_counts = [r["results"]["take_count"] for r in records]
    pass_counts = [r["results"]["pass_count"] for r in records]
    
    total_takes = sum(take_counts)
    total_passes = sum(pass_counts)
    iterations_with_takes = sum(1 for t in take_counts if t > 0)
    
    # Fitness stats
    fitness_scores = [r["results"]["fitness"] for r in records]
    best_fitness = max(fitness_scores)
    avg_fitness = sum(fitness_scores) / len(fitness_scores)
    
    # Best candidate scores
    best_candidate_scores = [r["results"]["best_candidate_score"] for r in records]
    highest_candidate_score = max(best_candidate_scores)
    
    # Parameter distribution
    risk_modes = [r["params"]["risk_mode"] for r in records]
    risk_mode_dist = Counter(risk_modes)
    
    horizons = [r["params"]["forward_horizon"] for r in records]
    horizon_dist = Counter(horizons)
    
    step_sizes = [r["params"]["step_size"] for r in records]
    step_size_dist = Counter(step_sizes)
    
    # Find best iteration
    best_idx = fitness_scores.index(best_fitness)
    best_record = records[best_idx]
    
    return {
        "summary": {
            "total_iterations": total_iterations,
            "last_iteration": last_iteration,
            "total_takes": total_takes,
            "total_passes": total_passes,
            "iterations_with_takes": iterations_with_takes,
            "best_fitness": round(best_fitness, 2),
            "avg_fitness": round(avg_fitness, 2),
            "highest_candidate_score": round(highest_candidate_score, 2),
        },
        "best_iteration": {
            "iteration": best_record["iteration"],
            "fitness": round(best_record["results"]["fitness"], 2),
            "take_count": best_record["results"]["take_count"],
            "params": best_record["params"],
            "timestamp": best_record["timestamp"],
        },
        "parameter_distributions": {
            "risk_mode": dict(risk_mode_dist),
            "forward_horizon": dict(horizon_dist),
            "step_size": dict(step_size_dist),
        },
    }


def print_analysis(analysis: dict[str, Any]):
    """Print analysis results in a readable format."""
    print("\n" + "="*60)
    print("OPTIMIZATION ANALYSIS")
    print("="*60)
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    summary = analysis["summary"]
    print(f"\nTotal Iterations: {summary['total_iterations']}")
    print(f"Last Iteration: {summary['last_iteration']}")
    print(f"\nTAKE/PASS Statistics:")
    print(f"  Total TAKE decisions: {summary['total_takes']}")
    print(f"  Total PASS decisions: {summary['total_passes']}")
    print(f"  Iterations with TAKEs: {summary['iterations_with_takes']}")
    
    print(f"\nFitness Statistics:")
    print(f"  Best fitness: {summary['best_fitness']}")
    print(f"  Average fitness: {summary['avg_fitness']}")
    print(f"  Highest candidate score: {summary['highest_candidate_score']}%")
    
    print(f"\nBest Configuration:")
    best = analysis["best_iteration"]
    print(f"  Iteration: {best['iteration']}")
    print(f"  Fitness: {best['fitness']}")
    print(f"  TAKE count: {best['take_count']}")
    print(f"  Parameters:")
    for key, value in best["params"].items():
        print(f"    {key}: {value}")
    
    print(f"\nParameter Distributions:")
    for param_name, dist in analysis["parameter_distributions"].items():
        print(f"  {param_name}:")
        for value, count in sorted(dist.items()):
            pct = count / summary["total_iterations"] * 100
            print(f"    {value}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*60)


def main():
    # Get checkpoint file from command line or use default
    if len(sys.argv) > 1:
        checkpoint_file = Path(sys.argv[1])
    else:
        checkpoint_file = Path("./optimization_checkpoints/checkpoint.jsonl")
    
    if not checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_file}")
        print(f"\nUsage: {sys.argv[0]} [checkpoint_file]")
        sys.exit(1)
    
    print(f"Loading checkpoints from: {checkpoint_file}")
    records = load_checkpoints(checkpoint_file)
    
    print(f"Loaded {len(records)} checkpoint records")
    
    analysis = analyze_checkpoints(records)
    print_analysis(analysis)
    
    # Save analysis to JSON
    output_file = checkpoint_file.parent / "analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {output_file}")


if __name__ == "__main__":
    main()
