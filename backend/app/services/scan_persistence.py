"""
Scan run persistence for institutional scanner.

Persists scan runs to JSON files for auditability and reproducibility.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Default persistence directory
PERSISTENCE_DIR = Path(os.getenv("SCAN_PERSISTENCE_DIR", "/tmp/institutional_scans"))


def _ensure_persistence_dir() -> Path:
    """Ensure the persistence directory exists."""
    PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)
    return PERSISTENCE_DIR


def save_scan_run(
    scan_type: str,
    inputs: dict[str, Any],
    results: dict[str, Any],
    *,
    run_id: Optional[str] = None,
) -> str:
    """
    Save a scan run to disk for audit trail.
    
    Args:
        scan_type: Type of scan (e.g., 'institutional_sp500', 'institutional_nyse_smid')
        inputs: Scan inputs (filters, parameters)
        results: Full scan results
        run_id: Optional custom run ID (defaults to timestamp)
    
    Returns:
        Run ID (filename stem)
    """
    persistence_dir = _ensure_persistence_dir()
    
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    
    scan_record = {
        "runId": run_id,
        "scanType": scan_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "results": {
            "asOf": results.get("asOf"),
            "marketRegime": results.get("marketRegime"),
            "candidates": results.get("candidates", []),
            "convexityAlerts": results.get("convexityAlerts", []),
            "meta": results.get("meta"),
        },
    }
    
    filepath = persistence_dir / f"{run_id}.json"
    
    with open(filepath, "w") as f:
        json.dump(scan_record, f, indent=2, default=str)
    
    return run_id


def load_scan_run(run_id: str) -> Optional[dict[str, Any]]:
    """
    Load a previously saved scan run.
    
    Args:
        run_id: Run ID (filename stem)
    
    Returns:
        Scan record or None if not found
    """
    persistence_dir = _ensure_persistence_dir()
    filepath = persistence_dir / f"{run_id}.json"
    
    if not filepath.exists():
        return None
    
    with open(filepath, "r") as f:
        return json.load(f)


def list_scan_runs(
    scan_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    List recent scan runs.
    
    Args:
        scan_type: Optional filter by scan type
        limit: Maximum number of runs to return
    
    Returns:
        List of scan metadata (without full results)
    """
    persistence_dir = _ensure_persistence_dir()
    
    scan_files = sorted(
        persistence_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    
    runs = []
    for filepath in scan_files[:limit * 2]:  # Read extra in case of filtering
        try:
            with open(filepath, "r") as f:
                record = json.load(f)
            
            if scan_type and record.get("scanType") != scan_type:
                continue
            
            runs.append({
                "runId": record["runId"],
                "scanType": record["scanType"],
                "timestamp": record["timestamp"],
                "inputs": record["inputs"],
                "summary": {
                    "takeCount": record["results"]["meta"].get("summary", {}).get("takeCount", 0),
                    "watchTierCount": record["results"]["meta"].get("summary", {}).get("watchTierCount", 0),
                    "totalScanned": record["results"]["meta"].get("summary", {}).get("totalScanned", 0),
                },
            })
            
            if len(runs) >= limit:
                break
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue
    
    return runs


def cleanup_old_runs(days_to_keep: int = 30) -> int:
    """
    Clean up scan runs older than the specified number of days.
    
    Args:
        days_to_keep: Number of days of history to retain
    
    Returns:
        Number of files deleted
    """
    import time
    
    persistence_dir = _ensure_persistence_dir()
    cutoff_time = time.time() - (days_to_keep * 86400)
    
    deleted_count = 0
    for filepath in persistence_dir.glob("*.json"):
        if filepath.stat().st_mtime < cutoff_time:
            filepath.unlink()
            deleted_count += 1
    
    return deleted_count
