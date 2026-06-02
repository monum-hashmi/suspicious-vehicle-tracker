"""
Evaluation harness for the Suspicious Vehicle & Person Tracker.

Compares Areeba's rule engine alert events against ground truth from
clips_manifest.csv and produces a precision/recall/F1 table for the
ablation study.

Usage:
    # Record a single configuration's results
    python scripts/evaluate.py --alerts results/alerts_dwell_only.json --config dwell_only

    # After all configurations recorded, print the ablation table
    python scripts/evaluate.py --report

Inputs:
    --alerts: JSON file produced by Areeba's rule engine for one config.
              Format: [{"clip_id": "clip_01", "alert_time": 87.5, "details": {...}}, ...]
              A clip is considered "alerted" if at least one alert appears for it.

    --config: name of the configuration (dwell_only, motion_only, dwell_plus_motion).
              Stored alongside the metrics for later comparison.

Outputs:
    results/metrics/ablation_results.json — accumulated metrics for all configs
    Printed table when --report is passed
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


REPO_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = REPO_ROOT / "docs" / "clips_manifest.csv"
RESULTS_PATH = REPO_ROOT / "results" / "metrics" / "ablation_results.json"


def load_ground_truth() -> Dict[str, bool]:
    """
    Read clips_manifest.csv and return {clip_id: expected_alert_bool}.
    expected_alert column 'yes' -> True, 'no' -> False.
    """
    if not MANIFEST_PATH.exists():
        sys.exit(f"ERROR: Manifest not found at {MANIFEST_PATH}")

    truth = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row["clip_id"].strip()
            expected = row["expected_alert"].strip().lower()
            if expected not in ("yes", "no"):
                sys.exit(f"ERROR: Invalid expected_alert '{expected}' for {clip_id}. Use 'yes' or 'no'.")
            truth[clip_id] = (expected == "yes")
    return truth


def load_alerts(alerts_path: Path) -> Set[str]:
    """
    Read the alerts JSON and return the set of clip_ids that produced at least one alert.
    """
    if not alerts_path.exists():
        sys.exit(f"ERROR: Alerts file not found at {alerts_path}")

    with open(alerts_path, "r", encoding="utf-8") as f:
        try:
            alerts = json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"ERROR: Invalid JSON in {alerts_path}: {e}")

    if not isinstance(alerts, list):
        sys.exit(f"ERROR: Alerts file must contain a JSON array, got {type(alerts).__name__}")

    alerted = set()
    for entry in alerts:
        if not isinstance(entry, dict) or "clip_id" not in entry:
            sys.exit(f"ERROR: Each alert must be an object with 'clip_id'. Got: {entry}")
        alerted.add(entry["clip_id"].strip())
    return alerted


def compute_metrics(truth: Dict[str, bool], alerted: Set[str]) -> Dict:
    """
    Compare ground truth against alerted clips.

    Returns dict with: tp, fp, fn, tn, precision, recall, f1, per_clip results.
    """
    tp = fp = fn = tn = 0
    per_clip = {}

    for clip_id, expected in truth.items():
        fired = clip_id in alerted
        if expected and fired:
            outcome = "TP"; tp += 1
        elif not expected and not fired:
            outcome = "TN"; tn += 1
        elif not expected and fired:
            outcome = "FP"; fp += 1
        else:  # expected and not fired
            outcome = "FN"; fn += 1
        per_clip[clip_id] = {
            "expected_alert": expected,
            "fired": fired,
            "outcome": outcome,
        }

    # Guard against division by zero on tiny datasets
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_clip": per_clip,
    }


def load_results() -> Dict:
    """Load accumulated results, or return empty dict if none yet."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: Dict):
    """Write accumulated results to disk."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def print_report(results: Dict):
    """Pretty-print the ablation comparison table."""
    if not results:
        sys.exit("ERROR: No results recorded yet. Run with --alerts and --config first.")

    print()
    print(f"{'Configuration':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Precision':>11} {'Recall':>8} {'F1':>6}")
    print("-" * 80)
    for config_name, metrics in results.items():
        print(f"{config_name:<25} "
              f"{metrics['tp']:>4} "
              f"{metrics['fp']:>4} "
              f"{metrics['fn']:>4} "
              f"{metrics['tn']:>4} "
              f"{metrics['precision']:>11.3f} "
              f"{metrics['recall']:>8.3f} "
              f"{metrics['f1']:>6.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation for the rule engine.")
    parser.add_argument("--alerts", type=Path, help="Path to alerts JSON file")
    parser.add_argument("--config", type=str, help="Configuration name (e.g. dwell_only)")
    parser.add_argument("--report", action="store_true", help="Print accumulated ablation table")
    parser.add_argument("--reset", action="store_true", help="Clear all recorded results")
    args = parser.parse_args()

    if args.reset:
        if RESULTS_PATH.exists():
            RESULTS_PATH.unlink()
            print(f"Cleared {RESULTS_PATH}")
        else:
            print("No results to clear.")
        return

    if args.report:
        results = load_results()
        print_report(results)
        return

    if not args.alerts or not args.config:
        parser.error("Must provide both --alerts and --config (or use --report / --reset).")

    truth = load_ground_truth()
    alerted = load_alerts(args.alerts)

    # Warn if alerts reference clips not in the manifest
    unknown = alerted - set(truth.keys())
    if unknown:
        print(f"WARNING: Alerts reference clip_ids not in manifest: {unknown}", file=sys.stderr)

    metrics = compute_metrics(truth, alerted)

    # Save into accumulated results
    results = load_results()
    results[args.config] = metrics
    save_results(results)

    # Print this run's summary
    print(f"\nConfiguration: {args.config}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"\nPer-clip results:")
    for clip_id, c in metrics["per_clip"].items():
        marker = "✓" if c["outcome"] in ("TP", "TN") else "✗"
        print(f"  {marker} {clip_id:<10} expected={'yes' if c['expected_alert'] else 'no':<3}  "
              f"fired={'yes' if c['fired'] else 'no':<3}  -> {c['outcome']}")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()