import csv
import json
import os
import cv2

from src.detection.detector import Detector
from src.tracking.tracker import Tracker
from src.tracking.roi import DEFAULT_ROI
from src.rule_engine.pair_matcher import match_pairs
from src.rule_engine.rule_engine import RuleEngine

MANIFEST_PATH = "docs/clips_manifest.csv"
RESULTS_PATH  = "results/ablation_results.csv"

CONFIGS = [
    {"name": "dwell_only",   "use_dwell": True,  "use_motion": False},
    {"name": "motion_only",  "use_dwell": False, "use_motion": True},
    {"name": "dwell_motion", "use_dwell": True,  "use_motion": True},
]

def run_clip(video_path, use_dwell, use_motion):
    """Run pipeline on one clip, return list of alert dicts."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return []

    detector    = Detector(conf_threshold=0.25)
    tracker     = Tracker(max_age=30)
    rule_engine = RuleEngine(use_dwell=use_dwell, use_motion=use_motion)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0
    alerts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame_idx += 1

        detections   = detector.detect(frame)
        tracks       = tracker.update(detections, frame)
        pairs        = match_pairs(tracks)
        tracks_by_id = {t["track_id"]: t for t in tracks}
        new_alerts   = rule_engine.update(pairs, tracks_by_id, DEFAULT_ROI, timestamp=timestamp)
        alerts.extend(new_alerts)

    cap.release()
    return alerts


def evaluate(alerts, expected_alert):
    """
    Simple precision/recall/FP-rate for one clip.
    expected_alert: "yes" or "no"
    """
    fired = len(alerts) > 0
    tp = 1 if (fired and expected_alert == "yes") else 0
    fp = 1 if (fired and expected_alert == "no")  else 0
    fn = 1 if (not fired and expected_alert == "yes") else 0
    tn = 1 if (not fired and expected_alert == "no")  else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_alerts": len(alerts)}


def main():
    # Load manifest
    clips = []
    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append(row)

    os.makedirs("results", exist_ok=True)

    rows = []

    for config in CONFIGS:
        print(f"\nConfig: {config['name']}")
        tp_total = fp_total = fn_total = tn_total = 0

        for clip in clips:
            video_path = f"clips/{clip['filename']}"
            expected   = clip["expected_alert"]
            print(f"  Processing {clip['clip_id']} (expected={expected})...")

            alerts  = run_clip(video_path, config["use_dwell"], config["use_motion"])
            metrics = evaluate(alerts, expected)

            tp_total += metrics["tp"]
            fp_total += metrics["fp"]
            fn_total += metrics["fn"]
            tn_total += metrics["tn"]

            print(f"    alerts={metrics['n_alerts']} tp={metrics['tp']} fp={metrics['fp']}")

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        fp_rate   = fp_total / (fp_total + tn_total) if (fp_total + tn_total) > 0 else 0

        rows.append({
            "config":    config["name"],
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "fp_rate":   round(fp_rate, 3),
            "tp":        tp_total,
            "fp":        fp_total,
            "fn":        fn_total,
            "tn":        tn_total,
        })

        print(f"  => precision={precision:.3f} recall={recall:.3f} fp_rate={fp_rate:.3f}")

    # Save results
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAblation results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()