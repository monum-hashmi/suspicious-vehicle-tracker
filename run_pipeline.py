import argparse
import json
import cv2
import time

from src.detection.detector import Detector
from src.tracking.tracker import Tracker
from src.tracking.roi import ROI, DEFAULT_ROI
from src.rule_engine.pair_matcher import match_pairs
from src.rule_engine.rule_engine import RuleEngine


def load_roi(roi_path):
    """Load ROI polygon from a JSON file."""
    with open(roi_path, "r") as f:
        data = json.load(f)
    return ROI(data["polygon"])


def run(video_path, roi, use_dwell=True, use_motion=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return

    detector    = Detector(conf_threshold=0.25)
    tracker     = Tracker(max_age=30)
    rule_engine = RuleEngine(use_dwell=use_dwell, use_motion=use_motion)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0

    print(f"Running pipeline on: {video_path}")
    print(f"Config — dwell: {use_dwell}, motion: {use_motion}")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame_idx += 1

        # Step 1 — Detect
        detections = detector.detect(frame)

        # Step 2 — Track
        tracks = tracker.update(detections, frame)

        # Step 3 — Match pairs
        pairs = match_pairs(tracks)

        # Step 4 — Rule engine
        tracks_by_id = {t["track_id"]: t for t in tracks}
        alerts = rule_engine.update(pairs, tracks_by_id, roi, timestamp=timestamp)

        # Print alerts as JSON
        for alert in alerts:
            print(json.dumps(alert))

    cap.release()
    print("-" * 50)
    print("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--roi",   default=None,  help="Path to ROI JSON file")
    parser.add_argument("--dwell_only",  action="store_true")
    parser.add_argument("--motion_only", action="store_true")
    args = parser.parse_args()

    if args.roi:
        roi = load_roi(args.roi)
    else:
        roi = DEFAULT_ROI

    use_dwell  = not args.motion_only
    use_motion = not args.dwell_only

    run(args.video, roi, use_dwell=use_dwell, use_motion=use_motion)