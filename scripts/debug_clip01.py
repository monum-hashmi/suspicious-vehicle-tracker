import csv
import json
import os
import cv2

from src.detection.detector import Detector
from src.tracking.tracker import Tracker
from src.tracking.roi import ROI
from src.rule_engine.pair_matcher import match_pairs
from src.rule_engine.rule_engine import RuleEngine

VIDEO_PATH  = "clips/clip_01_atm_loitering.mp4"
ROI_POLYGON = [[200, 80], [500, 80], [500, 240], [200, 240]]
OUTPUT_CSV  = "results/clip01_debug.csv"

def main():
    os.makedirs("results", exist_ok=True)

    roi         = ROI(ROI_POLYGON)
    cap         = cv2.VideoCapture(VIDEO_PATH)
    detector    = Detector(conf_threshold=0.25)
    tracker     = Tracker(max_age=300)
    rule_engine = RuleEngine(use_dwell=True, use_motion=True)

    fps       = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0
    rows      = []

    print(f"Running debug on {VIDEO_PATH}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp    = frame_idx / fps
        frame_idx   += 1

        detections   = detector.detect(frame)
        tracks       = tracker.update(detections, frame)
        pairs        = match_pairs(tracks)
        tracks_by_id = {t["track_id"]: t for t in tracks}

        for pair in pairs:
            person_id  = pair["person_id"]
            vehicle_id = pair["vehicle_id"]
            centroid   = pair["pair_centroid"]
            key        = (person_id, vehicle_id)

            in_roi     = roi.is_inside(centroid)

            # Get dwell time from rule engine internals
            dwell_time = 0
            if key in rule_engine.roi_entry_time:
                dwell_time = timestamp - rule_engine.roi_entry_time[key]

            # Get pair history length
            history_len = len(rule_engine.pair_history.get(key, []))

            rows.append({
                "frame":        frame_idx,
                "timestamp":    round(timestamp, 2),
                "person_id":    person_id,
                "vehicle_id":   vehicle_id,
                "centroid_x":   round(centroid[0], 1),
                "centroid_y":   round(centroid[1], 1),
                "in_roi":       in_roi,
                "dwell_time":   round(dwell_time, 2),
                "history_len":  history_len,
            })

        # Also update rule engine so dwell timers accumulate
        rule_engine.update(pairs, tracks_by_id, roi, timestamp=timestamp)

    cap.release()

    # Save CSV
    if rows:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Debug CSV saved to {OUTPUT_CSV} ({len(rows)} rows)")
    else:
        print("No pairs detected at all — check detector and pair_matcher.")

if __name__ == "__main__":
    main()