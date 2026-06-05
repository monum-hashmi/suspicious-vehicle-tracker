# Handoff Notes

## Week 1 → Week 2 (Monum → Areeba)
*(To be filled by Monum)*

---

## Week 2 → Week 3 (Areeba → Hafsa)

### What was built in Week 2
- `src/tracking/tracker.py` — DeepSORT wrapper, consumes detector output, emits stable track IDs with 60s history
- `src/tracking/roi.py` — ROI polygon module using cv2.pointPolygonTest
- `src/rule_engine/pair_matcher.py` — Matches person tracks to nearest vehicle within 1.5× vehicle width
- `src/rule_engine/rule_engine.py` — Two-signal rule engine (dwell + motion stagnation) with ablation support
- `run_pipeline.py` — End-to-end CLI script
- `scripts/run_ablation.py` — Runs all 3 ablation configs on all clips

### How to run the pipeline
1. Activate venv:
   `.\venv\Scripts\Activate.ps1`
2. Set PYTHONPATH:
   `$env:PYTHONPATH="."`
3. Run on a video:
   `python run_pipeline.py --video clips/clip_01_atm_loitering.mp4`
4. Run with custom ROI:
   `python run_pipeline.py --video clips/clip_01_atm_loitering.mp4 --roi roi.json`

### ROI JSON format
{"polygon": [[100,150],[500,150],[500,350],[100,350]]}

### Alert output format
{"timestamp": 47.2, "person_id": 3, "vehicle_id": 7, "roi_id": 0, "snapshot_bbox": [368,107,43,101]}

### Ablation results
- Location: `results/ablation_results.csv`
- Three configs: `dwell_only`, `motion_only`, `dwell_motion`
- Columns: config, precision, recall, fp_rate, tp, fp, fn, tn

### Notes for Hafsa
- Replace DEFAULT_ROI in `src/tracking/roi.py` with the ROI drawn by user in the dashboard
- Pass drawn ROI as JSON file using `--roi` flag or instantiate `ROI(polygon)` directly
- Areeba is on call for fixes — message if anything doesn't run cleanly