# suspicious-vehicle-tracker
Real-time multi-class detection and behavioral alert system for flagged zones (DL course project)
# 🛡️  Suspicious Vehicle & Person Tracker

> Automatically detects suspicious person-vehicle loitering in surveillance footage using YOLOv8 + DeepSORT + a custom two-signal rule engine, with a Streamlit dashboard for forensic alert review.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-purple) ![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-green) ![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## What It Does

Sentinel processes CCTV footage and flags events where a **person and vehicle linger together in a defined zone for over 45 seconds without significant movement** — a behavioral signature consistent with loitering, vehicle tampering, or ATM surveillance.

Each flagged event is stored with:
- Exact timestamp
- Person and vehicle track IDs
- A real cropped frame snapshot
- Full metadata exportable as JSON

---

## Sreenshot
<img width="1171" height="722" alt="Screenshot 2026-06-21 at 6 23 35 PM" src="https://github.com/user-attachments/assets/e9fc24d8-9d13-4950-a16b-c670dfe0033c" />



## Demo



https://github.com/user-attachments/assets/594a809f-6050-4151-a1a9-62f06d4d5fa2







---

## Pipeline

```
Video → YOLOv8 Detection → DeepSORT Tracking → Pair Matching → Rule Engine → Alert + Snapshot
                                                                      ↓
                                                               SQLite Database
                                                                      ↓
                                                          Streamlit Dashboard
```

**Two-signal rule** — both conditions must be true to fire an alert:
- **Dwell:** person-vehicle pair in ROI for > 45 seconds
- **Motion stagnation:** centroid moved < 0.3× vehicle width in last 10 seconds

---

## Results

| Clip | Scene | Alerts | Outcome |
|------|-------|--------|---------|
| clip_01_atm_loitering.mp4 | ATM loitering at night | 3 | ✅ Correctly detected |
| clip_02_normal_traffic.mp4 | Normal road traffic | 0 | ✅ No false positives |
| clip_03_busy_road.mp4 | Busy road scene | 0 | ✅ No false positives |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (ultralytics) |
| Multi-Object Tracking | DeepSORT (deep-sort-realtime) |
| Rule Engine | Custom Python (dwell + motion stagnation) |
| Dashboard | Streamlit |
| Storage | SQLite3 |
| Language | Python 3.12 |

---

## Quickstart

```bash
# 1. Clone and set up
git clone <repo-url>
cd suspicious-vehicle-tracker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the pipeline on a video
caffeinate -i python run_pipeline.py --video clips/clip_01_atm_loitering.mp4 2>&1 | tee results/clip01_output.txt

# 3. Load results into the database
python scripts/ingest_pipeline_output.py results/clip01_output.txt

# 4. Launch the dashboard
streamlit run src/dashboard/app.py
```

>  CPU-only inference takes 15–25 min per clip. Use `ps aux | grep run_pipeline` to verify it's running if the terminal looks frozen — output is buffered until completion.

---

## Project Structure

```
suspicious-vehicle-tracker/
├── clips/                        # Input video clips
├── results/
│   ├── alerts.db                 # SQLite database
│   └── snapshots/                # Cropped frame evidence per alert
├── scripts/
│   ├── ingest_pipeline_output.py # Load pipeline results into DB
│   ├── debug_clip01.py           # Tracker diagnosis script
│   └── evaluate.py               # Evaluation utilities
├── src/
│   ├── detection/detector.py     # YOLOv8 wrapper
│   ├── tracking/tracker.py       # DeepSORT wrapper
│   ├── tracking/roi.py           # ROI polygon logic
│   ├── rule_engine/pair_matcher.py
│   ├── rule_engine/rule_engine.py
│   └── dashboard/
│       ├── app.py                # Streamlit dashboard
│       └── db.py                 # SQLite schema + queries
├── run_pipeline.py               # Main pipeline entrypoint
└── tests/test_db.py
```

---

## Dashboard Features

-  Upload a clip or choose from library
-  Live alerts panel with severity filter and search
-  Event timeline with clickable markers
-  Alert investigation panel with real cropped snapshot
-  ROI definition tool — draw directly on the first frame
-  Export alert evidence as JSON

---

## Team

| Member | Role |
|--------|------|
| Monum | Detection, datasets, demo clips, architecture |
| Areeba | Tracking, ROI, rule engine, ablation |
| Hafsa | Dashboard, storage, pipeline integration, paper |

**UMT — Deep Learning Course Project**

---

## Limitations

- Batch processing only — ~15–25 min per clip on CPU; GPU required for real-time use
- Fixed-camera assumption — tracker degrades with PTZ cameras or very dense crowds
- Thresholds tuned for ATM/parking lot scenarios
