# Datasets

## Overview

This project uses three data sources, each with a specific role.

## 1. COCO (via YOLOv8 pretrained weights)

- **Role:** Detection evaluation (person, bicycle, motorcycle classes)
- **Source:** Bundled with YOLOv8 — no separate download needed
- **Used by:** Monum (Day 3 detection eval)

## 2. MOT17 (sequences 04 and 09)

- **Role:** Tracking evaluation — measure how well DeepSORT keeps stable IDs across frames
- **Source:** Kaggle dataset `kaggle.com/datasets/ahmedsamir1598/mot17challenge`
- **Kaggle mount path:** `/kaggle/input/datasets/ahmedsamir1598/mot17challenge/MOT17/`
- **Structure:**
  - `train/` — 21 sequence folders (7 sequences × 3 detector variants: DPM, FRCNN, SDP)
  - `test/` — 21 sequence folders (no ground truth, withheld for benchmarking — not used)
  - Each train sequence contains:
    - `img1/000001.jpg, ...` — frames (e.g. MOT17-04 has 1050 frames)
    - `gt/gt.txt` — ground truth in MOT format: `frame, track_id, x, y, w, h, conf, class, visibility`
    - `det/det.txt` — public detections (not used, we use our own YOLOv8 detections)
    - `seqinfo.ini` — sequence metadata
- **We use:** `train/MOT17-04-FRCNN/` and `train/MOT17-09-FRCNN/` only. FRCNN variant chosen as it's the strongest baseline detector among the three variants; the choice only affects `det/det.txt`, not the ground truth.
- **Used by:** Areeba (Week 2 tracker eval)

## 3. Demo CCTV clips

- **Role:** Real-world person-vehicle scenes for rule engine evaluation
- **Source:** Curated from YouTube CCTV compilations (Day 4)
- **Location:** Shared Google Drive folder `demo_clips/` (link in CONTRIBUTING.md)
- **Manifest:** `demo_clips/clips_manifest.csv` lists each clip with expected alert (yes/no)
- **Used by:** Areeba (ablation runs, Week 2), Hafsa (paper figures, Week 3)

## Why not MOT20?

MOT20 was considered but skipped — it's redundant with MOT17 for our purposes (still person-only annotations, no vehicles) and adds disk space without adding insight. If we need crowded scenes for stress testing later, we'll revisit.

## Why not VisDrone?

The original proposal mentioned VisDrone, but VisDrone is aerial drone footage — wrong perspective for our CCTV use case. Switched to MOT17 (for tracking eval) + COCO (already in YOLOv8 for detection eval) + real CCTV clips (for rule engine eval).

## Important caveat: MOT17 is person-only

MOT17 has no vehicle annotations. We use it only for tracking evaluation (does DeepSORT keep stable IDs across frames?). Vehicle detection accuracy is evaluated separately against COCO (already in YOLOv8 pretrained weights). End-to-end person+vehicle scenes are evaluated on the demo CCTV clips, not on MOT17.