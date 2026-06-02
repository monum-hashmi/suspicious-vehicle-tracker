# Contributing

Team conventions for the Suspicious Vehicle & Person Tracker project.

## Workflow

1. Pull `main` before starting any session
2. Create your own branch — never commit directly to `main`
   - `monum/<feature>`, `areeba/<feature>`, `hafsa/<feature>`
3. Commit often, push your branch
4. Open a PR when ready to merge

## Commit messages

Describe the change, not when. Examples:

- ✅ `Add DeepSORT wrapper with 60-second track history`
- ❌ `Day 3 work` / `Updated stuff` / `tracker stuff`

## Where things live

| What | Where |
|---|---|
| Python code | `src/` |
| Test/sanity scripts | `scripts/` |
| Documentation | `docs/`, root `.md` files |
| Architecture diagram and paper figures | `results/figures/` |
| Numerical results (JSONs) | `results/metrics/` |
| **Video clips (NEVER in git)** | Shared Google Drive folder |
| Trained model weights | Auto-downloaded locally; gitignored |

## Dataset access

The 3 demo clips live in the shared Google Drive folder. Each team member places the .mp4 files into their local `./clips/` folder (gitignored).

## Detector contract (locked — do not change without team discussion)

```python
from src.detection.detector import Detector

detector = Detector(weights="yolov8s.pt", conf_threshold=0.25, device="cpu")
detections = detector.detect(frame)
# Returns: [{"bbox": [x,y,w,h], "class": "person|bicycle|motorcycle", "confidence": float}, ...]
```

- `conf_threshold=0.25` is calibrated on MOT17-04 (mAP@0.5 = 0.52, FPS = 79). Don't change without re-running evaluation.
- Schema is consumed by `tracker.py` downstream — changes break things.

## Evaluation harness usage

For the ablation study, save alert outputs per configuration then run:

```bash
python scripts/evaluate.py --reset
python scripts/evaluate.py --alerts results/alerts_dwell_only.json --config dwell_only
python scripts/evaluate.py --alerts results/alerts_motion_only.json --config motion_only
python scripts/evaluate.py --alerts results/alerts_dwell_plus_motion.json --config dwell_plus_motion
python scripts/evaluate.py --report
```

Alert JSON format:

```json
[
    {"clip_id": "clip_01", "alert_time": 87.5, "details": {...}},
    ...
]
```

The script reads ground truth from `docs/clips_manifest.csv`.

## Code style

- Python 3.10+
- Type hints on public methods
- Short docstring on each public class/function
- Follow the patterns in `src/detection/detector.py`

## Before opening a PR

- Code runs without errors
- New files don't accidentally include videos (`*.mp4`), weights (`*.pt`), or large generated outputs
