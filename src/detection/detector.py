"""
YOLOv8-based detector for the Suspicious Vehicle & Person Tracker.

Wraps Ultralytics YOLOv8 and exposes a clean .detect() interface that returns
only the classes we care about (person, bicycle, motorcycle) in a stable format
that downstream modules (tracker, rule engine) can consume.
"""

from pathlib import Path
from typing import List, Dict, Union
import numpy as np
from ultralytics import YOLO


# COCO class IDs we care about. YOLOv8 pretrained weights are trained on COCO.
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    3: "motorcycle",
}


class Detector:
    """Lightweight wrapper around YOLOv8 for our 3-class detection task."""

    def __init__(
        self,
        weights: str = "yolov8s.pt",
        conf_threshold: float = 0.4,
        device: str = "cpu",
    ):
        """
        Args:
            weights: Path to YOLOv8 weights file. Default 'yolov8s.pt' auto-downloads.
            conf_threshold: Minimum confidence score to keep a detection. Default 0.4.
            device: 'cpu' or 'cuda'. Use 'cuda' on Kaggle, 'cpu' on local laptop.
        """
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image array as loaded by cv2.imread (H, W, 3).

        Returns:
            List of detection dicts. Each dict has:
                - "bbox": [x, y, w, h] in pixel coordinates (top-left + width/height)
                - "class": class name string ("person", "bicycle", "motorcycle")
                - "confidence": float in [0, 1]
        """
        # Run YOLOv8 inference. verbose=False suppresses per-frame logging.
        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                if cls_id not in TARGET_CLASSES:
                    continue

                # YOLO returns xyxy; we convert to xywh (top-left + width/height)
                # to match MOT format and OpenCV conventions downstream.
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1

                detections.append({
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "class": TARGET_CLASSES[cls_id],
                    "confidence": float(box.conf),
                })

        return detections


if __name__ == "__main__":
    # Quick self-test: run on the COCO sample bus image.
    import cv2

    print("Initializing detector with yolov8s.pt (will download if needed)...")
    detector = Detector(weights="yolov8s.pt", conf_threshold=0.4, device="cpu")
    print("Detector ready.\n")

    # Use the bus.jpg we already downloaded into _scratch/
    img_path = Path("_scratch/bus.jpg")
    if not img_path.exists():
        # Fall back to downloading it
        import urllib.request
        print("Downloading bus.jpg sample...")
        urllib.request.urlretrieve(
            "https://ultralytics.com/images/bus.jpg",
            "_scratch/bus.jpg",
        )

    frame = cv2.imread(str(img_path))
    print(f"Frame shape: {frame.shape}")

    detections = detector.detect(frame)
    print(f"\nDetected {len(detections)} target-class objects:")
    for d in detections:
        x, y, w, h = d["bbox"]
        print(f"  - {d['class']:12s} conf={d['confidence']:.2f}  bbox=({x:.0f},{y:.0f},{w:.0f},{h:.0f})")