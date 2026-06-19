import time
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=300):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=1,
            max_cosine_distance=0.7,
            nn_budget=None,
            embedder="mobilenet",
            half=False,
            bgr=True,
        )
        self.track_histories = {}

    def update(self, detections, frame):
        ds_detections = []
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            cls  = det["class"]
            ds_detections.append((bbox, conf, cls))

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        timestamp = time.time()
        active_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            x = ltrb[0]
            y = ltrb[1]
            w = ltrb[2] - ltrb[0]
            h = ltrb[3] - ltrb[1]
            bbox = [x, y, w, h]
            centroid = (x + w / 2, y + h / 2)

            if track_id not in self.track_histories:
                self.track_histories[track_id] = []

            self.track_histories[track_id].append((timestamp, bbox, centroid))
            self.track_histories[track_id] = [
                e for e in self.track_histories[track_id]
                if timestamp - e[0] <= 60
            ]

            active_tracks.append({
                "track_id": track_id,
                "bbox": bbox,
                "centroid": centroid,
                "class": track.det_class,
                "history": self.track_histories[track_id]
            })

        return active_tracks