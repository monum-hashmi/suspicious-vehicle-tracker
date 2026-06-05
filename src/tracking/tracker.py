from deep_sort_realtime.deepsort_tracker import DeepSort
import time

class Tracker:
    def __init__(self, max_age=30):
        """
        Wraps deep-sort-realtime.
        max_age: how many frames to keep a track alive without a detection match.
        """
        self.tracker = DeepSort(max_age=max_age)
        self.track_histories = {}  # track_id -> list of (timestamp, bbox, centroid)

    def update(self, detections, frame):
        """
        detections: list of dicts from Monum's detector:
            [{"bbox": [x,y,w,h], "class": "person|bicycle|motorcycle", "confidence": float}, ...]
        frame: the current video frame (numpy array)
        Returns: list of active tracks with stable IDs
        """
        # Convert Monum's detection format to deep-sort-realtime format
        # deep-sort expects: ([x,y,w,h], confidence, class_label)
        ds_detections = []
        for det in detections:
            bbox = det["bbox"]   # already xywh format
            conf = det["confidence"]
            cls  = det["class"]
            ds_detections.append((bbox, conf, cls))

        # Run DeepSORT
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        timestamp = time.time()
        active_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]

            # Convert to xywh
            x = ltrb[0]
            y = ltrb[1]
            w = ltrb[2] - ltrb[0]
            h = ltrb[3] - ltrb[1]
            bbox = [x, y, w, h]

            # Centroid
            centroid = (x + w / 2, y + h / 2)

            # Update history (keep last 60 seconds)
            if track_id not in self.track_histories:
                self.track_histories[track_id] = []

            self.track_histories[track_id].append((timestamp, bbox, centroid))

            # Prune entries older than 60 seconds
            self.track_histories[track_id] = [
                entry for entry in self.track_histories[track_id]
                if timestamp - entry[0] <= 60
            ]

            active_tracks.append({
                "track_id": track_id,
                "bbox": bbox,
                "centroid": centroid,
                "class": track.det_class,
                "history": self.track_histories[track_id]
            })

        return active_tracks