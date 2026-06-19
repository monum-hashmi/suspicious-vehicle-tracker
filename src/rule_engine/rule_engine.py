import time

class RuleEngine:
    def __init__(self, dwell_threshold=45, motion_threshold=0.3,
                 motion_window=10, cooldown=30,
                 use_dwell=True, use_motion=True):
        """
        dwell_threshold: seconds a pair must be in ROI to trigger Condition A
        motion_threshold: fraction of vehicle width — if pair moved less than this, Condition B triggers
        motion_window: seconds to look back for motion stagnation check
        cooldown: seconds before the same pair can trigger another alert
        use_dwell: enable/disable Condition A (for ablation)
        use_motion: enable/disable Condition B (for ablation)
        """
        self.dwell_threshold  = dwell_threshold
        self.motion_threshold = motion_threshold
        self.motion_window    = motion_window
        self.cooldown         = cooldown
        self.use_dwell        = use_dwell
        self.use_motion       = use_motion

        # pair_key -> timestamp of last alert
        self.last_alert_time  = {}
        # pair_key -> timestamp when pair first entered ROI
        self.roi_entry_time   = {}
        # pair_key -> list of (timestamp, centroid) for motion tracking
        self.pair_history     = {}

    def _pair_key(self, person_id, vehicle_id):
        return (person_id, vehicle_id)

    def update(self, pairs, tracks_by_id, roi, timestamp=None):
        """
        pairs: output of match_pairs()
        tracks_by_id: dict of track_id -> track dict
        roi: ROI instance
        timestamp: current time (defaults to time.time())

        Returns: list of alert dicts
        """
        if timestamp is None:
            timestamp = time.time()

        alerts = []

        for pair in pairs:
            person_id  = pair["person_id"]
            vehicle_id = pair["vehicle_id"]
            centroid   = pair["pair_centroid"]
            key = self._pair_key(person_id, vehicle_id)

            # Check if pair is inside ROI
            in_roi = roi.is_inside(centroid)

            if not in_roi:
                # Reset dwell timer if pair leaves ROI
                self.roi_entry_time.pop(key, None)
                self.pair_history.pop(key, None)
                continue

            # Track when pair entered ROI
            if key not in self.roi_entry_time:
                self.roi_entry_time[key] = timestamp

            # Update pair centroid history
            if key not in self.pair_history:
                self.pair_history[key] = []
            self.pair_history[key].append((timestamp, centroid))

            # Prune history older than motion_window
            self.pair_history[key] = [
                e for e in self.pair_history[key]
                if timestamp - e[0] <= self.motion_window
            ]

            # --- Condition A: Dwell ---
            dwell_time = timestamp - self.roi_entry_time[key]
            condition_a = dwell_time >= self.dwell_threshold

            # --- Condition B: Motion stagnation ---
            condition_b = False
            if vehicle_id in tracks_by_id:
                vehicle_track = tracks_by_id.get(vehicle_id)
                if vehicle_track:
                    vehicle_width = vehicle_track["bbox"][2]
                    threshold_px  = self.motion_threshold * vehicle_width

                    history = self.pair_history[key]
                    if len(history) >= 2:
                        oldest_centroid = history[0][1]
                        newest_centroid = history[-1][1]
                        dx = newest_centroid[0] - oldest_centroid[0]
                        dy = newest_centroid[1] - oldest_centroid[1]
                        distance_moved = (dx**2 + dy**2) ** 0.5
                        condition_b = distance_moved < threshold_px

            # --- Evaluate rule based on ablation config ---
            if self.use_dwell and self.use_motion:
                triggered = condition_a and condition_b
            elif self.use_dwell:
                triggered = condition_a
            elif self.use_motion:
                triggered = condition_b
            else:
                triggered = False

            if not triggered:
                continue

            # --- Cooldown check ---
            last = self.last_alert_time.get(key, 0)
            if timestamp - last < self.cooldown:
                continue

            # --- Emit alert ---
            self.last_alert_time[key] = timestamp

            # Get snapshot bbox (person bbox)
            person_track = tracks_by_id.get(person_id)
            snapshot_bbox = person_track["bbox"] if person_track else None

            alerts.append({
                "timestamp":    timestamp,
                "person_id":    person_id,
                "vehicle_id":   vehicle_id,
                "roi_id":       0,
                "snapshot_bbox": snapshot_bbox
            })

        return alerts