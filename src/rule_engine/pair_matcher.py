import math

def get_distance(point1, point2):
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def match_pairs(tracks):
    """
    For each person track, find the nearest vehicle track if the distance
    between their centroids is less than 1.5x the vehicle's width.

    tracks: list of track dicts from Tracker.update()
            each dict has: track_id, bbox, centroid, class, history

    Returns: list of dicts:
        {
            "person_id": int,
            "vehicle_id": int,
            "pair_centroid": (x, y)
        }
    """
    persons  = [t for t in tracks if t["class"] == "person"]
    vehicles = [t for t in tracks if t["class"] in ("bicycle", "motorcycle")]

    pairs = []

    for person in persons:
        best_vehicle = None
        best_distance = float("inf")

        for vehicle in vehicles:
            vehicle_width = vehicle["bbox"][2]  # w in xywh
            threshold = 1.5 * vehicle_width

            dist = get_distance(person["centroid"], vehicle["centroid"])

            if dist < threshold and dist < best_distance:
                best_distance = dist
                best_vehicle = vehicle

        if best_vehicle is not None:
            # Pair centroid = midpoint between person and vehicle centroids
            px, py = person["centroid"]
            vx, vy = best_vehicle["centroid"]
            pair_centroid = ((px + vx) / 2, (py + vy) / 2)

            pairs.append({
                "person_id": person["track_id"],
                "vehicle_id": best_vehicle["track_id"],
                "pair_centroid": pair_centroid
            })

    return pairs