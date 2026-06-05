import cv2
import numpy as np

class ROI:
    def __init__(self, polygon):
        """
        polygon: list of (x, y) points defining the ROI boundary.
        Example: [(100, 200), (300, 200), (300, 400), (100, 400)]
        """
        self.polygon = np.array(polygon, dtype=np.float32)

    def is_inside(self, point):
        """
        point: (x, y) tuple — typically a track centroid
        Returns True if point is inside the ROI polygon.
        """
        result = cv2.pointPolygonTest(self.polygon, point, False)
        return result >= 0  # 0 = on boundary, positive = inside

    def draw(self, frame, color=(0, 255, 0), thickness=2):
        """
        Draws the ROI polygon on a frame for visualization.
        """
        pts = self.polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
        return frame


# Hardcoded test ROI for development (Hafsa will replace with drawn UI)
DEFAULT_ROI = ROI([
    (100, 150),
    (500, 150),
    (500, 350),
    (100, 350)
])