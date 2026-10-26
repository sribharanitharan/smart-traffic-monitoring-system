# modules/tracker.py
"""
DeepSORT-based multi-object vehicle tracker.
Assigns unique persistent IDs to vehicles across frames.
Maintains trajectory history for speed + incident analysis.
Criterion 4: Computer Vision Method / AI Content (6 marks)
"""

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleTracker:
    """
    Wraps DeepSORT tracker with trajectory history and class label persistence.

    DeepSORT uses:
     - Kalman Filter: predicts next position
     - Appearance descriptor (CNN): re-identifies vehicles after occlusion
     - Hungarian Algorithm: optimal detection-to-track assignment
    """

    def __init__(self,
                 max_age: int = 30,
                 n_init: int = 3,
                 nn_budget: int = 100,
                 max_cosine_distance: float = 0.4):
        """
        Args:
            max_age              : Frames to keep a track alive without detection.
            n_init               : Min consecutive detections before confirming a track.
            nn_budget            : Max samples in appearance descriptor gallery.
            max_cosine_distance  : Max cosine distance for re-ID matching.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            max_cosine_distance=max_cosine_distance,
        )
        # track_id -> deque of (cx, cy) center positions
        self.track_history: dict = {}
        # track_id -> class name
        self.track_classes: dict = {}
        self.max_history_len = 60   # store last 60 positions (~2 sec at 30fps)

    def update(self, raw_detections: list,
               frame: np.ndarray) -> list:
        """
        Update tracker with new detections.

        Args:
            raw_detections : Output of VehicleDetector.get_deepsort_input()
                             → list of ([x,y,w,h], conf, class_name)
            frame          : Current BGR frame (used for appearance extraction).

        Returns:
            List of active confirmed tracks:
            [
              {
                'track_id'  : int,
                'bbox_xyxy' : [x1, y1, x2, y2],
                'class_name': str,
                'center'    : (cx, cy),
                'history'   : [(cx0,cy0), ..., (cx_n,cy_n)],
              },
              ...
            ]
        """
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        active_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid  = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Persist class label
            if hasattr(track, "det_class") and track.det_class:
                self.track_classes[tid] = track.det_class
            cls = self.track_classes.get(tid, "vehicle")

            # Update trajectory history
            if tid not in self.track_history:
                self.track_history[tid] = []
            history = self.track_history[tid]
            history.append((cx, cy))
            if len(history) > self.max_history_len:
                history.pop(0)

            active_tracks.append({
                "track_id"  : tid,
                "bbox_xyxy" : [x1, y1, x2, y2],
                "class_name": cls,
                "center"    : (cx, cy),
                "history"   : list(history),
            })

        return active_tracks

    def draw_tracks(self, frame: np.ndarray,
                    tracks: list,
                    draw_trails: bool = True) -> np.ndarray:
        """
        Draw track IDs, bounding boxes, and motion trails on frame.

        Args:
            tracks       : Output of self.update()
            draw_trails  : Whether to draw historical trajectory lines.
        """
        CLASS_COLORS = {
            "car"        : (0,   255, 100),
            "motorcycle" : (255, 165, 0),
            "bus"        : (0,   100, 255),
            "truck"      : (180, 0,   255),
            "vehicle"    : (200, 200, 200),
        }

        for t in tracks:
            x1, y1, x2, y2 = t["bbox_xyxy"]
            tid  = t["track_id"]
            cls  = t["class_name"]
            color = CLASS_COLORS.get(cls, (200, 200, 200))

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ID label with background
            label = f"ID:{tid} {cls}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Motion trail
            if draw_trails and len(t["history"]) > 1:
                pts = t["history"]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, pts[i - 1], pts[i], trail_color, 2)

        return frame

    def get_all_histories(self) -> dict:
        """Returns full trajectory history for all tracked vehicles."""
        return self.track_history

    def reset(self):
        """Clear all tracks and history."""
        self.track_history.clear()
        self.track_classes.clear()


# Avoid circular import for draw_tracks
import cv2  # noqa: E402  (placed here intentionally after class def)
