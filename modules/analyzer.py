# modules/analyzer.py
"""
Traffic analysis module: vehicle counting, density estimation,
congestion classification using a sliding time window.
Criterion 4 + 6: CV Method + Results & Evaluation
"""

import time
import numpy as np
from collections import deque


class TrafficAnalyzer:
    """
    Analyses traffic flow by:
    1. Counting vehicles crossing a virtual line (no duplicates via ID set)
    2. Measuring ROI density (vehicles per frame in region)
    3. Classifying congestion level using a 60-second sliding window
    4. Tracking per-class counts (car, truck, bus, motorcycle)
    """

    # (max_vehicles_per_min, label, BGR_color)
    CONGESTION_LEVELS = [
        (10,   "FREE FLOW",  (0,   210,   0)),
        (25,   "MODERATE",  (0,   210, 255)),
        (40,   "CONGESTED", (0,   140, 255)),
        (9999, "HEAVY JAM", (0,     0, 255)),
    ]

    def __init__(self,
                 line_y: int,
                 fps: float,
                 roi_box: tuple = None,
                 window_sec: int = 60):
        """
        Args:
            line_y      : Y-pixel coordinate of virtual counting line.
            fps         : Video frames per second.
            roi_box     : (x1, y1, x2, y2) region of interest for density.
            window_sec  : Rolling window duration in seconds.
        """
        self.line_y      = line_y
        self.fps         = fps
        self.roi_box     = roi_box
        self.window_sec  = window_sec

        self.counted_ids : set     = set()
        self.total_count : int     = 0
        self.class_counts: dict    = {"car": 0, "motorcycle": 0,
                                      "bus": 0, "truck": 0}
        # Timestamps of each crossing event for sliding window
        self.crossing_times: deque = deque()
        # Per-frame vehicle count in ROI over time (for density chart)
        self.density_history: list = []

    # ------------------------------------------------------------------

    def update(self, tracks: list) -> dict:
        """
        Process one frame of tracks.

        Args:
            tracks: List of track dicts from VehicleTracker.update()

        Returns:
            Frame-level stats dict.
        """
        now = time.time()

        # Purge old timestamps outside the sliding window
        while self.crossing_times and \
              now - self.crossing_times[0] > self.window_sec:
            self.crossing_times.popleft()

        in_roi_count = 0

        for t in tracks:
            tid = t["track_id"]
            cx, cy = t["center"]
            cls = t["class_name"]

            # ── Crossing line counter ──────────────────────────────────
            if cy > self.line_y and tid not in self.counted_ids:
                self.counted_ids.add(tid)
                self.total_count += 1
                self.crossing_times.append(now)
                if cls in self.class_counts:
                    self.class_counts[cls] += 1

            # ── ROI density count ─────────────────────────────────────
            if self.roi_box:
                rx1, ry1, rx2, ry2 = self.roi_box
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    in_roi_count += 1

        self.density_history.append(in_roi_count)

        return {
            "total_count" : self.total_count,
            "roi_density" : in_roi_count,
            "class_counts": dict(self.class_counts),
            "flow_rate"   : self.get_flow_rate(),
            "congestion"  : self.get_congestion_label(),
        }

    # ------------------------------------------------------------------

    def get_flow_rate(self) -> float:
        """Vehicles per minute in the current sliding window."""
        elapsed = min(self.window_sec,
                      time.time() - (self.crossing_times[0]
                                     if self.crossing_times else time.time()))
        if elapsed == 0:
            return 0.0
        return round(len(self.crossing_times) / (elapsed / 60), 1)

    def get_congestion_label(self) -> tuple:
        """Returns (label_str, BGR_color) based on current flow rate."""
        rate = self.get_flow_rate()
        for threshold, label, color in self.CONGESTION_LEVELS:
            if rate <= threshold:
                return label, color
        return "HEAVY JAM", (0, 0, 255)

    def draw_ui(self, frame: np.ndarray,
                tracks: list,
                show_line: bool = True,
                show_roi:  bool = True) -> np.ndarray:
        """
        Draws counting line, ROI box, and HUD stats on frame.
        """
        import cv2
        h, w = frame.shape[:2]

        # ── Counting line ──────────────────────────────────────────────
        if show_line:
            cv2.line(frame, (0, self.line_y), (w, self.line_y),
                     (0, 255, 255), 2)
            cv2.putText(frame, "COUNT LINE", (10, self.line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ── ROI box ────────────────────────────────────────────────────
        if show_roi and self.roi_box:
            rx1, ry1, rx2, ry2 = self.roi_box
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)

        # ── HUD Panel ─────────────────────────────────────────────────
        label, color = self.get_congestion_label()
        hud_lines = [
            f"Total: {self.total_count}",
            f"Flow : {self.get_flow_rate():.1f} veh/min",
            f"State: {label}",
        ]
        panel_x, panel_y = 10, 10
        for i, line in enumerate(hud_lines):
            y = panel_y + i * 22
            cv2.putText(frame, line, (panel_x, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def reset(self):
        """Reset all counters (for multi-segment processing)."""
        self.counted_ids.clear()
        self.total_count = 0
        self.class_counts = {k: 0 for k in self.class_counts}
        self.crossing_times.clear()
        self.density_history.clear()
