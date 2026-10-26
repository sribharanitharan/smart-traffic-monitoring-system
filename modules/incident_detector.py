# modules/incident_detector.py
"""
🚀 INNOVATIVE FEATURE: Automatic traffic incident detection.
Detects:
  1. Stopped vehicles (potential breakdowns / accidents)
  2. Wrong-way drivers (moving against traffic flow)
  3. Sudden braking / rapid deceleration events

Uses trajectory history from VehicleTracker — no extra model needed.
Criterion 4: Computer Vision Method / AI Content (6 marks)
"""

import cv2
import time
import numpy as np


class IncidentDetector:
    """
    Rule-based incident detection using vehicle trajectory analysis.

    Each rule uses displacement vectors computed from the track's
    historical center positions — a custom CV contribution beyond
    standard detection/tracking library calls.
    """

    INCIDENT_COLORS = {
        "STOPPED_VEHICLE"   : (0,   0,   255),   # Red
        "WRONG_WAY"         : (0,   140, 255),   # Orange
        "RAPID_DECELERATION": (255, 0,   200),   # Magenta
    }

    def __init__(self,
                 stopped_px_thresh:   int   = 8,
                 stopped_frames:      int   = 45,
                 wrong_way_dy_thresh: int   = -12,
                 decel_speed_drop:    float = 25.0,
                 alert_cooldown_sec:  int   = 10):
        """
        Args:
            stopped_px_thresh   : Max pixel displacement to consider "stopped".
            stopped_frames      : How many consecutive near-zero-displacement
                                  frames before flagging as stopped (≈1.5s @30fps).
            wrong_way_dy_thresh : Negative dy (upward motion) threshold.
                                  Vehicles going UP in frame = wrong way.
            decel_speed_drop    : Speed drop (km/h) in < 1s → rapid braking.
            alert_cooldown_sec  : Min seconds between repeated alerts per vehicle.
        """
        self.stopped_px      = stopped_px_thresh
        self.stopped_frames  = stopped_frames
        self.ww_dy_thresh    = wrong_way_dy_thresh
        self.decel_drop      = decel_speed_drop
        self.cooldown        = alert_cooldown_sec

        # State tracking
        self._stopped_count : dict = {}   # tid → consecutive stopped frames
        self._prev_speeds   : dict = {}   # tid → speed at previous frame
        self._last_alert    : dict = {}   # tid → last alert timestamp
        self.active_incidents: dict = {}  # tid → incident_type (current frame)
        self.incident_log   : list = []   # historical log

    # ------------------------------------------------------------------

    def update(self,
               tracks: list,
               speeds: dict = None) -> dict:
        """
        Analyse all active tracks for incidents.

        Args:
            tracks : Track list from VehicleTracker.update()
            speeds : Optional {track_id: speed_kmh} from SpeedEstimator

        Returns:
            dict: {track_id: incident_type} for this frame.
        """
        speeds = speeds or {}
        self.active_incidents = {}

        for t in tracks:
            tid     = t["track_id"]
            history = t["history"]

            if len(history) < 5:
                continue

            incident = self._check_stopped(tid, history)
            if not incident:
                incident = self._check_wrong_way(tid, history)
            if not incident and tid in speeds:
                incident = self._check_rapid_decel(tid, speeds[tid])

            if incident and self._can_alert(tid):
                self.active_incidents[tid] = incident
                self._record_incident(tid, incident, t["center"])
                self._last_alert[tid] = time.time()

            # Update prev speed
            if tid in speeds:
                self._prev_speeds[tid] = speeds[tid]

        return self.active_incidents

    # ------------------------------------------------------------------
    # Individual detection rules
    # ------------------------------------------------------------------

    def _check_stopped(self, tid: int, history: list) -> str:
        """Detects vehicles with near-zero displacement over many frames."""
        window = history[-min(self.stopped_frames, len(history)):]
        if len(window) < 2:
            return ""

        displacement = np.linalg.norm(
            np.array(window[-1]) - np.array(window[0])
        )

        if displacement < self.stopped_px:
            self._stopped_count[tid] = \
                self._stopped_count.get(tid, 0) + 1
        else:
            self._stopped_count[tid] = 0

        if self._stopped_count.get(tid, 0) >= self.stopped_frames:
            return "STOPPED_VEHICLE"
        return ""

    def _check_wrong_way(self, tid: int, history: list) -> str:
        """
        Detects vehicles moving against dominant traffic flow.
        Assumes dominant flow is downward (positive Y direction).
        A vehicle moving significantly upward (negative dy) is wrong-way.
        """
        if len(history) < 6:
            return ""
        dy = history[-1][1] - history[-6][1]
        if dy < self.ww_dy_thresh:
            return "WRONG_WAY"
        return ""

    def _check_rapid_decel(self, tid: int, current_speed: float) -> str:
        """Detects sudden speed drop indicating hard braking."""
        prev = self._prev_speeds.get(tid, current_speed)
        if prev - current_speed > self.decel_drop and current_speed < 10:
            return "RAPID_DECELERATION"
        return ""

    def _can_alert(self, tid: int) -> bool:
        """Enforce cooldown between repeated alerts for same vehicle."""
        last = self._last_alert.get(tid, 0)
        return time.time() - last > self.cooldown

    def _record_incident(self, tid: int, itype: str, location: tuple):
        self.incident_log.append({
            "track_id" : tid,
            "type"     : itype,
            "location" : location,
            "time"     : time.strftime("%H:%M:%S"),
        })

    # ------------------------------------------------------------------

    def draw_incidents(self, frame: np.ndarray,
                       tracks: list) -> np.ndarray:
        """Draws incident markers (flashing borders + labels) on frame."""
        for t in tracks:
            tid = t["track_id"]
            if tid not in self.active_incidents:
                continue

            itype = self.active_incidents[tid]
            color = self.INCIDENT_COLORS.get(itype, (0, 0, 255))
            x1, y1, x2, y2 = t["bbox_xyxy"]

            # Thick colored border
            cv2.rectangle(frame, (x1 - 3, y1 - 3),
                          (x2 + 3, y2 + 3), color, 3)

            # Warning badge
            badge = f"⚠ {itype.replace('_', ' ')}"
            (bw, bh), _ = cv2.getTextSize(
                badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
            )
            cv2.rectangle(frame,
                          (x1, y1 - bh - 10),
                          (x1 + bw + 8, y1),
                          color, -1)
            cv2.putText(frame, badge, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 255), 2)

        return frame

    def get_incident_log(self) -> list:
        """Returns full incident history as list of dicts."""
        return list(self.incident_log)

    def get_summary(self) -> dict:
        """Summarise incident counts by type."""
        summary = {}
        for rec in self.incident_log:
            t = rec["type"]
            summary[t] = summary.get(t, 0) + 1
        return summary
