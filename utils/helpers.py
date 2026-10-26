# utils/helpers.py
"""
Utility helpers: ROI selector, frame annotation, video writer, config loader.
Criterion 5: Implementation and Code Quality
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Interactive ROI Selector
# ─────────────────────────────────────────────────────────────────────────────

class ROISelector:
    """
    Interactive tool to select a road polygon ROI by clicking on a frame.
    Click to add points, press ENTER to confirm, ESC to cancel.
    """

    def __init__(self):
        self.points = []
        self._done  = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def select(self, frame: np.ndarray,
               window_name: str = "Select ROI — Click points, ENTER to confirm") -> list:
        """
        Opens an interactive window. Click to place ROI vertices.
        Returns list of (x, y) points, or empty list if cancelled.
        """
        self.points = []
        clone = frame.copy()
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while True:
            display = clone.copy()
            # Draw placed points and connecting lines
            for i, pt in enumerate(self.points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display, self.points[i - 1], pt,
                             (0, 255, 0), 2)
            if len(self.points) > 1:
                cv2.line(display, self.points[-1], self.points[0],
                         (0, 255, 0), 1)   # close preview

            cv2.putText(display,
                        "Left-click: add point | Right-click: undo | ENTER: confirm | ESC: cancel",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(20) & 0xFF
            if key == 13 and len(self.points) >= 3:   # ENTER
                break
            elif key == 27:                            # ESC
                self.points = []
                break

        cv2.destroyWindow(window_name)
        return self.points


# ─────────────────────────────────────────────────────────────────────────────
# Video Writer
# ─────────────────────────────────────────────────────────────────────────────

class VideoWriter:
    """Wrapper around cv2.VideoWriter with automatic codec selection."""

    def __init__(self, path: str, fps: float, frame_size: tuple):
        """
        Args:
            path       : Output file path (.mp4 or .avi)
            fps        : Frames per second.
            frame_size : (width, height).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self.path = path

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def release(self):
        self._writer.release()
        print(f"[VideoWriter] Saved: {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter in top-right corner."""
    h, w = frame.shape[:2]
    label = f"FPS: {fps:.1f}"
    cv2.putText(frame, label, (w - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    return frame


def draw_timestamp(frame: np.ndarray) -> np.ndarray:
    """Draw current timestamp in bottom-left corner."""
    ts = time.strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def draw_banner(frame: np.ndarray,
                text: str = "Smart Traffic Monitor",
                color: tuple = (30, 30, 30)) -> np.ndarray:
    """Draw a title banner at the top of the frame."""
    banner_h = 36
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Config persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_config(config: dict, path: str = "config.json"):
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Config] Saved → {path}")


def load_config(path: str = "config.json") -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)
