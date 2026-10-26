# main.py
"""
Smart Traffic Monitoring System — Main Pipeline Entry Point
===========================================================
Integrates all modules into a single processing loop.
Run: python main.py --source data/videos/sample.mp4

Criterion 5: Implementation and Code Quality (6 marks)
Criterion 8: Live Demonstration (3 marks)
"""

import cv2
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# ── Module imports ───────────────────────────────────────────────────────────
from modules.preprocessor     import Preprocessor
from modules.detector         import VehicleDetector
from modules.tracker          import VehicleTracker
from modules.analyzer         import TrafficAnalyzer
from modules.speed_estimator  import SpeedEstimator
from modules.heatmap          import TrafficHeatmap
from modules.incident_detector import IncidentDetector
from modules.alerts           import AlertManager
from database.traffic_db      import TrafficDB
from evaluation.metrics       import FPSCounter
from utils.helpers            import (draw_fps, draw_timestamp,
                                      draw_banner, VideoWriter)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Smart Traffic Monitoring System"
    )
    ap.add_argument("--source",    default="0",
                    help="Video path or camera index (default: 0 = webcam)")
    ap.add_argument("--model",     default="models/yolov8n.pt",
                    help="YOLOv8 model weights")
    ap.add_argument("--conf",      type=float, default=0.45,
                    help="Detection confidence threshold")
    ap.add_argument("--line-y",    type=int,   default=None,
                    help="Y-coordinate of counting line (auto = 60%% of height)")
    ap.add_argument("--save",      action="store_true",
                    help="Save annotated output video")
    ap.add_argument("--heatmap",   action="store_true",
                    help="Show density heatmap overlay")
    ap.add_argument("--no-display",action="store_true",
                    help="Run headless (no OpenCV window)")
    ap.add_argument("--skip",      type=int, default=1,
                    help="Process every N-th frame (1=all, 2=every other)")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Open video source ─────────────────────────────────────────────────
    if str(args.source).isdigit():
        source = int(args.source)
        # Use DirectShow on Windows to avoid MSMF capture lock issues
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        source = args.source
        cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        sys.exit(1)

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y    = args.line_y or int(height * 0.60)

    print(f"[INFO] Source  : {args.source}")
    print(f"[INFO] Res     : {width}×{height} @ {fps:.1f} FPS")
    print(f"[INFO] Line Y  : {line_y}")

    # ── Initialise modules ────────────────────────────────────────────────
    preprocessor = Preprocessor(target_size=(width, height))
    detector     = VehicleDetector(model_path=args.model, conf_threshold=args.conf)
    tracker      = VehicleTracker(max_age=30)
    analyzer     = TrafficAnalyzer(line_y=line_y, fps=fps,
                                   roi_box=(0, int(height*0.3),
                                            width, height))
    speed_cfg    = SpeedEstimator.default_config(width, height)
    speed_est    = SpeedEstimator(fps=fps, **speed_cfg)
    heatmap_gen  = TrafficHeatmap(frame_shape=(height, width, 3))
    incident_det = IncidentDetector()
    alert_mgr    = AlertManager()
    db           = TrafficDB()
    fps_counter  = FPSCounter(window=30)

    # ── Output video writer ───────────────────────────────────────────────
    writer = None
    if args.save:
        out_path = f"data/output/annotated_{Path(str(source)).stem}.mp4"
        writer   = VideoWriter(out_path, fps, (width, height))

    # ── Log interval (save to DB every N frames) ──────────────────────────
    LOG_EVERY_N  = int(fps)   # once per second
    frame_idx    = 0

    print("[INFO] Pipeline running. Press 'q' to quit, 'h' to toggle heatmap.\n")
    show_heatmap = args.heatmap

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        frame_idx += 1

        # Skip frames for performance
        if frame_idx % args.skip != 0:
            continue

        fps_counter.tick()

        # ── 1. Preprocess ──────────────────────────────────────────────
        proc_frame = preprocessor.process(raw_frame)

        # ── 2. Detect ─────────────────────────────────────────────────
        detections  = detector.detect(proc_frame)
        ds_input    = detector.get_deepsort_input(detections)

        # ── 3. Track ──────────────────────────────────────────────────
        tracks      = tracker.update(ds_input, proc_frame)

        # ── 4. Speed estimation ────────────────────────────────────────
        speeds      = speed_est.estimate_all(tracks)
        avg_speed   = speed_est.get_average_speed(speeds)

        # ── 5. Traffic analysis ────────────────────────────────────────
        stats       = analyzer.update(tracks)
        cong_label, cong_color = analyzer.get_congestion_label()

        # ── 6. Incident detection ──────────────────────────────────────
        incidents   = incident_det.update(tracks, speeds)

        # ── 7. Alerts ─────────────────────────────────────────────────
        c_alerts    = alert_mgr.check_congestion(
                          cong_label, stats["flow_rate"])
        i_alerts    = alert_mgr.check_incidents(incidents)
        for a in c_alerts + i_alerts:
            print(a)

        # ── 8. Heatmap update ─────────────────────────────────────────
        heatmap_gen.update(tracks)

        # ── 9. Annotate frame ─────────────────────────────────────────
        display = raw_frame.copy()

        if show_heatmap:
            display = heatmap_gen.get_overlay(display, alpha=0.4)

        display = tracker.draw_tracks(display, tracks, draw_trails=True)
        display = speed_est.draw_speeds(display, tracks, speeds)
        display = incident_det.draw_incidents(display, tracks)
        display = analyzer.draw_ui(display, tracks)
        display = draw_fps(display, fps_counter.fps())
        display = draw_timestamp(display)
        display = draw_banner(display,
                              f"Smart Traffic Monitor  |  {cong_label}",
                              color=(20, 20, 20))

        # Congestion color strip on right edge
        cv2.rectangle(display, (width - 12, 36),
                      (width, height), cong_color, -1)

        # Speed HUD
        cv2.putText(display, f"Avg Speed: {avg_speed} km/h",
                    (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        # ── 10. Save output ────────────────────────────────────────────
        if writer:
            writer.write(display)

        # ── 11. Database logging (every second) ───────────────────────
        if frame_idx % LOG_EVERY_N == 0:
            cls_counts = stats["class_counts"]
            db.log_frame({
                "vehicle_count" : stats["total_count"],
                "flow_rate"     : stats["flow_rate"],
                "avg_speed"     : avg_speed,
                "congestion"    : cong_label,
                "roi_density"   : stats["roi_density"],
                "cars"          : cls_counts.get("car", 0),
                "trucks"        : cls_counts.get("truck", 0),
                "buses"         : cls_counts.get("bus", 0),
                "motorcycles"   : cls_counts.get("motorcycle", 0),
                "incident_count": len(incidents),
            })
            for tid, itype in incidents.items():
                trk = next((t for t in tracks if t["track_id"] == tid), None)
                loc = trk["center"] if trk else (0, 0)
                db.log_incident(tid, itype, loc)

        # ── 12. Display ────────────────────────────────────────────────
        if not args.no_display:
            cv2.imshow("Smart Traffic Monitor", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                show_heatmap = not show_heatmap
                print(f"[INFO] Heatmap {'ON' if show_heatmap else 'OFF'}")

    # ── Summary & Cleanup ──────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    summary = db.get_summary() if hasattr(db, "get_summary") else {}
    print("\n[DONE] Session summary:")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    db.close()


if __name__ == "__main__":
    main()
