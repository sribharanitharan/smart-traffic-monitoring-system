# modules/detector.py
"""
YOLOv8-based vehicle detection module.
Detects: car, motorcycle, bus, truck from COCO pretrained weights.
Criterion 4: Computer Vision Method / AI Content (6 marks)
"""

import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """
    Vehicle detection using YOLOv8.
    Filters COCO classes to vehicle-only detections.
    Supports confidence thresholding and NMS IoU control.
    """

    # COCO dataset class IDs for vehicles
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    # Color palette per vehicle class (BGR)
    CLASS_COLORS = {
        "car"        : (0,   255, 100),
        "motorcycle" : (255, 165, 0),
        "bus"        : (0,   100, 255),
        "truck"      : (180, 0,   255),
    }

    def __init__(self,
                 model_path: str = "models/yolov8n.pt",
                 conf_threshold: float = 0.45,
                 iou_threshold:  float = 0.50,
                 device: str = "cpu"):
        """
        Args:
            model_path      : Path to YOLOv8 weights (.pt file).
            conf_threshold  : Minimum confidence to accept a detection.
            iou_threshold   : NMS IoU threshold (higher = allow more overlap).
            device          : 'cpu' or 'cuda' or '0' for GPU index.
        """
        self.model     = YOLO(model_path)
        self.conf      = conf_threshold
        self.iou       = iou_threshold
        self.device    = device
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> list:
        """
        Run YOLOv8 inference on a single frame.

        Returns:
            List of dicts:
                {
                  'bbox_xywh' : [x, y, w, h],      ← top-left + size
                  'bbox_xyxy' : [x1, y1, x2, y2],  ← corners
                  'confidence': float,
                  'class_id'  : int,
                  'class_name': str,
                  'color'     : (B, G, R),
                }
        """
        self._frame_count += 1
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                conf = round(float(box.conf[0]), 3)
                cls_name = self.VEHICLE_CLASSES[cls_id]

                detections.append({
                    "bbox_xywh"  : [x1, y1, w, h],
                    "bbox_xyxy"  : [x1, y1, x2, y2],
                    "confidence" : conf,
                    "class_id"   : cls_id,
                    "class_name" : cls_name,
                    "color"      : self.CLASS_COLORS.get(cls_name, (255, 255, 255)),
                })

        return detections

    def draw_detections(self, frame: np.ndarray,
                        detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on frame (raw detections only)."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            color = det["color"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    def get_deepsort_input(self, detections: list) -> list:
        """
        Converts detections to DeepSORT input format.
        Returns: list of ([x, y, w, h], confidence, class_name)
        """
        return [
            (d["bbox_xywh"], d["confidence"], d["class_name"])
            for d in detections
        ]

    @property
    def frame_count(self) -> int:
        return self._frame_count
