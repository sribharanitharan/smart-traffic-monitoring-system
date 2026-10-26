# evaluation/metrics.py
"""
Quantitative evaluation metrics for the Smart Traffic Monitoring System.
Criterion 6: Results, Visualization and Evaluation (5 marks)

Metrics:
  - IoU          : Intersection over Union (detection quality)
  - mAP@50       : Mean Average Precision at IoU=0.5
  - Counting Acc : Vehicle count accuracy vs ground truth
  - Speed MAE    : Mean Absolute Error of speed estimation
  - FPS          : Processing throughput
"""

import time
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(box_a: list, box_b: list) -> float:
    """
    Compute Intersection over Union between two [x1,y1,x2,y2] boxes.

    IoU = Area_of_Overlap / Area_of_Union
    Range: [0, 1] — 1.0 = perfect overlap.
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    xi1 = max(xa1, xb1);  yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2);  yi2 = min(ya2, yb2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter   = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# mAP (Mean Average Precision)
# ─────────────────────────────────────────────────────────────────────────────

def compute_average_precision(precisions: list, recalls: list) -> float:
    """
    Compute Average Precision using the area under precision-recall curve
    (11-point interpolation, consistent with PASCAL VOC standard).
    """
    ap = 0.0
    for thresh in np.arange(0, 1.1, 0.1):
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= thresh]
        ap += max(p_at_r) if p_at_r else 0.0
    return ap / 11.0


def compute_map(all_gt: dict, all_preds: dict,
                iou_threshold: float = 0.5) -> dict:
    """
    Compute mAP@iou_threshold across all classes.

    Args:
        all_gt    : {class_name: [list of gt boxes per image]}
        all_preds : {class_name: [(confidence, box), ...]}
        iou_threshold : IoU threshold for TP (default 0.5 = mAP@50)

    Returns:
        dict with per-class AP and overall mAP.
    """
    class_aps = {}

    for cls in all_gt:
        gt_boxes   = all_gt[cls]
        pred_items = sorted(all_preds.get(cls, []),
                            key=lambda x: x[0], reverse=True)
        n_gt = sum(len(g) for g in gt_boxes)
        if n_gt == 0:
            continue

        tp_list, fp_list = [], []
        matched = defaultdict(set)

        for conf, pred_box in pred_items:
            best_iou, best_img, best_idx = 0, -1, -1
            for img_id, gt_list in enumerate(gt_boxes):
                for j, gt_box in enumerate(gt_list):
                    if j in matched[img_id]:
                        continue
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_img, best_idx = iou, img_id, j

            if best_iou >= iou_threshold:
                tp_list.append(1); fp_list.append(0)
                matched[best_img].add(best_idx)
            else:
                tp_list.append(0); fp_list.append(1)

        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        recalls    = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

        class_aps[cls] = round(compute_average_precision(
            list(precisions), list(recalls)), 4)

    m_ap = round(np.mean(list(class_aps.values())), 4) if class_aps else 0.0
    return {"class_ap": class_aps, "mAP": m_ap}


# ─────────────────────────────────────────────────────────────────────────────
# Counting Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def counting_accuracy(pred_count: int, gt_count: int) -> float:
    """
    Vehicle counting accuracy.
    Formula: 1 - |pred - gt| / gt
    Range: [0, 1] — 1.0 = perfect count.
    """
    if gt_count == 0:
        return 1.0 if pred_count == 0 else 0.0
    return max(0.0, round(1.0 - abs(pred_count - gt_count) / gt_count, 4))


# ─────────────────────────────────────────────────────────────────────────────
# Speed Estimation Error
# ─────────────────────────────────────────────────────────────────────────────

def speed_mae(pred_speeds: list, gt_speeds: list) -> float:
    """
    Mean Absolute Error of speed estimates vs ground truth.
    Units: km/h.
    """
    if not pred_speeds or not gt_speeds:
        return 0.0
    n = min(len(pred_speeds), len(gt_speeds))
    return round(float(np.mean(np.abs(
        np.array(pred_speeds[:n]) - np.array(gt_speeds[:n])
    ))), 2)


# ─────────────────────────────────────────────────────────────────────────────
# FPS Tracker
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-window FPS measurement."""

    def __init__(self, window: int = 30):
        self._times = []
        self._window = window

    def tick(self):
        self._times.append(time.time())
        if len(self._times) > self._window:
            self._times.pop(0)

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return round((len(self._times) - 1) /
                     (self._times[-1] - self._times[0]), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Summary Report Printer
# ─────────────────────────────────────────────────────────────────────────────

def print_evaluation_report(results: dict):
    """Pretty-print the evaluation results to console."""
    sep = "─" * 52
    print(f"\n{sep}")
    print("  📊 EVALUATION REPORT — Smart Traffic Monitor")
    print(sep)
    print(f"  mAP@50           : {results.get('mAP', 'N/A')}")
    print(f"  Counting Accuracy: {results.get('counting_acc', 'N/A')}")
    print(f"  Speed MAE        : {results.get('speed_mae', 'N/A')} km/h")
    print(f"  Avg FPS          : {results.get('fps', 'N/A')}")
    print(sep)
    if "class_ap" in results:
        for cls, ap in results["class_ap"].items():
            print(f"    AP [{cls:12s}] : {ap:.4f}")
    print(sep + "\n")
