import os
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import confusion_matrix  # Optionally for further analysis

# ----- Import configuration paths -----
from config import (
    GT_JSONS_DIR, GT_MASKS_DIR,
    YOLO_PREDICTED_JSONS_DIR, YOLO_PREDICTED_MASKS_DIR,
    MASKRCNN_PREDICTED_JSONS_DIR, MASKRCNN_PREDICTED_MASKS_DIR,
    YOLO_DEEPLAB_PREDICTED_JSONS_DIR, YOLO_DEEPLAB_PREDICTED_MASKS_DIR,
    IOU_THRESHOLD, CONFIDENCE_THRESHOLD, MAX_INSTANCES
)

# ----- Import loader functions for GT and predictions -----
from data_loader import load_gt_json, load_gt_masks, load_predicted_json, load_predicted_masks


# ----------------------
# Helper Metric Functions
# ----------------------
def compute_pixel_iou(gt_mask, pred_mask):
    """Computes pixel-wise IoU between two masks."""
    gt_bin = (gt_mask > 0).astype(np.uint8)
    pred_bin = (pred_mask > 0).astype(np.uint8)
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum() + 1e-6
    return intersection / union


def compute_dice(gt_mask, pred_mask):
    """Computes the Dice coefficient between two masks."""
    gt_bin = (gt_mask > 0).astype(np.uint8)
    pred_bin = (pred_mask > 0).astype(np.uint8)
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    dice = (2 * intersection) / (gt_bin.sum() + pred_bin.sum() + 1e-6)
    return dice


def compute_iou(boxA, boxB):
    """Computes IoU between two bounding boxes, each defined as [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def compute_detection_metrics(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_threshold=0.5):
    """
    Computes detection metrics (precision, recall, F1, label accuracy) based on IoU matching.
    Additionally, it computes the average IoU of all matched boxes.

    Note: This function uses all predictions (i.e., raw metrics), ignoring confidence scores.
    """
    TP = 0
    FP = 0
    FN = 0
    label_correct = 0
    matched_gt = set()
    matched_ious = []  # To store IoU for each matched pair.

    for p_box, p_label in zip(pred_boxes, pred_labels):
        best_iou = 0
        best_idx = -1
        for idx, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in matched_gt:
                continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(best_idx)
            matched_ious.append(best_iou)
            if p_label == gt_labels[best_idx]:
                label_correct += 1
        else:
            FP += 1
    FN = len(gt_boxes) - len(matched_gt)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    label_accuracy = label_correct / (TP + 1e-6)
    mean_box_iou = np.mean(matched_ious) if matched_ious else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_accuracy": label_accuracy,
        "mean_box_iou": mean_box_iou,
        "TP": TP,
        "FP": FP,
        "FN": FN
    }


def compute_detection_metrics_per_class(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_threshold=0.5):
    """
    Computes precision for each class by grouping detections based on the predicted class.
    Returns a dictionary mapping class label to precision.
    """
    classes = set(gt_labels).union(set(pred_labels))
    class_precisions = {}
    for cls in classes:
        # Get GT and prediction boxes for the class.
        gt_cls_boxes = [box for box, label in zip(gt_boxes, gt_labels) if label == cls]
        pred_cls_boxes = [box for box, label in zip(pred_boxes, pred_labels) if label == cls]
        TP = 0
        FP = 0
        matched = set()
        for p_box in pred_cls_boxes:
            best_iou = 0
            best_idx = -1
            for idx, g_box in enumerate(gt_cls_boxes):
                if idx in matched:
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold:
                TP += 1
                matched.add(best_idx)
            else:
                FP += 1
        precision = TP / (TP + FP + 1e-6)
        class_precisions[cls] = precision
    return class_precisions


def compute_map_from_class_precisions(class_precisions):
    """
    Computes mAP as the unweighted mean of per-class precision values.
    (This is a simplified approximation, not the full AP integration over all thresholds.)
    """
    if len(class_precisions) == 0:
        return 0.0
    return np.mean(list(class_precisions.values()))


def aggregate_metric(values):
    """
    Aggregates a list of metric values across frames.
    Returns a dict with mean, std, and variance.
    """
    if len(values) == 0:
        return {"mean": None, "std": None, "variance": None}
    mean_val = np.mean(values)
    std_val = np.std(values)
    var_val = np.var(values)
    return {"mean": float(mean_val), "std": float(std_val), "variance": float(var_val)}


# ----------------------
# Evaluation per Sequence
# ----------------------
def evaluate_sequence_metrics(sequence_name, model_name, det_iou_threshold=IOU_THRESHOLD):
    """
    Evaluates a given sequence for a specified model by computing:
      - Pixel-wise IoU and Dice for segmentation masks.
      - Detection metrics for bounding boxes and labels (including mean box IoU).
      - Per-class precision and a computed mAP (as an unweighted mean of per-class precision).

    Note: For detection metrics, this function uses all predictions (raw metrics without filtering by score).
    Also computes temporal consistency (mean, std, variance) of these metrics across frames.

    Returns:
        dict: Aggregated metrics for the sequence.
    """
    # Load GT annotations and masks.
    gt_ann = load_gt_json(sequence_name)
    gt_masks = load_gt_masks(sequence_name)

    # Load predicted annotations and masks for the model.
    pred_ann = load_predicted_json(model_name, sequence_name)
    pred_masks = load_predicted_masks(model_name, sequence_name)

    # Lists to store per-frame segmentation metrics.
    pixel_iou_vals = []
    dice_vals = []
    # Lists for detection metrics.
    precision_vals = []
    recall_vals = []
    f1_vals = []
    label_acc_vals = []
    box_iou_vals = []  # To store average IoU for matched boxes per frame.

    # Also accumulate all boxes and labels for per-class evaluation.
    all_gt_boxes = []
    all_gt_labels = []
    all_pred_boxes = []
    all_pred_labels = []

    for frame, gt_data in gt_ann.items():
        # Segmentation metrics.
        if frame in gt_masks and frame in pred_masks:
            gt_mask = gt_masks[frame]
            pred_mask = pred_masks[frame]
            pixel_iou = compute_pixel_iou(gt_mask, pred_mask)
            dice_score = compute_dice(gt_mask, pred_mask)
            pixel_iou_vals.append(pixel_iou)
            dice_vals.append(dice_score)

        # Detection metrics.
        gt_boxes = gt_data.get("boxes", [])
        gt_labels = gt_data.get("labels", [])
        all_gt_boxes.extend(gt_boxes)
        all_gt_labels.extend(gt_labels)
        if frame in pred_ann:
            pred_frame = pred_ann[frame]
            pred_boxes = pred_frame.get("boxes", [])
            pred_labels = pred_frame.get("labels", [])
            all_pred_boxes.extend(pred_boxes)
            all_pred_labels.extend(pred_labels)
            det_metrics = compute_detection_metrics(gt_boxes, gt_labels, pred_boxes, pred_labels,
                                                    iou_threshold=det_iou_threshold)
            precision_vals.append(det_metrics["precision"])
            recall_vals.append(det_metrics["recall"])
            f1_vals.append(det_metrics["f1"])
            label_acc_vals.append(det_metrics["label_accuracy"])
            box_iou_vals.append(det_metrics["mean_box_iou"])

    aggregated_seg = {
        "pixel_iou": aggregate_metric(pixel_iou_vals),
        "dice": aggregate_metric(dice_vals)
    }
    aggregated_det = {
        "precision": aggregate_metric(precision_vals),
        "recall": aggregate_metric(recall_vals),
        "f1": aggregate_metric(f1_vals),
        "label_accuracy": aggregate_metric(label_acc_vals),
        "box_iou": aggregate_metric(box_iou_vals)
    }

    # Compute per-class detection precision and mAP for the entire sequence.
    class_precisions = compute_detection_metrics_per_class(all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_labels,
                                                           iou_threshold=det_iou_threshold)
    mAP = compute_map_from_class_precisions(class_precisions)
    aggregated_det["per_class_precision"] = class_precisions
    aggregated_det["mAP"] = mAP

    return {
        "segmentation": aggregated_seg,
        "detection": aggregated_det
    }


# ----------------------
# Evaluation per Model
# ----------------------
def evaluate_model_metrics(model_name, det_iou_threshold=IOU_THRESHOLD):
    """
    Evaluates all sequences for a given model.

    Returns:
        dict: Mapping from sequence name to aggregated metrics.
    """
    # Get list of sequences from GT JSON files.
    gt_json_dir = Path(GT_JSONS_DIR)
    seq_files = list(gt_json_dir.glob("*_gt.json"))
    sequence_names = [f.stem.replace("_gt", "") for f in seq_files]

    model_metrics = {}
    for seq in sequence_names:
        print(f"Evaluating sequence '{seq}' for model '{model_name}'...")
        seq_metrics = evaluate_sequence_metrics(seq, model_name, det_iou_threshold)
        model_metrics[seq] = seq_metrics
    return model_metrics


def save_model_metrics(metrics, model_name):
    """
    Saves the aggregated metrics for a model into a JSON file in the 'model_metrics' directory.
    """
    output_dir = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics for model '{model_name}' to {output_file}")


# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    models = ["yolo", "maskrcnn", "yolo_deeplab"]
    # We now compute raw metrics (without filtering predictions by score).
    for model in models:
        print(f"Evaluating metrics for model: {model}")
        model_metrics = evaluate_model_metrics(model, det_iou_threshold=IOU_THRESHOLD)
        save_model_metrics(model_metrics, model)