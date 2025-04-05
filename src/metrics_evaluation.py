#!/usr/bin/env python3
import os
import json
import numpy as np
import cv2
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix  # Optional

# ----- Import configuration paths from config -----
from config import (
    REP_BBOX_JSON, GT_JSONS_DIR, GT_MASKS_DIR,
    YOLO_PREDICTED_JSONS_DIR, YOLO_PREDICTED_MASKS_DIR,
    MASKRCNN_PREDICTED_JSONS_DIR, MASKRCNN_PREDICTED_MASKS_DIR,
    YOLO_DEEPLAB_PREDICTED_JSONS_DIR, YOLO_DEEPLAB_PREDICTED_MASKS_DIR,
    IOU_THRESHOLD, CONFIDENCE_THRESHOLD, MAX_INSTANCES, OUTPUT_DIR
)

# ----- Import loader functions for GT and predictions -----
from data_loader import load_gt_json, load_gt_masks, load_predicted_json, load_predicted_masks


# ===================================================
# RAW METRICS COMPUTATION FUNCTIONS (Per-Sequence)
# ===================================================

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
    return (2 * intersection) / (gt_bin.sum() + pred_bin.sum() + 1e-6)


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
    Also computes the average IoU of all matched boxes.
    Note: This function uses all predictions (raw metrics), ignoring confidence scores.
    """
    TP = 0
    FP = 0
    FN = 0
    label_correct = 0
    matched_gt = set()
    matched_ious = []  # List to store IoU for each matched pair.

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
    Computes per-class precision by grouping detections based on the predicted class.
    Returns a dictionary mapping each class label to its precision.
    """
    classes = set(gt_labels).union(set(pred_labels))
    class_precisions = {}
    for cls in classes:
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
    (This is a simplified approximation compared to full AP integration over thresholds.)
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


# ===================================================
# RAW METRICS EVALUATION (Default Thresholds)
# ===================================================

def evaluate_sequence_metrics(sequence_name, model_name, det_iou_threshold=IOU_THRESHOLD):
    """
    Evaluates a given sequence for a specified model using default thresholds.
    Computes segmentation metrics (pixel IoU, Dice), detection metrics (precision, recall, F1, label accuracy, box IoU),
    per-class detection metrics (and mAP), and processing time.
    Returns:
        dict: Aggregated metrics for the sequence.
    """
    start_time = time.time()
    gt_ann = load_gt_json(sequence_name)
    gt_masks = load_gt_masks(sequence_name)
    pred_ann = load_predicted_json(model_name, sequence_name)
    pred_masks = load_predicted_masks(model_name, sequence_name)

    pixel_iou_vals = []
    dice_vals = []
    precision_vals = []
    recall_vals = []
    f1_vals = []
    label_acc_vals = []
    box_iou_vals = []

    all_gt_boxes = []
    all_gt_labels = []
    all_pred_boxes = []
    all_pred_labels = []

    for frame, gt_data in gt_ann.items():
        if frame in gt_masks and frame in pred_masks:
            gt_mask = gt_masks[frame]
            pred_mask = pred_masks[frame]
            pixel_iou_vals.append(compute_pixel_iou(gt_mask, pred_mask))
            dice_vals.append(compute_dice(gt_mask, pred_mask))

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
    class_precisions = compute_detection_metrics_per_class(all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_labels,
                                                           iou_threshold=det_iou_threshold)
    mAP = compute_map_from_class_precisions(class_precisions)
    aggregated_det["per_class_precision"] = class_precisions
    aggregated_det["mAP"] = mAP

    processing_time = time.time() - start_time
    return {
        "segmentation": aggregated_seg,
        "detection": aggregated_det,
        "processing_time": processing_time
    }


def evaluate_model_metrics(model_name, det_iou_threshold=IOU_THRESHOLD):
    """
    Evaluates all sequences for a given model using default thresholds.
    Returns:
        dict: A mapping from sequence names to aggregated metrics, plus overall processing time.
    """
    gt_json_dir = Path(GT_JSONS_DIR)
    seq_files = list(gt_json_dir.glob("*_gt.json"))
    sequence_names = [f.stem.replace("_gt", "") for f in seq_files]

    model_metrics = {}
    processing_times = []
    for seq in sequence_names:
        print(f"Evaluating sequence '{seq}' for model '{model_name}'...")
        seq_metrics = evaluate_sequence_metrics(seq, model_name, det_iou_threshold)
        model_metrics[seq] = seq_metrics
        processing_times.append(seq_metrics.get("processing_time", 0))
    model_metrics["processing_time"] = aggregate_metric(processing_times)
    return model_metrics


def save_model_metrics(metrics, model_name):
    """
    Saves the aggregated raw metrics for a model into a JSON file in the model_metrics directory.
    """
    output_dir = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved raw metrics for model '{model_name}' to {output_file}")


# ===================================================
# AGGREGATION FUNCTIONS (Over Sequence Categories)
# ===================================================

def get_sequence_lists_from_gt():
    """
    Loads the GT bounding boxes and labels JSON and extracts sequence names
    for single_object and multi_object.

    Returns:
        tuple: (single_object_list, multi_object_list)
    """
    gt_json_path = REP_BBOX_JSON
    if not gt_json_path.exists():
        raise FileNotFoundError(f"GT JSON not found: {gt_json_path}")
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)
    single_object_list = list(gt_data.get("single_object", {}).keys())
    multi_object_list = list(gt_data.get("multi_object", {}).keys())
    return single_object_list, multi_object_list


def aggregate_over_sequences(metrics_dict, sequence_list):
    """
    Aggregates each metric (segmentation, detection, processing_time) across the provided sequences.

    Returns:
        dict: Aggregated metrics (mean, std, variance) for each metric.
    """
    seg_vals = {"pixel_iou": [], "dice": []}
    det_vals = {"precision": [], "recall": [], "f1": [], "label_accuracy": [], "box_iou": []}
    mAP_values = []
    processing_times = []

    def extract_mean(seq_metric, keys):
        for key in keys:
            seq_metric = seq_metric.get(key, {})
        return seq_metric.get("mean", None) if isinstance(seq_metric, dict) else seq_metric

    for seq in sequence_list:
        if seq in metrics_dict:
            seq_metrics = metrics_dict[seq]
            for key in seg_vals:
                val = extract_mean(seq_metrics, ["segmentation", key])
                if val is not None:
                    seg_vals[key].append(val)
            for key in det_vals:
                val = extract_mean(seq_metrics, ["detection", key])
                if val is not None:
                    det_vals[key].append(val)
            mAP = seq_metrics.get("detection", {}).get("mAP", None)
            if mAP is not None:
                mAP_values.append(mAP)
            proc_time = seq_metrics.get("processing_time", None)
            if proc_time is not None:
                processing_times.append(proc_time)

    def aggregate_list(values):
        if len(values) == 0:
            return {"mean": None, "std": None, "variance": None}
        return {"mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "variance": float(np.var(values))}

    aggregated = {
        "segmentation": {key: aggregate_list(seg_vals[key]) for key in seg_vals},
        "detection": {key: aggregate_list(det_vals[key]) for key in det_vals},
        "mAP": float(np.mean(mAP_values)) if mAP_values else None,
        "processing_time": aggregate_list(processing_times)
    }
    return aggregated


def aggregate_model_metrics(raw_metrics):
    """
    Aggregates raw per-sequence metrics for a model over:
      - single_object sequences,
      - multi_object sequences,
      - all sequences combined.

    Returns:
        dict: Aggregated metrics with keys "single_object", "multi_object", and "all_objects".
    """
    single_object_list, multi_object_list = get_sequence_lists_from_gt()
    all_sequences = list(raw_metrics.keys())
    agg_single = aggregate_over_sequences(raw_metrics, single_object_list)
    agg_multi = aggregate_over_sequences(raw_metrics, multi_object_list)
    agg_all = aggregate_over_sequences(raw_metrics, all_sequences)
    return {
        "single_object": agg_single,
        "multi_object": agg_multi,
        "all_objects": agg_all
    }

# ===================================================
# THRESHOLDED METRICS AGGREGATION
# ===================================================

def aggregate_thresholded_metrics(thresholded_metrics):
    """
    Aggregates thresholded per-sequence metrics into single_object, multi_object, and all_objects categories.
    """
    single_object_list, multi_object_list = get_sequence_lists_from_gt()
    all_sequences = list(thresholded_metrics.keys())

    agg_single = aggregate_over_sequences(thresholded_metrics, single_object_list)
    agg_multi = aggregate_over_sequences(thresholded_metrics, multi_object_list)
    agg_all = aggregate_over_sequences(thresholded_metrics, all_sequences)

    return {
        "single_object": agg_single,
        "multi_object": agg_multi,
        "all_objects": agg_all
    }


# ===================================================
# CACHING FUNCTIONS
# ===================================================

def get_cached_raw_metrics(model_name):
    """
    Checks if the raw metrics file exists for the given model.
    If yes, loads and returns it; otherwise, computes the raw metrics,
    saves them, and returns the result.
    """
    cache_file = Path(
        "/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics") / model_name / f"{model_name}_metrics.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            print(f"Loading cached raw metrics for {model_name}")
            return json.load(f)
    else:
        print(f"Computing raw metrics for {model_name} (cache not found)")
        raw_metrics = evaluate_model_metrics(model_name, det_iou_threshold=IOU_THRESHOLD)
        save_model_metrics(raw_metrics, model_name)
        return raw_metrics


def get_cached_aggregated_metrics(model_name):
    """
    Checks if the aggregated metrics file exists for the given model.
    If yes, loads and returns it; otherwise, aggregates raw metrics, saves them, and returns the result.
    """
    cache_file = Path(
        "/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics") / model_name / f"{model_name}_aggregated_metrics.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            print(f"Loading cached aggregated metrics for {model_name}")
            return json.load(f)
    else:
        raw_metrics = get_cached_raw_metrics(model_name)
        aggregated_metrics = aggregate_model_metrics(raw_metrics)
        with open(cache_file, "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        print(f"Saved aggregated metrics for model '{model_name}' to {cache_file}")
        return aggregated_metrics


# ===================================================
# THRESHOLDED EVALUATION FUNCTIONALITY
# ===================================================

def evaluate_with_thresholds(model_name, conf_threshold, max_instances):
    """
    Re-evaluates all sequences for the given model by applying the specified confidence threshold
    and max_instances limit to the raw predictions. Aggregates metrics across single_object, multi_object,
    and all_objects categories.

    Returns:
        overall_mAP (float): Overall mAP averaged over sequences.
        aggregated_metrics (dict): Aggregated metrics grouped by categories.
    """
    gt_json_dir = Path(GT_JSONS_DIR)
    seq_files = list(gt_json_dir.glob("*_gt.json"))
    sequence_names = [f.stem.replace("_gt", "") for f in seq_files]

    mAPs = []
    all_metrics = {}
    for seq in sequence_names:
        print(f"Evaluating sequence {seq} with conf_threshold={conf_threshold} and max_instances={max_instances}")
        seq_metrics = evaluate_sequence_metrics_thresholded(seq, model_name, conf_threshold, max_instances,
                                                            det_iou_threshold=IOU_THRESHOLD)
        all_metrics[seq] = seq_metrics
        mAPs.append(seq_metrics.get("detection", {}).get("mAP", 0))

    overall_mAP = float(np.mean(mAPs)) if mAPs else 0.0
    aggregated_metrics = aggregate_thresholded_metrics(all_metrics)

    return overall_mAP, aggregated_metrics


def evaluate_sequence_metrics_thresholded(sequence_name, model_name, conf_threshold, max_instances,
                                          det_iou_threshold=IOU_THRESHOLD):
    """
    Evaluates a given sequence for a specified model by applying a confidence threshold and max_instances limit.
    This function re-loads the raw predictions, filters detections by conf_threshold, limits the number of detections
    to max_instances, and then computes segmentation and detection metrics.

    Returns:
        dict: Aggregated metrics for the sequence (including processing_time).
    """
    start_time = time.time()
    gt_ann = load_gt_json(sequence_name)
    gt_masks = load_gt_masks(sequence_name)
    pred_ann = load_predicted_json(model_name, sequence_name)
    pred_masks = load_predicted_masks(model_name, sequence_name)

    pixel_iou_vals = []
    dice_vals = []
    precision_vals = []
    recall_vals = []
    f1_vals = []
    label_acc_vals = []
    box_iou_vals = []

    all_gt_boxes = []
    all_gt_labels = []
    all_pred_boxes = []
    all_pred_labels = []

    for frame, gt_data in gt_ann.items():
        if frame in gt_masks and frame in pred_masks:
            gt_mask = gt_masks[frame]
            pred_mask = pred_masks[frame]
            pixel_iou_vals.append(compute_pixel_iou(gt_mask, pred_mask))
            dice_vals.append(compute_dice(gt_mask, pred_mask))

        gt_boxes = gt_data.get("boxes", [])
        gt_labels = gt_data.get("labels", [])
        all_gt_boxes.extend(gt_boxes)
        all_gt_labels.extend(gt_labels)

        if frame in pred_ann:
            pred_frame = pred_ann[frame]
            raw_pred_boxes = pred_frame.get("boxes", [])
            raw_pred_labels = pred_frame.get("labels", [])
            raw_pred_scores = pred_frame.get("scores", [])

            # Filter out predictions below the confidence threshold.
            filtered_boxes = []
            filtered_labels = []
            for box, label, score in zip(raw_pred_boxes, raw_pred_labels, raw_pred_scores):
                if score >= conf_threshold:
                    filtered_boxes.append(box)
                    filtered_labels.append(label)

            # Limit the number of detections to max_instances.
            if len(filtered_boxes) > max_instances:
                valid_scores = [score for score in raw_pred_scores if score >= conf_threshold]
                sorted_indices = np.argsort([-s for s in valid_scores])
                filtered_boxes = [filtered_boxes[i] for i in sorted_indices[:max_instances]]
                filtered_labels = [filtered_labels[i] for i in sorted_indices[:max_instances]]

            all_pred_boxes.extend(filtered_boxes)
            all_pred_labels.extend(filtered_labels)

            det_metrics = compute_detection_metrics(gt_boxes, gt_labels, filtered_boxes, filtered_labels,
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
    class_precisions = compute_detection_metrics_per_class(all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_labels,
                                                           iou_threshold=det_iou_threshold)
    mAP = compute_map_from_class_precisions(class_precisions)
    aggregated_det["per_class_precision"] = class_precisions
    aggregated_det["mAP"] = mAP

    processing_time = time.time() - start_time
    return {
        "segmentation": aggregated_seg,
        "detection": aggregated_det,
        "processing_time": processing_time
    }


# ===================================================
# MAIN EXECUTION (Raw Metrics and Aggregation)
# ===================================================

if __name__ == "__main__":
    models = ["yolo", "maskrcnn", "yolo_deeplab"]
    base_model_metrics_dir = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics")

    # --- Raw Metrics Computation & Saving ---
    for model in models:
        print(f"Evaluating raw metrics for model: {model}")
        model_metrics = evaluate_model_metrics(model, det_iou_threshold=IOU_THRESHOLD)
        save_model_metrics(model_metrics, model)

    # --- Aggregation of Raw Metrics ---
    for model in models:
        model_metrics_file = base_model_metrics_dir / model / f"{model}_metrics.json"
        if not model_metrics_file.exists():
            print(f"Raw metrics file not found for model {model}: {model_metrics_file}")
            continue
        with open(model_metrics_file, "r") as f:
            raw_metrics = json.load(f)
        aggregated_metrics = aggregate_model_metrics(raw_metrics)
        output_file = base_model_metrics_dir / model / f"{model}_aggregated_metrics.json"
        with open(output_file, "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        print(f"Aggregated metrics for model '{model}' saved to {output_file}")