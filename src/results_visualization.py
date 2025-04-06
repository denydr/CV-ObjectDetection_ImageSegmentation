# --CONFUSION MATRICES & LABELS, BOXES GRAPHS--
#
# import json
# import os
# from pathlib import Path
# from collections import defaultdict
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
#
# from config import GT_JSONS_DIR, CANONICAL_MAPPING_PATH
# from data_loader import load_gt_json, load_predicted_json, load_predicted_json_with_thresholding
# from metrics_evaluation import compute_iou
#
#
# def load_canonical_mapping():
#     with open(CANONICAL_MAPPING_PATH, "r") as f:
#         mapping = json.load(f)
#     return {str(v): k for k, v in mapping.items()}  # inverse: {"0": "person", ...}
#
#
# def collect_labels_and_boxes(model_name, label_map, use_thresholding=False, conf=0.5, max_inst=5):
#     gt_labels, pred_labels = [], []
#     gt_boxes_all, pred_boxes_all = [], []
#
#     for gt_file in Path(GT_JSONS_DIR).glob("*_gt.json"):
#         sequence = gt_file.stem.replace("_gt", "")
#         gt_ann = load_gt_json(sequence)
#         if use_thresholding:
#             pred_ann = load_predicted_json_with_thresholding(model_name, sequence, conf, max_inst)
#         else:
#             pred_ann = load_predicted_json(model_name, sequence)
#
#         for frame, gt_data in gt_ann.items():
#             gt_raw_labels = gt_data.get("labels", [])
#             gt_raw_boxes = gt_data.get("boxes", [])
#             pred_data = pred_ann.get(frame, {}) if pred_ann and frame in pred_ann else {}
#             pred_raw_labels = pred_data.get("labels", [])
#             pred_raw_boxes = pred_data.get("boxes", [])
#
#             # Match only up to min length to keep alignment
#             min_len = min(len(gt_raw_labels), len(pred_raw_labels))
#             gt_raw_labels, pred_raw_labels = gt_raw_labels[:min_len], pred_raw_labels[:min_len]
#             gt_raw_boxes, pred_raw_boxes = gt_raw_boxes[:min_len], pred_raw_boxes[:min_len]
#
#             for g_lbl, p_lbl in zip(gt_raw_labels, pred_raw_labels):
#                 if g_lbl == -1 or p_lbl == -1:
#                     continue
#                 gt_labels.append(label_map.get(str(g_lbl), f"label_{g_lbl}"))
#                 pred_labels.append(label_map.get(str(p_lbl), f"label_{p_lbl}"))
#
#             gt_boxes_all.extend(gt_raw_boxes)
#             pred_boxes_all.extend(pred_raw_boxes)
#
#     return gt_labels, pred_labels, gt_boxes_all, pred_boxes_all
#
#
# def plot_conf_matrix(y_true, y_pred, title, output_path):
#     import numpy as np
#     labels = sorted(set(y_true + y_pred))
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#
#     plt.figure(figsize=(12, 10))
#     ax = plt.gca()
#
#     mask = np.zeros_like(cm, dtype=bool)
#     for i in range(len(labels)):
#         for j in range(len(labels)):
#             if i != j:
#                 mask[i, j] = True
#
#     sns.heatmap(cm, mask=mask, annot=True, fmt='d', cmap="Blues",
#                 xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
#     sns.heatmap(cm, mask=~mask, annot=True, fmt='d', cmap="Reds",
#                 xticklabels=labels, yticklabels=labels, alpha=0.7, cbar=False, ax=ax)
#
#     ax.set_xlabel("Predicted Labels")
#     ax.set_ylabel("Ground Truth Labels")
#     ax.set_title(title)
#     plt.xticks(rotation=90)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()
#     print(f"✅ Saved confusion matrix: {output_path}")
#
#
# def plot_tp_fp_barplot(correct, incorrect, title, output_path):
#     total = correct + incorrect
#     percentages = [correct / total * 100, incorrect / total * 100]
#     labels = ['TP', 'FP']
#     colors = ['green', 'red']
#
#     plt.figure(figsize=(6, 6))
#     sns.barplot(x=labels, y=percentages, palette=colors)
#     plt.title(title)
#     plt.ylabel("Percentage (%)")
#     plt.ylim(0, 100)
#     for i, v in enumerate(percentages):
#         plt.text(i, v + 1, f"{v:.1f}%", ha='center')
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()
#     print(f"✅ Saved TP/FP bar plot: {output_path}")
#
#
# def evaluate_and_plot(model_name, label_map, mode, conf=None, max_inst=None):
#     is_thresholded = mode != "raw"
#     mode_str = f"conf{conf}_max{max_inst}" if is_thresholded else "raw"
#
#     out_base_cm = Path(f"/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/results_visualization/confusion_matrices/{model_name}")
#     out_base_label = Path(f"/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/results_visualization/graphs/label_accuracy/{model_name}")
#     out_base_box = Path(f"/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/results_visualization/graphs/box_accuracy/{model_name}")
#     out_base_cm.mkdir(parents=True, exist_ok=True)
#     out_base_label.mkdir(parents=True, exist_ok=True)
#     out_base_box.mkdir(parents=True, exist_ok=True)
#
#     gt_labels, pred_labels, gt_boxes, pred_boxes = collect_labels_and_boxes(
#         model_name, label_map, use_thresholding=is_thresholded, conf=conf, max_inst=max_inst
#     )
#
#     # Confusion matrix
#     plot_conf_matrix(
#         gt_labels, pred_labels,
#         title=f"{model_name.upper()} - {mode.title()}",
#         output_path=out_base_cm / f"{model_name}_{mode_str}_confusion_matrix.png"
#     )
#
#     # TP/FP label-wise
#     correct_labels = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
#     incorrect_labels = len(gt_labels) - correct_labels
#     plot_tp_fp_barplot(
#         correct_labels, incorrect_labels,
#         title=f"{model_name.upper()} - Label Accuracy ({mode_str})",
#         output_path=out_base_label / f"{model_name}_{mode_str}_label_accuracy.png"
#     )
#
#     # TP/FP box-wise (IoU > 0.5)
#     correct_boxes = 0
#     for p_box in pred_boxes:
#         for g_box in gt_boxes:
#             if compute_iou(p_box, g_box) >= 0.5:
#                 correct_boxes += 1
#                 break
#     incorrect_boxes = len(pred_boxes) - correct_boxes
#     plot_tp_fp_barplot(
#         correct_boxes, incorrect_boxes,
#         title=f"{model_name.upper()} - Box IoU > 0.5 Accuracy ({mode_str})",
#         output_path=out_base_box / f"{model_name}_{mode_str}_box_accuracy.png"
#     )
#
#
# def full_pipeline(model_name):
#     label_map = load_canonical_mapping()
#     evaluate_and_plot(model_name, label_map, mode="raw")
#     evaluate_and_plot(model_name, label_map, mode="default", conf=0.5, max_inst=5)
#     evaluate_and_plot(model_name, label_map, mode="calibrated", conf=0.8, max_inst=3)
#
#
# # Run full pipeline for all models
# if __name__ == "__main__":
#     for model in ["yolo", "maskrcnn", "yolo_deeplab"]:
#         print(f"\n🔍 Running full visualization pipeline for {model}")
#         full_pipeline(model)



#--SEGMENTATION METRICS GRAPHS--------------------------------
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from config import GT_JSONS_DIR, GT_MASKS_DIR
# from data_loader import load_gt_json, load_gt_masks, load_predicted_masks
#
# # ------------------------
# # Metric computation
# # ------------------------
#
# def compute_pixel_iou(gt_mask, pred_mask):
#     gt_bin = (gt_mask > 0).astype(np.uint8)
#     pred_bin = (pred_mask > 0).astype(np.uint8)
#     intersection = np.logical_and(gt_bin, pred_bin).sum()
#     union = np.logical_or(gt_bin, pred_bin).sum() + 1e-6
#     return intersection / union
#
# def compute_dice(gt_mask, pred_mask):
#     gt_bin = (gt_mask > 0).astype(np.uint8)
#     pred_bin = (pred_mask > 0).astype(np.uint8)
#     intersection = np.logical_and(gt_bin, pred_bin).sum()
#     return (2 * intersection) / (gt_bin.sum() + pred_bin.sum() + 1e-6)
#
# # ------------------------
# # Evaluation logic
# # ------------------------
#
# def evaluate_segmentation(model_name, mode="raw"):
#     """
#     Computes average pixel IoU and Dice score across all sequences and frames.
#     """
#     iou_scores = []
#     dice_scores = []
#
#     for gt_file in Path(GT_JSONS_DIR).glob("*_gt.json"):
#         sequence = gt_file.stem.replace("_gt", "")
#         gt_masks = load_gt_masks(sequence)
#         pred_masks = load_predicted_masks(model_name, sequence)
#
#         for frame_name in gt_masks:
#             if frame_name not in pred_masks:
#                 continue
#             gt = gt_masks[frame_name]
#             pred = pred_masks[frame_name]
#
#             iou_scores.append(compute_pixel_iou(gt, pred))
#             dice_scores.append(compute_dice(gt, pred))
#
#     return {
#         "Pixel IoU": np.mean(iou_scores),
#         "Dice": np.mean(dice_scores),
#         "N frames": len(iou_scores)
#     }
#
# # ------------------------
# # Plotting logic
# # ------------------------
#
# def plot_segmentation_scores(model_name, scores_dict, mode_suffix):
#     """
#     Saves bar chart with average pixel IoU and Dice score.
#     """
#     output_dir = Path(f"/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/results_visualization/graphs/segmentation_accuracy/{model_name}")
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     metrics = ["Pixel IoU", "Dice"]
#     values = [scores_dict["Pixel IoU"], scores_dict["Dice"]]
#
#     plt.figure(figsize=(6, 4))
#     bars = plt.bar(metrics, values, color=["skyblue", "lightgreen"])
#     plt.ylim(0, 1)
#     plt.title(f"{model_name.upper()} - Segmentation ({mode_suffix})")
#     plt.ylabel("Score")
#     for bar in bars:
#         bar.set_edgecolor("black")
#     plt.tight_layout()
#     filename = f"{model_name}_{mode_suffix.replace(' ', '_').lower()}_segmentation.png"
#     plt.savefig(output_dir / filename)
#     plt.close()
#     print(f"✅ Saved segmentation chart: {output_dir / filename}")
#
# # ------------------------
# # Pipeline Entry
# # ------------------------
#
# def run_segmentation_pipeline():
#     models = ["yolo", "maskrcnn", "yolo_deeplab"]
#     modes = {
#         "raw": "raw"
#     }
#
#     for model in models:
#         for mode_key, mode_label in modes.items():
#             result = evaluate_segmentation(model, mode=mode_key)
#             plot_segmentation_scores(model, result, mode_label)
#
# # ------------------------
# # Main
# # ------------------------
#
# if __name__ == "__main__":
#     run_segmentation_pipeline()


#--SEGMENTATION ACCURACY GRAPHS--------------------------------
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from config import GT_MASKS_DIR, YOLO_PREDICTED_MASKS_DIR, MASKRCNN_PREDICTED_MASKS_DIR, YOLO_DEEPLAB_PREDICTED_MASKS_DIR
#
# # Define where to save results
# VIS_DIR = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/results_visualization/graphs/segmentation_accuracy/{model_name}")
# VIS_DIR.mkdir(parents=True, exist_ok=True)
#
# # Multi-object sequences use label-to-color conversion (converted to single-channel already)
# MULTI_OBJECT_SEQUENCES = [
#     "bike-packing", "boxing-fisheye", "cat-girl", "classic-car", "dancing", "dogs-jump",
#     "hockey", "horsejump-high", "judo", "kid-football", "lady-running", "mbike-trick",
#     "motocross-jump", "paragliding", "pigs", "skate-park", "snowboard", "stroller", "upside-down"
# ]
#
# MODEL_DIRS = {
#     "yolo": YOLO_PREDICTED_MASKS_DIR,
#     "maskrcnn": MASKRCNN_PREDICTED_MASKS_DIR,
#     "yolo_deeplab": YOLO_DEEPLAB_PREDICTED_MASKS_DIR
# }
#
#
# def compute_segmentation_tp_fp_fn(gt_mask, pred_mask):
#     gt_bin = (gt_mask > 0).astype(np.uint8)
#     pred_bin = (pred_mask > 0).astype(np.uint8)
#
#     tp = np.logical_and(gt_bin == 1, pred_bin == 1).sum()
#     fp = np.logical_and(gt_bin == 0, pred_bin == 1).sum()
#     fn = np.logical_and(gt_bin == 1, pred_bin == 0).sum()
#
#     return tp, fp, fn
#
#
# def process_model_masks(model_name, pred_base_path):
#     total_tp, total_fp, total_fn = 0, 0, 0
#     for sequence_dir in pred_base_path.iterdir():
#         if not sequence_dir.is_dir():
#             continue
#         sequence_name = sequence_dir.name
#         gt_seq_dir = GT_MASKS_DIR / sequence_name
#
#         for pred_mask_file in sequence_dir.glob("*.png"):
#             gt_mask_file = gt_seq_dir / pred_mask_file.name
#             if not gt_mask_file.exists():
#                 print(f"⚠️ Skipping {gt_mask_file} (not found)")
#                 continue
#
#             # Load GT and predicted masks
#             gt_mask = cv2.imread(str(gt_mask_file), cv2.IMREAD_GRAYSCALE)
#             pred_mask = cv2.imread(str(pred_mask_file), cv2.IMREAD_GRAYSCALE)
#
#             if gt_mask is None or pred_mask is None:
#                 print(f"⚠️ Skipping unreadable mask: {pred_mask_file.name}")
#                 continue
#
#             # Resize if mismatch
#             if gt_mask.shape != pred_mask.shape:
#                 pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
#
#             # Convert to binary for single-object
#             if sequence_name not in MULTI_OBJECT_SEQUENCES:
#                 gt_mask = (gt_mask > 0).astype(np.uint8)
#                 pred_mask = (pred_mask > 0).astype(np.uint8)
#
#             tp, fp, fn = compute_segmentation_tp_fp_fn(gt_mask, pred_mask)
#             total_tp += tp
#             total_fp += fp
#             total_fn += fn
#
#     return total_tp, total_fp, total_fn
#
#
# def plot_tp_fp_fn_bar(model_name, tp, fp, fn):
#     plt.figure(figsize=(7, 5))
#     plt.bar(["True Positives", "False Positives", "False Negatives"], [tp, fp, fn],
#             color=["green", "red", "orange"])
#     plt.ylabel("Pixel Count")
#     plt.title(f"{model_name.upper()} - Segmentation TP/FP/FN Summary")
#     plt.tight_layout()
#
#     out_path = VIS_DIR / f"{model_name}_mask_tp_fp_fn.png"
#     plt.savefig(out_path)
#     plt.close()
#     print(f"📊 Saved mask TP/FP/FN graph: {out_path}")
#
#
# def run_for_all_models():
#     for model_name, pred_path in MODEL_DIRS.items():
#         print(f"🧠 Processing {model_name}...")
#         tp, fp, fn = process_model_masks(model_name, pred_path)
#         plot_tp_fp_fn_bar(model_name, tp, fp, fn)
#
#
# if __name__ == "__main__":
#     run_for_all_models()



# --Temporal consistency analysis--------------------------------
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# # Define model metric file paths
# base_path = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics")
#
# models = {
#     "YOLO": {
#         "raw": base_path / "yolo" / "yolo_metrics.json",
#         "thresholded": base_path / "yolo" / "thresholded_metrics" / "per_sequence" / "yolo_conf0.50_max5_per_sequence.json"
#     },
#     "MaskRCNN": {
#         "raw": base_path / "maskrcnn" / "maskrcnn_metrics.json",
#         "thresholded": base_path / "maskrcnn" / "thresholded_metrics" / "per_sequence" / "maskrcnn_conf0.50_max5_per_sequence.json"
#     },
#     "YOLO_DeepLab": {
#         "raw": base_path / "yolo_deeplab" / "yolo_deeplab_metrics.json",
#         "thresholded": base_path / "yolo_deeplab" / "thresholded_metrics" / "per_sequence" / "yolo_deeplab_conf0.50_max5_per_sequence.json"
#     }
# }
#
# def compute_temporal_std(file_path):
#     if not file_path.exists():
#         return None
#     with open(file_path, "r") as f:
#         data = json.load(f)
#     dice_scores = [
#         seq_data.get("segmentation", {}).get("dice", {}).get("mean")
#         for seq, seq_data in data.items()
#         if seq != "processing_time" and seq_data.get("segmentation", {}).get("dice", {}).get("mean") is not None
#     ]
#     return round(float(np.std(dice_scores)), 4) if dice_scores else None
#
# # Compute for both raw and thresholded
# temporal_consistency = {"raw": {}, "thresholded": {}}
# for model, paths in models.items():
#     for mode in ["raw", "thresholded"]:
#         std = compute_temporal_std(paths[mode])
#         if std is not None:
#             temporal_consistency[mode][model] = std
#
# # Plot
# output_dir = base_path.parent / "results_visualization" / "graphs" / "temporal_consistency"
# output_dir.mkdir(parents=True, exist_ok=True)
#
# labels = list(models.keys())
# x = np.arange(len(labels))
# width = 0.35
#
# fig, ax = plt.subplots(figsize=(9, 5))
# raw_vals = [temporal_consistency["raw"].get(label, 0) for label in labels]
# thresh_vals = [temporal_consistency["thresholded"].get(label, 0) for label in labels]
#
# bars1 = ax.bar(x - width / 2, raw_vals, width, label="Raw", color='skyblue')
# bars2 = ax.bar(x + width / 2, thresh_vals, width, label="Thresholded", color='orange')
#
# ax.set_ylabel("Std Deviation of Dice Scores")
# ax.set_title("Temporal Consistency Across Models (Lower is Better)")
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
# ax.grid(axis="y", linestyle="--", alpha=0.6)
#
# plt.tight_layout()
# plt.savefig(output_dir / "temporal_consistency_comparison_barplot.png")
# plt.show()



# --Temporal consistency analysis-> PER-SEQUENCE-------------------------------
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Paths
BASE_METRICS_DIR = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/model_metrics")
OUTPUT_DIR = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/results_visualization/graphs/temporal_consistency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["yolo", "maskrcnn", "yolo_deeplab"]

def load_dice_std_per_sequence(path):
    with open(path, "r") as f:
        data = json.load(f)
    stds = {}
    for seq, metrics in data.items():
        if seq == "processing_time":
            continue
        dice_std = metrics.get("segmentation", {}).get("dice", {}).get("std", None)
        if dice_std is not None:
            stds[seq] = dice_std
    return stds

def visualize_raw_temporal_consistency():
    for model in MODELS:
        raw_path = BASE_METRICS_DIR / model / f"{model}_metrics.json"
        print(f"🔍 Raw path: {raw_path}, exists? {raw_path.exists()}")

        if not raw_path.exists():
            continue

        results = load_dice_std_per_sequence(raw_path)

        # Sort sequences by name
        sequences = sorted(results.keys())
        dice_stds = [results[seq] for seq in sequences]

        plt.figure(figsize=(14, 6))
        plt.plot(sequences, dice_stds, marker='o', linestyle='-', color='blue', label='Raw Dice Std')
        plt.xticks(rotation=90)
        plt.title(f"{model.upper()} - Raw Temporal Consistency (Dice Std per Sequence)")
        plt.xlabel("Sequence")
        plt.ylabel("Dice Std Deviation")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{model}_raw_temporal_consistency_per_sequence.png")
        plt.close()
        print(f"✅ Saved: {model}_raw_temporal_consistency_per_sequence.png")

if __name__ == "__main__":
    visualize_raw_temporal_consistency()