import json
import os

import cv2
import sys
from pathlib import Path
import numpy as np
from src.data_loader import load_frames  # load_frames is available in data_loader.py
from src.model import YOLOv8Model, SAMModel, MaskRCNNModel, HybridSAMYOLOModel
from sklearn.metrics import average_precision_score, jaccard_score

# Adjust system path to ensure modules are found correctly

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

output_dir = project_root / "output"  # Set output directory

# Set paths for evaluation (Using our Generated Annotations)
GT_ANNOTATIONS_DIR_MULTI = Path(os.path.join(output_dir, "multi_object_annotations"))
GT_ANNOTATIONS_DIR_SINGLE = Path(os.path.join(output_dir, "single_object_annotations"))

# Paths for Bounding Boxes & Labels (Our Ground Truth)
GT_BBOX_JSON_MULTI = Path(os.path.join(output_dir, "multi_object_boundingboxes_labels.json"))
GT_BBOX_JSON_SINGLE = Path(os.path.join(output_dir, "single_object_boundingboxes_labels.json"))

# Load ground-truth bounding boxes and labels
with open(GT_BBOX_JSON_MULTI, "r") as f:
    gt_bboxes_multi = json.load(f)

with open(GT_BBOX_JSON_SINGLE, "r") as f:
    gt_bboxes_single = json.load(f)


# ------------------------
# HELPER FUNCTIONS
# ------------------------
def load_ground_truth_mask(sequence_name, frame_name, is_multi_object=True):
    """
    Load the ground-truth segmentation mask for a given frame.
    Uses our generated masks from multi-object or single-object annotations.

    Args:
        sequence_name (str): The sequence name (video name).
        frame_name (str): The frame name (e.g., "00000.png").
        is_multi_object (bool): Whether to fetch from multi-object or single-object annotations.

    Returns:
        mask (np.ndarray): The loaded segmentation mask.
    """
    annotations_dir = GT_ANNOTATIONS_DIR_MULTI if is_multi_object else GT_ANNOTATIONS_DIR_SINGLE
    mask_path = annotations_dir / sequence_name / frame_name

    if not mask_path.exists():
        print(f"Warning: Ground-truth mask not found for {frame_name} in {sequence_name}")
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask

def calculate_bbox_iou(boxA, boxB):
    """
    Compute IoU between two bounding boxes in [x, y, w, h] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    return interArea / unionArea if unionArea > 0 else 0

def calculate_map(pred_boxes, gt_boxes):
    """
    Compute Mean Average Precision (mAP) for object detection.

    Args:
        pred_boxes (list): Predicted bounding boxes [[x,y,w,h], ...].
        gt_boxes (list): Ground-truth bounding boxes [[x,y,w,h], ...].

    Returns:
        float: mAP score.
    """
    matched = np.zeros(len(gt_boxes))
    scores = np.zeros(len(gt_boxes))  # prediction correctness

    for i, gt in enumerate(gt_boxes):
        for pred in pred_boxes:
            iou = calculate_bbox_iou(pred, gt)
            if iou > 0.5:
                matched[i] = 1
                scores[i] = 1
                break

    if len(gt_boxes) == 0:
        return 0.0

    print("y_true:", matched)
    print("y_pred:", scores)

    return average_precision_score(matched, scores)

def calculate_maks_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union (IoU) for segmentation masks.

    Args:
        pred_mask (np.ndarray): The predicted segmentation mask.
        gt_mask (np.ndarray): The ground-truth segmentation mask.

    Returns:
        float: The IoU score.
    """
    if pred_mask is None or gt_mask is None:
        return 0  # If one mask is missing, IoU is 0

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    return intersection / union if union > 0 else 0

# ------------------------
# EVALUATION FUNCTION
# ------------------------
def evaluate_model(model, sequence_name, gt_bboxes, is_multi_object=True):
    """
    Evaluate the model on a given video sequence using ground-truth annotations.

    Args:
        model: The object detection & segmentation model to evaluate.
        sequence_name (str): Name of the sequence being evaluated.
        gt_bboxes (dict): The ground-truth bounding boxes & labels.
        is_multi_object (bool): Whether evaluating multi-object or single-object sequences.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    total_iou = []
    total_map = []

    # Load all frames from the sequence
    frames = load_frames(sequence_name)

    for frame_idx, frame in enumerate(frames):
        frame_name = f"{frame_idx:05d}.png"  # Assuming frame format is "00000.png"

        # Run model inference
        results = model.predict(frame)
        if hasattr(results, "boxes") and results.boxes is not None:
            xyxy = results.boxes.xyxy.cpu().numpy()  # shape: (N, 4) -> [x1, y1, x2, y2]
            pred_boxes = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in xyxy]
        else:
            pred_boxes = []

        if hasattr(results, "masks") and results.masks is not None:
            pred_masks = results.masks.data[0].cpu().numpy().astype(np.uint8)  # Binary mask
        else:
            pred_masks = None

        # Load Ground Truth
        gt_mask = load_ground_truth_mask(sequence_name, frame_name, is_multi_object)

        # Extract list of [x, y, w, h] from label->bbox mapping
        gt_boxes_raw = gt_bboxes.get(sequence_name, {}).get(frame_name, {})
        gt_boxes = list(gt_boxes_raw.values()) if isinstance(gt_boxes_raw, dict) else gt_boxes_raw

        # Compute IoU for segmentation
        iou = calculate_maks_iou(pred_masks, gt_mask)
        total_iou.append(iou)

        # Compute mAP for object detection
        mAP = calculate_map(pred_boxes, gt_boxes)
        total_map.append(mAP)

        # Log failure cases (no matched predictions)
        y_true = np.zeros(len(pred_boxes))
        y_pred = np.zeros(len(pred_boxes))

        for i, pred in enumerate(pred_boxes):
            for gt in gt_boxes:
                iou = calculate_bbox_iou(pred, gt)
                if iou > 0.5:
                    y_true[i] = 1
                    y_pred[i] = 1
                    break

        if np.sum(y_true) == 0:
            print(f"⚠️ No true positives found in sequence: {sequence_name}, frame: {frame_name}")
            print("Ground Truth Boxes:", gt_boxes)
            print("Predicted Boxes:", pred_boxes)

    # Compute Average Metrics
    avg_iou = np.mean(total_iou) if total_iou else 0
    avg_map = np.mean(total_map) if total_map else 0

    return {"IoU": avg_iou, "mAP": avg_map}


# ------------------------
# RUN EVALUATION
# ------------------------
if __name__ == "__main__":
    # Sequences to evaluate (change as needed)
    sequences_multi = list(gt_bboxes_multi.keys())[:5]  # Test on first 5 sequences
    sequences_single = list(gt_bboxes_single.keys())[:5]

    # Load Models
    models = {
        "YOLOv8": YOLOv8Model(),
        "SAM": SAMModel(),
        "MaskRCNN": MaskRCNNModel(),
        "HybridSAMYOLO": HybridSAMYOLOModel(),
    }

    # Evaluate each model
    results = {}
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} on Multi-Object Sequences...")
        results[f"{model_name}_multi"] = {
            seq: evaluate_model(model, seq, gt_bboxes_multi, is_multi_object=True) for seq in sequences_multi
        }

        print(f"\nEvaluating {model_name} on Single-Object Sequences...")
        results[f"{model_name}_single"] = {
            seq: evaluate_model(model, seq, gt_bboxes_single, is_multi_object=False) for seq in sequences_single
        }

    # Save results
    output_results_path = Path("output") / "evaluation_results.json"
    with open(output_results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation Complete. Results saved to:", output_results_path.resolve())



