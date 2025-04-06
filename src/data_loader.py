import cv2
import json
from pathlib import Path
import os

# ----- Original Imports from config -----
from config import (
    DAVIS_RAW_FRAMES_DIR,
    RAW_MASKS_DIR,
    REP_BBOX_JSON,
    REP_MASKS_MULTI,
    REP_MASKS_SINGLE,
    # New GT and predicted paths:
    GT_MASKS_DIR,
    GT_JSONS_DIR,
    YOLO_PREDICTED_MASKS_DIR,
    YOLO_PREDICTED_JSONS_DIR,
    MASKRCNN_PREDICTED_MASKS_DIR,
    MASKRCNN_PREDICTED_JSONS_DIR,
    YOLO_DEEPLAB_PREDICTED_MASKS_DIR,
    YOLO_DEEPLAB_PREDICTED_JSONS_DIR
)


# ----- Original Functions -----
def load_representative_bbox_annotations():
    """
    Loads the representative dataset bounding boxes and labels JSON.
    Returns:
        dict: The annotations dictionary.
    """
    json_path = REP_BBOX_JSON
    if not json_path.exists():
        raise FileNotFoundError(f"Bounding boxes JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_raw_frames(sequence_name):
    """
    Loads raw image frames for a given sequence from DAVIS dataset.

    Args:
        sequence_name (str): Name of the sequence folder.

    Returns:
        list of tuples: Each tuple is (frame_filename, image_array).
    """
    seq_path = DAVIS_RAW_FRAMES_DIR / sequence_name
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence folder not found: {seq_path}")

    frame_files = sorted(seq_path.glob("*.png"))
    frames = []
    for frame_file in frame_files:
        img = cv2.imread(str(frame_file))
        if img is None:
            print(f"Warning: Could not load frame {frame_file.name}")
            continue
        frames.append((frame_file.name, img))
    return frames


def load_converted_masks(sequence_name, is_multi_object=True):
    """
    Loads the converted (evaluation-ready) masks for a given sequence.

    Args:
        sequence_name (str): Name of the sequence.
        is_multi_object (bool): If True, loads masks from multi-object folder;
                                otherwise, from single-object folder.

    Returns:
        dict: Mapping from frame filename to mask image (loaded with IMREAD_UNCHANGED).
    """
    mask_root = REP_MASKS_MULTI if is_multi_object else REP_MASKS_SINGLE
    seq_mask_path = mask_root / sequence_name
    masks = {}
    if not seq_mask_path.exists():
        print(f"Warning: Mask folder not found for sequence: {seq_mask_path}")
        return masks
    mask_files = sorted(seq_mask_path.glob("*.png"))
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Could not load mask {mask_file.name}")
            continue
        masks[mask_file.name] = mask
    return masks


def load_raw_masks(sequence_name):
    """
    Loads the raw RGB ground-truth masks for a given sequence.

    Args:
        sequence_name (str): Name of the sequence.

    Returns:
        dict: Mapping from frame filename to raw mask image (loaded in color).
    """
    seq_mask_path = RAW_MASKS_DIR / sequence_name
    masks = {}
    if not seq_mask_path.exists():
        print(f"Warning: Raw mask folder not found for sequence: {seq_mask_path}")
        return masks
    mask_files = sorted(seq_mask_path.glob("*.png"))
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
        if mask is None:
            print(f"Warning: Could not load raw mask {mask_file.name}")
            continue
        masks[mask_file.name] = mask
    return masks


# ----- New Functions for GT Annotations and Masks -----
def load_gt_json(sequence_name):
    """
    Loads the ground truth JSON for the given sequence.
    Assumes filename: "<sequence_name>_gt.json" in GT_JSONS_DIR.

    Returns:
        dict: The parsed GT JSON.
    """
    from config import GT_JSONS_DIR  # Import here to use updated path.
    json_path = GT_JSONS_DIR / f"{sequence_name}_gt.json"
    if not json_path.exists():
        raise FileNotFoundError(f"GT JSON not found for sequence {sequence_name}: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_gt_masks(sequence_name):
    """
    Loads ground truth masks for a given sequence from GT_MASKS_DIR.

    Returns:
        dict: Mapping from filename to mask image (loaded with IMREAD_UNCHANGED).
    """
    from config import GT_MASKS_DIR
    seq_path = GT_MASKS_DIR / sequence_name
    masks = {}
    if not seq_path.exists():
        print(f"Warning: GT mask folder not found for sequence {sequence_name}: {seq_path}")
        return masks
    for mask_file in sorted(seq_path.glob("*.png")):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Could not load GT mask {mask_file.name}")
            continue
        masks[mask_file.name] = mask
    return masks


# ----- New Functions for Predicted Annotations and Masks -----
def load_predicted_json(model_name, sequence_name):
    """
    Loads the predicted JSON for a given sequence and model from the predicted JSONs directory.
    Assumes file is named "<sequence_name>_predictions.json" in the model's predicted JSON directory.

    Returns:
        dict: Parsed predicted JSON.
    """
    from config import YOLO_PREDICTED_JSONS_DIR, MASKRCNN_PREDICTED_JSONS_DIR, YOLO_DEEPLAB_PREDICTED_JSONS_DIR
    model_name_lower = model_name.lower()
    if model_name_lower == "yolo":
        base = YOLO_PREDICTED_JSONS_DIR
    elif model_name_lower == "maskrcnn":
        base = MASKRCNN_PREDICTED_JSONS_DIR
    elif model_name_lower == "yolo_deeplab":
        base = YOLO_DEEPLAB_PREDICTED_JSONS_DIR
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    json_path = base / f"{sequence_name}_predictions.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Predicted JSON not found for {model_name} sequence {sequence_name}: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def load_predicted_json_with_thresholding(model_name, sequence_name, confidence_threshold=None, max_instances=None):
    """
    Loads the predicted JSON and applies confidence filtering and instance limiting.
    """
    predictions = load_predicted_json(model_name, sequence_name)
    filtered_predictions = {}

    for frame_name, frame_data in predictions.items():
        boxes = frame_data.get("boxes", [])
        labels = frame_data.get("labels", [])
        scores = frame_data.get("scores", [])  # Only needed if thresholding

        if confidence_threshold is not None and scores:
            # Filter by confidence
            filtered = [
                (b, l, s) for b, l, s in zip(boxes, labels, scores) if s >= confidence_threshold
            ]
        else:
            filtered = list(zip(boxes, labels, scores)) if scores else list(zip(boxes, labels))

        # Limit max instances
        if max_instances is not None:
            filtered = sorted(filtered, key=lambda x: x[2] if len(x) == 3 else 1.0, reverse=True)
            filtered = filtered[:max_instances]

        # Unpack again
        new_boxes = [item[0] for item in filtered]
        new_labels = [item[1] for item in filtered]

        filtered_predictions[frame_name] = {
            "boxes": new_boxes,
            "labels": new_labels
        }

    return filtered_predictions


def load_predicted_masks(model_name, sequence_name):
    """
    Loads predicted masks for a given sequence and model from the predicted masks directory.

    Returns:
        dict: Mapping from filename to mask image (loaded with IMREAD_UNCHANGED).
    """
    from config import YOLO_PREDICTED_MASKS_DIR, MASKRCNN_PREDICTED_MASKS_DIR, YOLO_DEEPLAB_PREDICTED_MASKS_DIR
    model_name_lower = model_name.lower()
    if model_name_lower == "yolo":
        base = YOLO_PREDICTED_MASKS_DIR
    elif model_name_lower == "maskrcnn":
        base = MASKRCNN_PREDICTED_MASKS_DIR
    elif model_name_lower == "yolo_deeplab":
        base = YOLO_DEEPLAB_PREDICTED_MASKS_DIR
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    seq_path = base / sequence_name
    masks = {}
    if not seq_path.exists():
        print(f"Warning: Predicted mask folder not found for {model_name} sequence {sequence_name}: {seq_path}")
        return masks
    for mask_file in sorted(seq_path.glob("*.png")):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Warning: Could not load predicted mask {mask_file.name}")
            continue
        masks[mask_file.name] = mask
    return masks


if __name__ == "__main__":
    # For testing purposes:
    sequence_name = "bike-packing"
    print("Loading representative bounding boxes and labels...")
    annotations = load_representative_bbox_annotations()
    multi_obj_annotations = annotations.get("multi_object", {}).get(sequence_name, {})
    print(f"Annotations for {sequence_name}:", multi_obj_annotations)

    print("\nLoading raw frames...")
    frames = load_raw_frames(sequence_name)
    print(f"Loaded {len(frames)} raw frames for sequence '{sequence_name}'.")

    print("\nLoading converted masks (multi-object)...")
    conv_masks = load_converted_masks(sequence_name, is_multi_object=True)
    print(f"Loaded {len(conv_masks)} converted masks for sequence '{sequence_name}'.")

    print("\nLoading raw RGB masks...")
    raw_masks = load_raw_masks(sequence_name)
    print(f"Loaded {len(raw_masks)} raw masks for sequence '{sequence_name}'.")

    # Test new GT functions:
    try:
        gt_json = load_gt_json(sequence_name)
        print(f"Loaded GT JSON for '{sequence_name}'.")
    except Exception as e:
        print(e)
    gt_masks = load_gt_masks(sequence_name)
    print(f"Loaded {len(gt_masks)} GT masks for sequence '{sequence_name}'.")

    # Test new predicted functions (adjust model_name as needed):
    try:
        pred_json = load_predicted_json("maskrcnn", sequence_name)
        print(f"Loaded predicted JSON for 'maskrcnn' sequence '{sequence_name}'.")
    except Exception as e:
        print(e)
    pred_masks = load_predicted_masks("maskrcnn", sequence_name)
    print(f"Loaded {len(pred_masks)} predicted masks for 'maskrcnn' sequence '{sequence_name}'.")