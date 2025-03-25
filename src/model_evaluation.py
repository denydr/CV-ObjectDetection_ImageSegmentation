import re
import numpy as np
import cv2
import sys
import json
from pathlib import Path
import data_loader  # This module should contain functions like
                    # load_raw_frames, load_bbox_annotations, load_converted_masks, and load_raw_masks
from config import (
    DAVIS_RAW_FRAMES_DIR,
    REP_BBOX_JSON,
    REP_MASKS_MULTI,
    REP_MASKS_SINGLE,
    RAW_MASKS_DIR,
    CANONICAL_MAPPING_PATH,
    YOLO_CHECKPOINT
)
from model_handler import YOLOv8SegmentationModel


# Ensure the project root is in sys.path so that config.py can be imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# Import configuration settings from config.py
from config import (
    DAVIS_RAW_FRAMES_DIR,
    RAW_MASKS_DIR,
    REP_BBOX_JSON,
    REP_MASKS_MULTI,
    REP_MASKS_SINGLE,
    CANONICAL_MAPPING_PATH
)

def load_canonical_mapping():
    """
    Loads the canonical label mapping from the specified JSON file.

    Returns:
        dict: A dictionary mapping canonical labels to their standard numeric IDs.
    """
    mapping_path = Path(CANONICAL_MAPPING_PATH)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Canonical mapping file not found: {mapping_path}")
    with open(mapping_path, "r") as f:
        canonical_mapping = json.load(f)
    return canonical_mapping

#if __name__ == "__main__":
    # Load the canonical mapping
#    canonical_mapping = load_canonical_mapping()

    # Print out configuration info for verification
#    print("Configuration loaded:")
#    print(f"  DAVIS raw frames: {DAVIS_RAW_FRAMES_DIR}")
#   print(f"  Raw masks: {RAW_MASKS_DIR}")
#    print(f"  Representative BBox JSON: {REP_BBOX_JSON}")
#    print(f"  Multi-Object Masks: {REP_MASKS_MULTI}")
#    print(f"  Single-Object Masks: {REP_MASKS_SINGLE}")
#    print(f"  Canonical mapping file: {CANONICAL_MAPPING_PATH}")

    # Print canonical mapping for verification
#    print("\nCanonical Mapping:")
#    for key, value in canonical_mapping.items():
#        print(f"  {key}: {value}")


def load_representative_annotations():
    """
    Loads the representative bounding boxes+labels annotations JSON.

    Returns:
        dict: The annotations dictionary.
    """
    json_path = Path(REP_BBOX_JSON)
    if not json_path.exists():
        raise FileNotFoundError(f"Representative annotations not found: {json_path}")
    with open(json_path, "r") as f:
        annotations = json.load(f)
    return annotations

def list_representative_sequences(object_type="multi_object"):
    """
    Extracts and returns a list of sequence names from the annotations.

    Args:
        object_type (str): "multi_object" or "single_object" (depending on the annotations structure).

    Returns:
        list: Sequence names.
    """
    annotations = load_representative_annotations()
    sequences = list(annotations.get(object_type, {}).keys())
    return sequences

def list_frames_for_sequence(sequence_name, object_type="multi_object"):
    """
    Returns a list of frame filenames for a given sequence based on the annotations.

    Args:
        sequence_name (str): The name of the sequence.
        object_type (str): "multi_object" or "single_object".

    Returns:
        list: Frame filenames (e.g., "00000.png", "00001.png", etc.).
    """
    annotations = load_representative_annotations()
    seq_data = annotations.get(object_type, {}).get(sequence_name, {})
    return list(seq_data.keys())

def check_converted_masks_exist(sequence_name, object_type="multi_object"):
    """
    Checks whether the converted masks exist for the given sequence.

    Args:
        sequence_name (str): Name of the sequence.
        object_type (str): "multi_object" or "single_object".

    Returns:
        bool: True if masks exist, False otherwise.
    """
    mask_root = Path(REP_MASKS_MULTI) if object_type == "multi_object" else Path(REP_MASKS_SINGLE)
    seq_mask_dir = mask_root / sequence_name
    return seq_mask_dir.exists() and len(list(seq_mask_dir.glob("*.png"))) > 0

#if __name__ == "__main__":
    # Example for multi-object sequences:
#    sequences = list_representative_sequences("multi_object")
#    print("Representative Multi-Object Sequences:")
#    for seq in sequences:
#        print(f"Sequence: {seq}")
#        frames = list_frames_for_sequence(seq, "multi_object")
#        print(f"  Number of annotated frames: {len(frames)}")
#        masks_available = check_converted_masks_exist(seq, "multi_object")
#        print(f"  Converted masks available: {masks_available}")

    # If you also want to check single-object sequences:
#    single_sequences = list_representative_sequences("single_object")
#    print("\nRepresentative Single-Object Sequences:")
#    for seq in single_sequences:
#        print(f"Sequence: {seq}")
#        frames = list_frames_for_sequence(seq, "single_object")
#        print(f"  Number of annotated frames: {len(frames)}")
#       masks_available = check_converted_masks_exist(seq, "single_object")
#        print(f"  Converted masks available: {masks_available}")


def evaluate_sequence(sequence_name, object_type):
    print(f"\nEvaluating sequence: {sequence_name} ({object_type})")

    # Load raw frames from the DAVIS dataset.
    raw_frames = data_loader.load_raw_frames(sequence_name)
    print(f"  Loaded {len(raw_frames)} raw frames from {DAVIS_RAW_FRAMES_DIR / sequence_name}")

    # Load ground‑truth bounding boxes and labels from the representative JSON.
    all_annotations = data_loader.load_representative_bbox_annotations()
    gt_annotations = all_annotations.get(object_type, {}).get(sequence_name, {})
    print(f"  Loaded bounding box annotations for {len(gt_annotations)} frames from {REP_BBOX_JSON}")

    # Determine whether we're dealing with multi-object or single-object sequences.
    is_multi_object = (object_type == "multi_object")
    mask_dir = REP_MASKS_MULTI if is_multi_object else REP_MASKS_SINGLE

    # Load evaluation‑ready (converted) masks.
    eval_masks = data_loader.load_converted_masks(sequence_name, is_multi_object)
    print(f"  Loaded {len(eval_masks)} evaluation‑ready masks from {mask_dir}")

    # Optionally, load the raw RGB masks (for overlay comparisons).
    raw_masks = data_loader.load_raw_masks(sequence_name)
    print(f"  Loaded {len(raw_masks)} raw masks from {RAW_MASKS_DIR}")

    # Loop over each raw frame and overlay its ground‑truth bounding boxes.
    for frame_filename, frame_img in raw_frames:
        annotated_frame = frame_img.copy()
        if frame_filename in gt_annotations:
            for label, bbox in gt_annotations[frame_filename].items():
                # Assuming bbox format: [x, y, w, h]
                x, y, w, h = bbox
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # If a corresponding raw mask is available, overlay it with transparency.
        if frame_filename in raw_masks:
            mask = raw_masks[frame_filename]
            # Convert single-channel mask to BGR for display if needed.
            if len(mask.shape) == 2:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_bgr = mask
            display_frame = cv2.addWeighted(annotated_frame, 0.7, mask_bgr, 0.3, 0)
        else:
            display_frame = annotated_frame

        cv2.imshow(f"{sequence_name} - {frame_filename}", display_frame)
        # Wait 50ms between frames; press 'q' to exit early.
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        cv2.destroyWindow(f"{sequence_name} - {frame_filename}")
    cv2.destroyAllWindows()

# def main():
#     # Load the complete ground‑truth annotations JSON.
#     all_annotations = data_loader.load_representative_bbox_annotations()
#
#     # Process multi-object sequences.
#     multi_sequences = list(all_annotations.get("multi_object", {}).keys())
#     print(f"\nFound {len(multi_sequences)} multi_object sequences.")
#     for sequence_name in multi_sequences:
#         evaluate_sequence(sequence_name, "multi_object")
#         print(f"Finished evaluating sequence: {sequence_name}\n")
#
#     # Process single-object sequences.
#     single_sequences = list(all_annotations.get("single_object", {}).keys())
#     print(f"\nFound {len(single_sequences)} single_object sequences.")
#     for sequence_name in single_sequences:
#         evaluate_sequence(sequence_name, "single_object")
#         print(f"Finished evaluating sequence: {sequence_name}\n")
#
#
# if __name__ == "__main__":
#     main()


def load_canonical_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        return json.load(f)

def extract_base_label(label):
    # If label is a NumPy bytes type (or any bytes-like), decode it.
    if isinstance(label, np.bytes_):
        label = label.decode('utf-8')
    # Otherwise, if it's not already a string, try to decode it if possible.
    elif not isinstance(label, str):
        try:
            label = label.decode('utf-8')
        except Exception:
            label = str(label)
    # Now remove any trailing underscores or digits.
    base = re.sub(r'[_\d]+$', '', label)
    return base

def standardize_labels(pred_labels, canonical_mapping):
    standardized = []
    for label in pred_labels:
        # If label is numeric (e.g., a float), convert it to int and then to string.
        try:
            # First, try to convert to float and then to int.
            numeric = int(float(label))
            label_str = str(numeric)
        except Exception:
            # If conversion fails, decode or convert to string.
            if isinstance(label, bytes):
                label_str = label.decode('utf-8')
            else:
                label_str = str(label)
        # Now extract the base label (if necessary) or use the label directly.
        # In many cases, if labels are already numeric as strings, you might not need a regex.
        base_label = label_str  # or use extract_base_label(label_str) if additional cleaning is needed.
        # Map to the canonical mapping.
        std = canonical_mapping.get(base_label, -1)
        if std == -1:
            print(f"Warning: label '{label_str}' (extracted as '{base_label}') not found in canonical mapping")
        standardized.append(std)
    return standardized

def run_model_inference_on_frame(model, frame):
    """
    Runs the model on a single frame and returns predictions.
    Expected to return boxes, labels, scores, and masks.
    """
    boxes, labels, scores, masks = model.predict(frame)
    return boxes, labels, scores, masks

def evaluate_sequence(sequence_name, object_type, model, canonical_mapping):
    print(f"\nEvaluating sequence: {sequence_name} ({object_type})")

    # Load raw frames.
    raw_frames = data_loader.load_raw_frames(sequence_name)
    print(f"  Loaded {len(raw_frames)} raw frames from {DAVIS_RAW_FRAMES_DIR / sequence_name}")

    # Load ground-truth annotations.
    all_annotations = data_loader.load_representative_bbox_annotations()
    gt_annotations = all_annotations.get(object_type, {}).get(sequence_name, {})
    print(f"  Loaded bounding box annotations for {len(gt_annotations)} frames from {REP_BBOX_JSON}")

    # Load evaluation-ready masks.
    eval_masks = data_loader.load_converted_masks(sequence_name, object_type)
    mask_dir = REP_MASKS_MULTI if object_type == "multi_object" else REP_MASKS_SINGLE
    print(f"  Loaded {len(eval_masks)} evaluation-ready masks from {mask_dir}")

    # Optionally, load raw RGB masks for overlay comparisons.
    raw_masks = data_loader.load_raw_masks(sequence_name)
    print(f"  Loaded {len(raw_masks)} raw masks from {RAW_MASKS_DIR}")

    # For each frame, run model inference and standardize predicted labels.
    for frame_filename, frame_img in raw_frames:
        # Run model inference on the raw frame.
        boxes, pred_labels, scores, pred_masks = run_model_inference_on_frame(model, frame_img)
        std_labels = standardize_labels(pred_labels, canonical_mapping)
        print(f"Frame {frame_filename}:")
        print(f"  Predicted boxes: {boxes}")
        print(f"  Predicted labels: {pred_labels}")
        print(f"  Standardized labels: {std_labels}")
        print(f"  Scores: {scores}")
        # (You can also overlay the predictions on the frame for visualization)
        annotated_frame = frame_img.copy()
        # Draw predicted boxes (for example purposes, we use green rectangles)
        for box, label in zip(boxes, std_labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, str(label), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f"{sequence_name} - {frame_filename}", annotated_frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        cv2.destroyWindow(f"{sequence_name} - {frame_filename}")
    cv2.destroyAllWindows()

def main():
    # Load canonical mapping.
    canonical_mapping = load_canonical_mapping(Path(CANONICAL_MAPPING_PATH))
    print("Canonical Mapping:")
    for key, value in canonical_mapping.items():
        print(f"  {key}: {value}")

    # Initialize the model.
    model = YOLOv8SegmentationModel(model_path="yolov8n-seg.pt", device="cpu")

    # Load the complete ground-truth annotations JSON.
    all_annotations = data_loader.load_representative_bbox_annotations()

    # Loop over multi-object and single-object sequences.
    for object_type in ["multi_object", "single_object"]:
        sequences = list(all_annotations.get(object_type, {}).keys())
        print(f"\nFound {len(sequences)} sequences for object type '{object_type}'")
        for sequence_name in sequences:
            evaluate_sequence(sequence_name, object_type, model, canonical_mapping)
            print(f"Finished evaluating sequence: {sequence_name}\n")
            # Optional: pause between sequences
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()