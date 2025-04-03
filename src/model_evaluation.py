import re
import torch
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
    CONFIDENCE_THRESHOLD,
    MAX_INSTANCES,
    CANONICAL_MAPPING_PATH,
    YOLO_CHECKPOINT,
    MASKRCNN_MODEL_PATH,
    MASKRCNN_BACKBONE_PATH,
    YOLO_DETECTION_PATH,
    DEEPLAB_PATH,
    DEEPLAB_DIR,
)

# Import the semantic standardization functions from your label_standardization.py module.
from label_standardization import (
    load_canonical_mapping,
    compute_canonical_embeddings,
    standardize_labels_semantic
)

from label_standardization_coco import (
    standardize_coco_labels
)

from model_handler import get_model


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


#-------------MODEL INFERENCE---------------------YOLO
# Converts numeric prediction indices into their corresponding string labels
# using the model's names mapping.
# def convert_indices_to_labels(pred_indices, model):
#     if hasattr(model, "names"):
#         names = model.names
#     elif hasattr(model, "model") and hasattr(model.model, "names"):
#         names = model.model.names
#     else:
#         raise AttributeError("The model does not have a 'names' attribute for label lookup.")
#     return [names.get(int(idx), "unknown") for idx in pred_indices]
#
# def run_model_inference_on_frame(model, frame):
#     boxes, pred_indices, scores, masks = model.predict(frame)
#     return boxes, pred_indices, scores, masks
#
# def evaluate_sequence(sequence_name, object_type, model, canonical_mapping, canonical_embeddings):
#     print(f"\nEvaluating sequence: {sequence_name} ({object_type})")
#
#     # Load raw frames.
#     raw_frames = data_loader.load_raw_frames(sequence_name)
#     print(f"  Loaded {len(raw_frames)} raw frames from {DAVIS_RAW_FRAMES_DIR / sequence_name}")
#
#     # Load ground‑truth annotations.
#     all_annotations = data_loader.load_representative_bbox_annotations()
#     gt_annotations = all_annotations.get(object_type, {}).get(sequence_name, {})
#     print(f"  Loaded bounding box annotations for {len(gt_annotations)} frames from {REP_BBOX_JSON}")
#
#     # Load evaluation‑ready masks.
#     eval_masks = data_loader.load_converted_masks(sequence_name, is_multi_object=(object_type=="multi_object"))
#     mask_dir = REP_MASKS_MULTI if object_type == "multi_object" else REP_MASKS_SINGLE
#     print(f"  Loaded {len(eval_masks)} evaluation‑ready masks from {mask_dir}")
#
#     # Optionally, load raw RGB masks for overlay comparisons.
#     raw_masks = data_loader.load_raw_masks(sequence_name)
#     print(f"  Loaded {len(raw_masks)} raw masks from {RAW_MASKS_DIR}")
#
#     # Create a single window for display.
#     window_name = f"{sequence_name} Evaluation"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#
#     for frame_filename, frame_img in raw_frames:
#         # Run model inference.
#         boxes, pred_indices, scores, pred_masks = run_model_inference_on_frame(model, frame_img)
#
#         # Convert indices to label strings.
#         pred_label_strings = convert_indices_to_labels(pred_indices, model)
#         # Standardize labels semantically using the cached canonical embeddings.
#         std_labels = standardize_labels_semantic(pred_label_strings, canonical_mapping, canonical_embeddings, threshold=0.8)
#
#         print(f"Frame {frame_filename}:")
#         print(f"  Predicted boxes: {boxes}")
#         print(f"  Predicted labels: {pred_label_strings}")
#         print(f"  Standardized labels: {std_labels}")
#         print(f"  Scores: {scores}")
#
#         # Start with a fresh copy of the raw frame.
#         annotated_frame = frame_img.copy()
#
#         # Draw predicted boxes and standardized labels.
#         for box, label in zip(boxes, std_labels):
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(annotated_frame, str(label), (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         # Create a composite mask image.
#         composite_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
#         if pred_masks is not None and pred_masks.size > 0:
#             # Define a fixed palette of colors.
#             colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
#                       (0, 255, 255), (255, 0, 255), (255, 255, 0)]
#             for i in range(pred_masks.shape[0]):
#                 binary_mask = (pred_masks[i] > 0.5).astype(np.uint8) * 255
#                 # Resize mask if needed.
#                 if binary_mask.shape != annotated_frame.shape[:2]:
#                     binary_mask = cv2.resize(binary_mask, (annotated_frame.shape[1], annotated_frame.shape[0]))
#                 mask_color = np.zeros_like(annotated_frame, dtype=np.uint8)
#                 color = colors[i % len(colors)]
#                 mask_color[binary_mask == 255] = color
#                 composite_mask = cv2.addWeighted(composite_mask, 1.0, mask_color, 0.5, 0)
#             display_frame = cv2.addWeighted(annotated_frame, 0.7, composite_mask, 0.3, 0)
#         elif frame_filename in raw_masks:
#             mask = raw_masks[frame_filename]
#             if mask.shape[:2] != annotated_frame.shape[:2]:
#                 mask = cv2.resize(mask, (annotated_frame.shape[1], annotated_frame.shape[0]))
#             if len(mask.shape) == 2:
#                 mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#             else:
#                 mask_bgr = mask
#             display_frame = cv2.addWeighted(annotated_frame, 0.7, mask_bgr, 0.3, 0)
#         else:
#             display_frame = annotated_frame
#
#         cv2.imshow(window_name, display_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#
# def main():
#     # Load canonical mapping from config.
#     try:
#         canonical_mapping = load_canonical_mapping()
#     except Exception as e:
#         print(f"Error loading canonical mapping: {e}")
#         return
#
#     print("Canonical Mapping:")
#     for key, value in canonical_mapping.items():
#         print(f"  {key}: {value}")
#
#     # Precompute canonical embeddings.
#     canonical_embeddings = compute_canonical_embeddings(canonical_mapping)
#
#     # Initialize the model (YOLOv8 segmentation).
#     model = get_model("yolo")  # Using the default checkpoint and device from config.
#
#     # Load complete ground‑truth annotations JSON.
#     all_annotations = data_loader.load_representative_bbox_annotations()
#
#     # Process multi-object sequences.
#     multi_sequences = list(all_annotations.get("multi_object", {}).keys())
#     print(f"\nFound {len(multi_sequences)} sequences for object type 'multi_object'")
#     for sequence_name in multi_sequences:
#         evaluate_sequence(sequence_name, "multi_object", model, canonical_mapping, canonical_embeddings)
#         print(f"Finished evaluating sequence: {sequence_name}\n")
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Process single-object sequences.
#     single_sequences = list(all_annotations.get("single_object", {}).keys())
#     print(f"\nFound {len(single_sequences)} sequences for object type 'single_object'")
#     for sequence_name in single_sequences:
#         evaluate_sequence(sequence_name, "single_object", model, canonical_mapping, canonical_embeddings)
#         print(f"Finished evaluating sequence: {sequence_name}\n")
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# if __name__ == "__main__":
#     main()


#-------------MODEL INFERENCE---------------------YOLO, MASKRCNN, YOLO-DeepLabV3
# def convert_indices_to_labels(pred_indices, model):
#     if hasattr(model, "names"):
#         names = model.names
#     elif hasattr(model, "model") and hasattr(model.model, "names"):
#         names = model.model.names
#     else:
#         return [str(i) for i in pred_indices]  # fallback
#     return [names.get(int(idx), "unknown") for idx in pred_indices]
#
# def run_model_inference_on_frame(model, frame):
#     return model.predict(frame)
#
# def evaluate_sequence(sequence_name, object_type, model, canonical_mapping, canonical_embeddings):
#     print(f"\n📽️ Evaluating sequence: {sequence_name} ({object_type})")
#
#     raw_frames = data_loader.load_raw_frames(sequence_name)
#     print(f"  📸 Loaded {len(raw_frames)} raw frames")
#
#     all_annotations = data_loader.load_representative_bbox_annotations()
#     gt_annotations = all_annotations.get(object_type, {}).get(sequence_name, {})
#     eval_masks = data_loader.load_converted_masks(sequence_name, is_multi_object=(object_type == "multi_object"))
#     raw_masks = data_loader.load_raw_masks(sequence_name)
#
#     mask_dir = REP_MASKS_MULTI if object_type == "multi_object" else REP_MASKS_SINGLE
#     print(f"  📦 GT annotations: {len(gt_annotations)}  |  Eval masks: {len(eval_masks)}  |  Raw masks: {len(raw_masks)}")
#
#     window_name = f"{sequence_name} Evaluation"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#
#     for frame_filename, frame_img in raw_frames:
#         boxes, pred_indices, scores, pred_masks = run_model_inference_on_frame(model, frame_img)
#
#         # Apply confidence threshold
#         keep_indices = scores >= CONFIDENCE_THRESHOLD
#         boxes = boxes[keep_indices]
#         pred_indices = pred_indices[keep_indices]
#         scores = scores[keep_indices]
#         if pred_masks is not None and pred_masks.shape[0] == len(keep_indices):
#             pred_masks = pred_masks[keep_indices]
#
#         # Sort by confidence and keep top N
#         if len(scores) > MAX_INSTANCES:
#             top_indices = np.argsort(scores)[-MAX_INSTANCES:][::-1]  # top scores first
#             boxes = boxes[top_indices]
#             pred_indices = pred_indices[top_indices]
#             scores = scores[top_indices]
#             if pred_masks is not None and pred_masks.shape[0] >= len(top_indices):
#                 pred_masks = pred_masks[top_indices]
#
#         # 🔁 Handle label logic based on model type
#         model_type = type(model).__name__.lower()
#         if "maskrcnn" in model_type:
#             std_labels = standardize_coco_labels(pred_indices, canonical_mapping, canonical_embeddings, threshold=0.75)
#             pred_label_strings = [str(label) for label in pred_indices]
#         else:
#             pred_label_strings = convert_indices_to_labels(pred_indices, model)
#             std_labels = standardize_labels_semantic(pred_label_strings, canonical_mapping, canonical_embeddings, threshold=0.8)
#
#         print(f"🖼️ Frame {frame_filename}:")
#         print(f"  🟥 Boxes: {boxes}")
#         print(f"  🏷️ Labels: {pred_label_strings}")
#         print(f"  ✅ Standardized: {std_labels}")
#         print(f"  📊 Scores: {scores}")
#
#         annotated_frame = frame_img.copy()
#
#         for box, label in zip(boxes, std_labels):
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(annotated_frame, str(label), (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         # Create composite mask image
#         composite_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
#         if pred_masks is not None and pred_masks.size > 0:
#             colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
#                       (0, 255, 255), (255, 0, 255), (255, 255, 0)]
#             for i in range(pred_masks.shape[0]):
#                 binary_mask = (pred_masks[i] > 0.5).astype(np.uint8) * 255
#                 if binary_mask.shape != annotated_frame.shape[:2]:
#                     binary_mask = cv2.resize(binary_mask, (annotated_frame.shape[1], annotated_frame.shape[0]))
#                 mask_color = np.zeros_like(annotated_frame, dtype=np.uint8)
#                 mask_color[binary_mask == 255] = colors[i % len(colors)]
#                 composite_mask = cv2.addWeighted(composite_mask, 1.0, mask_color, 0.5, 0)
#             display_frame = cv2.addWeighted(annotated_frame, 0.7, composite_mask, 0.3, 0)
#         elif frame_filename in raw_masks:
#             mask = raw_masks[frame_filename]
#             if mask.shape[:2] != annotated_frame.shape[:2]:
#                 mask = cv2.resize(mask, (annotated_frame.shape[1], annotated_frame.shape[0]))
#             mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
#             display_frame = cv2.addWeighted(annotated_frame, 0.7, mask_bgr, 0.3, 0)
#         else:
#             display_frame = annotated_frame
#
#         cv2.imshow(window_name, display_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#
# def main():
#     try:
#         canonical_mapping = load_canonical_mapping()
#     except Exception as e:
#         print(f"❌ Error loading canonical mapping: {e}")
#         return
#
#     canonical_embeddings = compute_canonical_embeddings(canonical_mapping)
#
#     # 🔀 Choose model here
#     model = get_model("yolo")
#     model = get_model("maskrcnn")
#     model = get_model("yolo_deeplab")
#
#     all_annotations = data_loader.load_representative_bbox_annotations()
#
#     for object_type in ["multi_object", "single_object"]:
#         sequences = list(all_annotations.get(object_type, {}).keys())
#         print(f"\n📂 Found {len(sequences)} sequences for {object_type}")
#         for sequence_name in sequences:
#             evaluate_sequence(sequence_name, object_type, model, canonical_mapping, canonical_embeddings)
#             print(f"✅ Finished sequence: {sequence_name}")
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
# if __name__ == "__main__":
#     main()


# -------------MODEL INFERENCE & SAVING PREDICTIONS---------------------YOLO, MASKRCNN, YOLO-DeepLabV3
# import os
# import json
# import cv2
# import numpy as np
# from pathlib import Path
#
# # ----- Import configuration and helper functions -----
# from config import DAVIS_RAW_FRAMES_DIR, OUTPUT_DIR, REP_BBOX_JSON, REP_MASKS_MULTI, REP_MASKS_SINGLE, \
#     CANONICAL_MAPPING_PATH
# from data_loader import load_raw_frames  # load_raw_frames, load_representative_bbox_annotations, etc.
# # Note: The function get_sequence_names is not provided in data_loader.
# from label_standardization import load_canonical_mapping, compute_canonical_embeddings, standardize_labels_semantic
# from label_standardization_coco import standardize_coco_labels
# from model_handler import get_model
#
#
# # ---------------------------------------------------------------------
# def list_representative_sequences(object_type="multi_object"):
#     """
#     Extracts and returns a list of sequence names from the annotations.
#
#     Args:
#         object_type (str): "multi_object" or "single_object" (depending on the annotations structure).
#
#     Returns:
#         list: Sequence names.
#     """
#     annotations = load_representative_annotations()
#     sequences = list(annotations.get(object_type, {}).keys())
#     return sequences
# # Define convert_indices_to_labels since it is not provided in label_standardization.
# def convert_indices_to_labels(pred_indices, model):
#     """
#     Converts numeric prediction indices into their corresponding string labels using the model's mapping.
#     """
#     if hasattr(model, "names"):
#         names = model.names
#     elif hasattr(model, "model") and hasattr(model.model, "names"):
#         names = model.model.names
#     else:
#         return [str(i) for i in pred_indices]
#     return [names.get(int(idx), "unknown") for idx in pred_indices]
#
#
# # ---------------------------------------------------------------------
# def run_inference_and_save(sequence_name, object_type, model, model_name, canonical_mapping, canonical_embeddings):
#     """
#     Runs inference on each frame in a sequence and saves the predicted boxes,
#     standardized labels, scores, and (if available) masks.
#
#     The predicted results (JSON and mask images) are saved under:
#     /Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/metrics_artifacts/predictions/maskrcnn_predicted_masks/<model_name>/
#     with a separate subdirectory for each sequence.
#     """
#     base_dir = "/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/metrics_artifacts"
#     predictions_dir = os.path.join(base_dir, "predictions", "maskrcnn_predicted_masks", model_name)
#     output_json_path = os.path.join(predictions_dir, f"{sequence_name}_predictions.json")
#     output_mask_dir = os.path.join(predictions_dir, sequence_name)
#
#     os.makedirs(output_mask_dir, exist_ok=True)
#     os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
#
#     predictions = {}  # To store predictions per frame.
#     raw_frames = load_raw_frames(sequence_name)
#     print(f"Processing {len(raw_frames)} frames for sequence: {sequence_name} using model: {model_name}")
#
#     # Helper: Generate a unique color for each instance (background remains black).
#     def get_color(instance_index):
#         colors = [
#             (128, 0, 0), (0, 128, 0), (0, 0, 128),
#             (128, 128, 0), (128, 0, 128), (0, 128, 128),
#             (64, 0, 0), (0, 64, 0), (0, 0, 64),
#             (64, 64, 0), (64, 0, 64), (0, 64, 64)
#         ]
#         if instance_index - 1 < len(colors):
#             return colors[instance_index - 1]
#         else:
#             np.random.seed(instance_index)
#             return tuple(np.random.randint(1, 256, size=3).tolist())
#
#     for frame_filename, frame_img in raw_frames:
#         # Run inference on the frame.
#         boxes, pred_indices, scores, pred_masks = model.predict(frame_img)
#
#         # Convert indices to labels and standardize them.
#         model_type = type(model).__name__.lower()
#         if "maskrcnn" in model_type:
#             std_labels = standardize_coco_labels(pred_indices, canonical_mapping, canonical_embeddings, threshold=0.75)
#             pred_label_strings = [str(label) for label in pred_indices]
#         else:
#             pred_label_strings = convert_indices_to_labels(pred_indices, model)
#             std_labels = standardize_labels_semantic(pred_label_strings, canonical_mapping, canonical_embeddings,
#                                                      threshold=0.8)
#
#         # Save predicted masks.
#         mask_filenames = []
#         if pred_masks is not None and pred_masks.size > 0:
#             if object_type == "single_object":
#                 # Combine all predicted masks (logical OR) into one binary mask.
#                 combined_binary_mask = np.zeros(frame_img.shape[:2], dtype=np.uint8)
#                 for i in range(pred_masks.shape[0]):
#                     binary_mask = (pred_masks[i] > 0.5).astype(np.uint8)
#                     if binary_mask.shape != frame_img.shape[:2]:
#                         binary_mask = cv2.resize(binary_mask, (frame_img.shape[1], frame_img.shape[0]),
#                                                  interpolation=cv2.INTER_NEAREST)
#                     combined_binary_mask = np.maximum(combined_binary_mask, binary_mask)
#                 combined_binary_mask = combined_binary_mask * 255
#                 mask_filename = os.path.basename(frame_filename)  # e.g., "00000.png"
#                 mask_filepath = os.path.join(output_mask_dir, mask_filename)
#                 cv2.imwrite(mask_filepath, combined_binary_mask)
#                 mask_filenames.append(mask_filename)
#             elif object_type == "multi_object":
#                 # Create a combined color-coded (BGR) segmentation mask.
#                 height, width = frame_img.shape[:2]
#                 combined_color_mask = np.zeros((height, width, 3), dtype=np.uint8)
#                 for i in range(pred_masks.shape[0]):
#                     instance_mask = (pred_masks[i] > 0.5).astype(np.uint8)
#                     if instance_mask.shape != (height, width):
#                         instance_mask = cv2.resize(instance_mask, (width, height), interpolation=cv2.INTER_NEAREST)
#                     color = get_color(i + 1)  # Instance indices start at 1 to avoid black.
#                     color_image = np.full((height, width, 3), color, dtype=np.uint8)
#                     combined_color_mask = np.where(instance_mask[..., None].astype(bool), color_image,
#                                                    combined_color_mask)
#                 mask_filename = os.path.basename(frame_filename)  # e.g., "00000.png"
#                 mask_filepath = os.path.join(output_mask_dir, mask_filename)
#                 cv2.imwrite(mask_filepath, combined_color_mask)
#                 mask_filenames.append(mask_filename)
#
#         predictions[frame_filename] = {
#             "boxes": boxes.tolist() if hasattr(boxes, "tolist") else boxes,
#             "labels": std_labels,
#             "scores": scores.tolist() if hasattr(scores, "tolist") else scores,
#             "mask_files": mask_filenames
#         }
#         print(f"Saved predictions for frame {frame_filename}")
#
#     with open(output_json_path, "w") as f:
#         json.dump(predictions, f, indent=4)
#     print(f"Predictions for sequence '{sequence_name}' saved to: {output_json_path}")
#
#
# # ---------------------------------------------------------------------
# def main():
#     # Load canonical mapping and compute embeddings.
#     canonical_mapping = load_canonical_mapping()
#     canonical_embeddings = compute_canonical_embeddings(canonical_mapping)
#
#     # Get the three SoA models.
#     models = {
#         "yolo": get_model("yolo"),
#         "maskrcnn": get_model("maskrcnn"),
#         "yolo_deeplab": get_model("yolo_deeplab")
#     }
#
#     # For each object type, get sequence names using the evaluation module's function.
#     for object_type in ["single_object", "multi_object"]:
#         sequences = list_representative_sequences(object_type)
#         for sequence_name in sequences:
#             for model_name, model in models.items():
#                 print(
#                     f"Running inference for sequence '{sequence_name}', object type '{object_type}', using model '{model_name}'")
#                 run_inference_and_save(sequence_name, object_type, model, model_name, canonical_mapping,
#                                        canonical_embeddings)
#
#
# if __name__ == "__main__":
#     main()


# -------------MODEL INFERENCE & SAVING PREPROCESSED PREDICTIONS-------BINARY MASKS+LABEL-TO-COLOUR--------------YOLO, MASKRCNN, YOLO-DeepLabV3
# import os
# import shutil
# import cv2
# import numpy as np
# from pathlib import Path
#
#
# def color_mask_to_label_image(mask):
#     """
#     Converts a color-coded segmentation mask (BGR) into a single-channel label image.
#     Each unique color (except background, assumed black) is mapped to a unique integer.
#
#     Args:
#         mask (np.ndarray): Color-coded mask (BGR).
#
#     Returns:
#         label_img (np.ndarray): Single-channel label image (dtype=uint16).
#         color_to_label (dict): Mapping from each unique color (tuple) to its assigned integer label.
#     """
#     # Convert BGR to RGB for consistency.
#     mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#     h, w, _ = mask_rgb.shape
#     # Reshape into a list of pixels.
#     pixels = mask_rgb.reshape(-1, 3)
#     unique_colors = {tuple(color) for color in np.unique(pixels, axis=0)}
#
#     color_to_label = {}
#     current_label = 1
#     for color in unique_colors:
#         if color == (0, 0, 0):  # Background remains black.
#             color_to_label[color] = 0
#         else:
#             color_to_label[color] = current_label
#             current_label += 1
#
#     label_img = np.zeros((h, w), dtype=np.uint16)
#     # For clarity, using loops (this could be vectorized for speed).
#     for y in range(h):
#         for x in range(w):
#             col = tuple(int(c) for c in mask_rgb[y, x])
#             label_img[y, x] = color_to_label.get(col, 0)
#     return label_img, color_to_label
#
#
# # List of multi-object sequences (as provided).
# multi_object_sequences = [
#     "bike-packing", "boxing-fisheye", "cat-girl", "classic-car", "dancing",
#     "dogs-jump", "hockey", "horsejump-high", "judo", "kid-football", "lady-running",
#     "mbike-trick", "motocross-jump", "paragliding", "pigs", "skate-park", "snowboard",
#     "stroller", "upside-down"
# ]
#
# # Define source and destination base directories.
# src_base = Path(
#     "/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/metrics_artifacts/predictions/maskrcnn_predicted_masks")
# dst_base = Path(
#     "/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/metrics_artifacts/predictions/predicted_masks_preprocessed")
#
# # Ensure the destination base directory exists.
# dst_base.mkdir(parents=True, exist_ok=True)
#
# # Process each model directory (e.g. "maskrcnn", "yolo", "yolo_deeplab").
# for model_dir in src_base.iterdir():
#     if not model_dir.is_dir():
#         continue
#     print(f"Processing model directory: {model_dir.name}")
#     dst_model_dir = dst_base / model_dir.name
#     dst_model_dir.mkdir(parents=True, exist_ok=True)
#
#     # Copy any JSON prediction files from the model folder.
#     for item in model_dir.iterdir():
#         if item.is_file() and item.suffix == ".json":
#             shutil.copy2(item, dst_model_dir / item.name)
#             print(f"Copied JSON file: {item.name}")
#
#     # Process each sequence folder within the model directory.
#     for seq_dir in model_dir.iterdir():
#         if not seq_dir.is_dir():
#             continue
#         sequence_name = seq_dir.name
#         dst_seq_dir = dst_model_dir / sequence_name
#         dst_seq_dir.mkdir(parents=True, exist_ok=True)
#
#         if sequence_name in multi_object_sequences:
#             print(f"Converting multi-object sequence '{sequence_name}' in model '{model_dir.name}'")
#             # For each mask image in the multi-object sequence, convert to label image.
#             for mask_file in seq_dir.glob("*.png"):
#                 mask = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
#                 if mask is None:
#                     print(f"Warning: Could not load mask {mask_file.name}")
#                     continue
#                 label_img, _ = color_mask_to_label_image(mask)
#                 dst_mask_path = dst_seq_dir / mask_file.name
#                 # Save as 16-bit PNG (to preserve label values).
#                 cv2.imwrite(str(dst_mask_path), label_img)
#                 print(f"Converted and saved {mask_file.name} to {dst_mask_path}")
#         else:
#             # For single-object sequences, simply copy the predicted mask files.
#             print(f"Copying single-object sequence '{sequence_name}' in model '{model_dir.name}'")
#             for mask_file in seq_dir.glob("*.png"):
#                 dst_mask_path = dst_seq_dir / mask_file.name
#                 shutil.copy2(mask_file, dst_mask_path)
#                 print(f"Copied {mask_file.name} to {dst_mask_path}")