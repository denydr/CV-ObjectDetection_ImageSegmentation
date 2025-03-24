import cv2
import json
from pathlib import Path
import os

# Import paths and parameters from config.py
from config import (
    DAVIS_RAW_FRAMES_DIR,
    RAW_MASKS_DIR,
    REP_BBOX_JSON,
    REP_MASKS_MULTI,
    REP_MASKS_SINGLE
)
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

if __name__ == "__main__":
    # For testing purposes, choose a representative sequence name.
    sequence_name = "bike-packing"

    print("Loading representative bounding boxes and labels...")
    annotations = load_representative_bbox_annotations()
    multi_obj_annotations = annotations.get("multi_object", {}).get(sequence_name, {})
    print(f"Annotations for {sequence_name}:", multi_obj_annotations)

    print("\nLoading raw frames...")
    frames = load_raw_frames(sequence_name)
    print(f"Loaded {len(frames)} raw frames for sequence '{sequence_name}'.")

    print("\nLoading converted masks for evaluation (multi-object)...")
    conv_masks = load_converted_masks(sequence_name, is_multi_object=True)
    print(f"Loaded {len(conv_masks)} converted masks for sequence '{sequence_name}'.")

    print("\nLoading raw RGB masks...")
    raw_masks = load_raw_masks(sequence_name)
    print(f"Loaded {len(raw_masks)} raw masks for sequence '{sequence_name}'.")