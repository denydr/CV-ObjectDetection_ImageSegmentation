from sentence_transformers import SentenceTransformer, util
import torch
import json
import re
from pathlib import Path
from config import CANONICAL_MAPPING_PATH  # Import the mapping path from config

def load_canonical_mapping(mapping_path=CANONICAL_MAPPING_PATH):
    """Loads the canonical mapping from a JSON file."""
    with open(Path(mapping_path), "r") as f:
        return json.load(f)

# Load a pretrained SentenceTransformer model globally.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_label_embedding(label: str) -> torch.Tensor:
    """
    Encodes a label string into an embedding.
    """
    return embedding_model.encode(label, convert_to_tensor=True)

def compute_canonical_embeddings(canonical_mapping: dict) -> dict:
    """
    Precompute and cache embeddings for each canonical label.
    Returns a dictionary mapping each canonical label (string) to its embedding tensor.
    """
    canonical_embeddings = {}
    for label in canonical_mapping.keys():
        canonical_embeddings[label] = get_label_embedding(label)
    return canonical_embeddings

def standardize_label_semantic(pred_label: str, canonical_mapping: dict, canonical_embeddings: dict, threshold: float = 0.8) -> int:
    """
    Standardizes a predicted label by comparing its semantic embedding with those of the canonical labels.
    If the highest cosine similarity score exceeds the threshold, returns the canonical integer code;
    otherwise, returns -1.
    """
    # Ensure the predicted label is a string.
    if not isinstance(pred_label, str):
        pred_label = str(pred_label)
    pred_embedding = get_label_embedding(pred_label)

    best_match = None
    best_score = -1.0

    for canon_label, code in canonical_mapping.items():
        # Use the cached embedding for the canonical label.
        canon_embedding = canonical_embeddings[canon_label]
        score = util.pytorch_cos_sim(pred_embedding, canon_embedding).item()
        if score > best_score:
            best_score = score
            best_match = code

    if best_score >= threshold:
        return best_match
    else:
        print(f"Warning: Predicted label '{pred_label}' did not match any canonical label (best score: {best_score:.2f}).")
        return -1

def standardize_labels_semantic(pred_labels, canonical_mapping, canonical_embeddings, threshold: float = 0.8):
    """
    Standardizes a list of predicted labels using semantic similarity and cached canonical embeddings.
    Returns a list of standardized label integer codes.
    """
    standardized = []
    for label in pred_labels:
        if not isinstance(label, str):
            label = str(label)
        std_code = standardize_label_semantic(label, canonical_mapping, canonical_embeddings, threshold)
        standardized.append(std_code)
    return standardized

# Example usage.
if __name__ == "__main__":
    # Load the canonical mapping using the path from config.
    canonical_mapping = load_canonical_mapping()
    print("Canonical Mapping:")
    for key, value in canonical_mapping.items():
        print(f"  {key}: {value}")

    # Precompute canonical embeddings.
    canonical_embeddings = compute_canonical_embeddings(canonical_mapping)

    # Example predicted labels (for instance, from your model).
    predicted_labels = ["puppy", "kitten", "avian", "human"]

    std_labels = standardize_labels_semantic(predicted_labels, canonical_mapping, canonical_embeddings, threshold=0.75)
    print("Predicted labels:", predicted_labels)
    print("Standardized labels:", std_labels)


# from sentence_transformers import SentenceTransformer, util
# import torch
# import json
# import re
# from pathlib import Path
# from config import CANONICAL_MAPPING_PATH  # Import the mapping path from config
#
# # COCO class ID to name mapping
# COCO_ID_TO_NAME = {
#     0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
#     6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
#     11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
#     16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
#     22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
#     27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
#     32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
#     36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
#     40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
#     46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
#     51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
#     57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
#     62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
#     68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
#     73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
#     78: "hair drier", 79: "toothbrush"
# }
#
#
# def load_canonical_mapping(mapping_path=CANONICAL_MAPPING_PATH):
#     """Loads the canonical mapping from a JSON file."""
#     with open(Path(mapping_path), "r") as f:
#         return json.load(f)
#
# # Load a pretrained SentenceTransformer model globally.
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#
# def get_label_embedding(label: str) -> torch.Tensor:
#     """
#     Encodes a label string into an embedding.
#     """
#     return embedding_model.encode(label, convert_to_tensor=True)
#
# def compute_canonical_embeddings(canonical_mapping: dict) -> dict:
#     """
#     Precompute and cache embeddings for each canonical label.
#     Returns a dictionary mapping each canonical label (string) to its embedding tensor.
#     """
#     canonical_embeddings = {}
#     for label in canonical_mapping.keys():
#         canonical_embeddings[label] = get_label_embedding(label)
#     return canonical_embeddings
#
# def standardize_label_semantic(pred_label: str, canonical_mapping: dict, canonical_embeddings: dict, threshold: float = 0.8) -> int:
#     """
#     Standardizes a predicted label by comparing its semantic embedding with those of the canonical labels.
#     If the highest cosine similarity score exceeds the threshold, returns the canonical integer code;
#     otherwise, returns -1.
#     """
#     # Ensure the predicted label is a string.
#     if not isinstance(pred_label, str):
#         pred_label = str(pred_label)
#     pred_embedding = get_label_embedding(pred_label)
#
#     best_match = None
#     best_score = -1.0
#
#     for canon_label, code in canonical_mapping.items():
#         # Use the cached embedding for the canonical label.
#         canon_embedding = canonical_embeddings[canon_label]
#         score = util.pytorch_cos_sim(pred_embedding, canon_embedding).item()
#         if score > best_score:
#             best_score = score
#             best_match = code
#
#     if best_score >= threshold:
#         return best_match
#     else:
#         print(f"Warning: Predicted label '{pred_label}' did not match any canonical label (best score: {best_score:.2f}).")
#         return -1
#
# def standardize_labels_semantic(pred_labels, canonical_mapping, canonical_embeddings, threshold: float = 0.8):
#     """
#     Standardizes a list of predicted labels using semantic similarity and cached canonical embeddings.
#     Handles both string labels and COCO-style integer IDs.
#     """
#     standardized = []
#     for label in pred_labels:
#         # Convert integer label to COCO name if needed
#         if isinstance(label, (int, float)):
#             label_str = COCO_ID_TO_NAME.get(int(label), str(label))
#         else:
#             label_str = str(label)
#
#         std_code = standardize_label_semantic(label_str, canonical_mapping, canonical_embeddings, threshold)
#         standardized.append(std_code)
#     return standardized
#
# # Example usage.
# if __name__ == "__main__":
#     canonical_mapping = load_canonical_mapping()
#     print("Canonical Mapping:")
#     for key, value in canonical_mapping.items():
#         print(f"  {key}: {value}")
#
#     canonical_embeddings = compute_canonical_embeddings(canonical_mapping)
#
#     # Mixed test set: YOLO-style strings + COCO-style numeric labels
#     predicted_labels = ["puppy", "kitten", "bird", "human", 0, 16]  # 0=person, 16=dog in COCO
#
#     std_labels = standardize_labels_semantic(predicted_labels, canonical_mapping, canonical_embeddings, threshold=0.75)
#     print("Predicted labels:", predicted_labels)
#     print("Standardized labels (canonical numeric codes):", std_labels)
