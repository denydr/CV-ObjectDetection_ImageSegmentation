import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from config import CANONICAL_MAPPING_PATH

# -------------------------
# COCO Label Mapping (official 80-class IDs)
# -------------------------
COCO_ID_TO_NAME = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
    19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra",
    25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
    65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

# -------------------------
# Load canonical mapping
# -------------------------
def load_canonical_mapping(mapping_path=CANONICAL_MAPPING_PATH):
    with open(Path(mapping_path), "r") as f:
        return json.load(f)

# -------------------------
# Semantic matching logic
# -------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_label_embedding(label: str) -> torch.Tensor:
    return embedding_model.encode(label, convert_to_tensor=True)

def compute_canonical_embeddings(canonical_mapping: dict) -> dict:
    return {
        label: get_label_embedding(label)
        for label in canonical_mapping
    }

def standardize_label_semantic(pred_label: str, canonical_mapping: dict, canonical_embeddings: dict, threshold: float = 0.8) -> int:
    pred_embedding = get_label_embedding(pred_label)
    best_match = None
    best_score = -1.0

    for canon_label, code in canonical_mapping.items():
        score = util.pytorch_cos_sim(pred_embedding, canonical_embeddings[canon_label]).item()
        if score > best_score:
            best_score = score
            best_match = code

    return best_match if best_score >= threshold else -1

# -------------------------
# Entry point for COCO-style numeric labels
# -------------------------
def standardize_coco_labels(coco_ids, canonical_mapping, canonical_embeddings, threshold: float = 0.8):
    standardized = []
    for coco_id in coco_ids:
        label_name = COCO_ID_TO_NAME.get(int(coco_id), f"unknown({coco_id})")
        std_id = standardize_label_semantic(label_name, canonical_mapping, canonical_embeddings, threshold)
        standardized.append(std_id)
    return standardized


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    canonical_mapping = load_canonical_mapping()
    canonical_embeddings = compute_canonical_embeddings(canonical_mapping)

    coco_ids = [1, 84, 16, 200]  # person, book, bird, invalid
    std_labels = standardize_coco_labels(coco_ids, canonical_mapping, canonical_embeddings, threshold=0.75)

    print("Original COCO IDs:", coco_ids)
    print("Standardized Canonical IDs:", std_labels)