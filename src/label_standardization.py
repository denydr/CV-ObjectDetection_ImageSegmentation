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


# Load a pretrained SentenceTransformer model.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_label_embedding(label: str) -> torch.Tensor:
    """
    Encodes a label string into an embedding.
    """
    return embedding_model.encode(label, convert_to_tensor=True)


def standardize_label_semantic(pred_label: str, canonical_mapping: dict, threshold: float = 0.8) -> int:
    """
    Standardizes a predicted label by comparing its semantic embedding with those of the canonical labels.
    If the highest cosine similarity score exceeds the threshold, returns the canonical integer code;
    otherwise, returns -1.
    """
    # Get the embedding of the predicted label.
    pred_embedding = get_label_embedding(pred_label)

    best_match = None
    best_score = -1.0

    # Compare against each canonical label.
    for canon_label, code in canonical_mapping.items():
        canon_embedding = get_label_embedding(canon_label)
        score = util.pytorch_cos_sim(pred_embedding, canon_embedding).item()
        if score > best_score:
            best_score = score
            best_match = code  # Store the canonical code

    if best_score >= threshold:
        return best_match
    else:
        print(
            f"Warning: Predicted label '{pred_label}' did not match any canonical label (best score: {best_score:.2f}).")
        return -1


def standardize_labels_semantic(pred_labels, canonical_mapping, threshold: float = 0.8):
    """
    Standardizes a list of predicted labels using semantic similarity.
    """
    standardized = []
    for label in pred_labels:
        # Ensure the label is a string.
        if not isinstance(label, str):
            label = str(label)
        std_code = standardize_label_semantic(label, canonical_mapping, threshold)
        standardized.append(std_code)
    return standardized


# Example usage:
if __name__ == "__main__":
    # Load the canonical mapping using the path from config.py.
    canonical_mapping = load_canonical_mapping()
    print("Canonical Mapping:")
    for key, value in canonical_mapping.items():
        print(f"  {key}: {value}")

    # Example predicted labels (these might come from your model's predictions)
    predicted_labels = ["puppy", "kitten", "avian", "human"]

    # Standardize the predicted labels using semantic similarity.
    std_labels = standardize_labels_semantic(predicted_labels, canonical_mapping, threshold=0.75)
    print("Predicted labels:", predicted_labels)
    print("Standardized labels:", std_labels)