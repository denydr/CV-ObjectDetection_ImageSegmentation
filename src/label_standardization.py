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

