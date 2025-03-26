# Evaluation Pipeline Configuration Guide

This guide outlines the steps to set up and configure your evaluation pipeline for object detection and segmentation. Follow the steps below to structure your scripts and ensure that all components work together seamlessly.

---

## 1. Configuration & Setup

### config.py
- **Purpose:** Store overall settings such as paths, IoU thresholds, device settings, etc.
- **Contents:**
  - Absolute paths to:
    - Ground‑truth JSON files
    - Masks
    - Raw image sequences
  - Other parameters (e.g., evaluation thresholds)

### Canonical Mapping
- **Purpose:** Define a canonical mapping for category labels.
- **Details:** Ensures that ground‑truth labels and model outputs are compared on the same basis.

---

## 2. Data Loading

### data_loader.py
- **Purpose:** Provide functions to load:
  - Raw images
  - Ground‑truth bounding boxes (and labels) from the representative JSON
  - Ground‑truth masks from the `representative_dataset_masks` directory

### Integration
- **Details:**  
  In your evaluation script, import the `data_loader` module to load the sequences from your representative dataset. Iterate over each sequence and its frames.

---

## 3. Model Handling

### model_handler.py
- **Purpose:** Contains the YOLO configuration and inference code.

### Extension for Multiple Approaches
- **Details:**  
  Design the interface to later add two more state‑of‑the‑art (SoA) approaches. This could involve additional classes or functions that follow the same API.

### Usage in Evaluation
- **Details:**  
  Import `model_handler` in your evaluation script and instantiate the YOLO model to perform inference on each frame.

---

## 4. Evaluation Pipeline – Chronological Steps

### a. Load Configuration & Mapping
- **Step:**  
  Import your configuration settings and load the canonical mapping (either defined directly in `config.py` or loaded from a JSON file).

### b. Select Sequences
- **Step:**  
  Use your representative dataset JSON (e.g., `/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation/output/representative_dataset_boundingboxes_labels.json`) to determine which sequences and frames to evaluate.

### c. Data Preparation
- **Steps:**
  - For each sequence, load:
    - **Raw Frames:** From the dataset (e.g., from the DAVIS dataset at `datasets/DAVIS_dataset/PNGImages/...`).
    - **Ground‑Truth Annotations:** Bounding boxes and labels along with ground‑truth masks from your output directories.
  - **Note:** Ensure that images and masks are loaded with the correct flags (e.g., using `cv2.IMREAD_UNCHANGED` for masks) to avoid any loss of information.

### d. Model Inference
- **Steps:**
  - For each frame in a sequence:
    - Pass the raw frame into your YOLO model (via `model_handler.py`) to obtain predictions for:
      - Bounding boxes
      - Labels
      - Segmentation masks
    - Use your canonical mapping to standardize the predicted labels (post‑processing) so that they match your ground‑truth format.

### e. Matching & Filtering
- **For Segmentation Masks:**
  - Match predicted masks to ground‑truth masks using an IoU threshold.
  - Filter out predictions that do not sufficiently overlap with any ground‑truth object.
- **For Bounding Boxes + Labels:**
  - Match predicted boxes to ground‑truth boxes (e.g., via IoU).
  - Compare the standardized labels.

### f. Metric Computation
- **Validity:**
  - Compute pixel‑wise IoU and Dice coefficient for segmentation masks.
  - Compute detection metrics (e.g., mAP, precision, recall, and label accuracy) for bounding boxes and labels.
- **Reliability:**
  - For video sequences, compute temporal consistency by analyzing the variation (e.g., standard deviation, variance) of these metrics across frames.
- **Objectivity:**
  - Use standardized metrics and labeling to ensure that the evaluation remains objective.

### g. Result Aggregation & Saving
- **Steps:**
  - Aggregate the computed metrics per sequence and across the entire dataset.
  - Save the aggregated results (e.g., as a JSON or CSV file) for later use and comparison.

### h. Visualization (Optional)
- **Step:**  
  Optionally, generate annotated videos or overlay images (raw frame with predicted and ground‑truth boxes/masks) to visually inspect the quality of predictions.

---

## 5. Extensibility for Multiple SoA Approaches

- **Modular Design:**
  - Structure the model inference part to be modular.
  - Define an abstract interface (or a consistent function signature) that all state‑of‑the‑art models will adhere to.
- **Comparative Evaluation:**
  - Loop over the different approaches and compare their metrics side by side.

---

## 6. Final Script Integration

### evaluate.py
- **Purpose:** Main evaluation script that ties together the components.
- **Details:**
  - Import `config`, `data_loader`, and `model_handler` modules.
  - Follow the sequential steps:
    1. Load configuration and canonical mappings.
    2. Load representative dataset annotations and raw frames using `data_loader`.
    3. Run model inference (currently using YOLO, with room for extensions) via `model_handler`.
    4. Match predictions to ground truth (for both masks and bounding boxes + labels) using IoU-based filtering.
    5. Compute evaluation metrics (IoU, Dice, mAP, precision, recall, label accuracy, and temporal consistency).
    6. Aggregate, visualize, and save evaluation results.

---

## Summary of Steps

1. **Load configuration and canonical mappings.**
2. **Load the representative dataset annotations and raw frames** using `data_loader`.
3. **Run model inference** (currently YOLO, with room for extensions) via `model_handler`.
4. **Match predictions to ground truth** (for both masks and bounding boxes + labels) using IoU-based filtering.
5. **Compute evaluation metrics** (IoU, Dice, mAP, precision, recall, label accuracy, temporal consistency).
6. **Aggregate, visualize, and save evaluation results.**
