import os
from pathlib import Path

# PROJECT ROOT (adjust if necessary)
PROJECT_ROOT = Path("/Users/dd/PycharmProjects/CV-ObjectDetection_ImageSegmentation").resolve()

# Data Directories
DATASETS_DIR = PROJECT_ROOT / "datasets"
DAVIS_RAW_FRAMES_DIR = DATASETS_DIR / "DAVIS_dataset" / "representative_dataset_PNGImages" / "480p"  # Raw image sequences
# Raw RGB masks for the representative dataset (annotations in original format)
RAW_MASKS_DIR = DATASETS_DIR / "DAVIS_dataset" / "representative_dataset_Annotations" / "480p"

# Output Directories
OUTPUT_DIR = PROJECT_ROOT / "output"
REP_BBOX_JSON = OUTPUT_DIR / "representative_dataset_boundingboxes_labels.json"

# Converted masks (for evaluation)
# Multi-object sequences
REP_MASKS_MULTI = OUTPUT_DIR / "representative_dataset_masks" / "multi_object"
# Single-object sequences
REP_MASKS_SINGLE = OUTPUT_DIR / "representative_dataset_masks" / "single_object"

# Canonical Mapping file path (for standardizing category labels)
CANONICAL_MAPPING_PATH = PROJECT_ROOT / "canonical_label_mapping.json"

# Checkpoints (for model weights)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# YOLOv8 segmentation model
YOLO_CHECKPOINT = CHECKPOINT_DIR / "yolov8n-seg.pt"

# Mask R-CNN paths
MASKRCNN_DIR = CHECKPOINT_DIR / "mask_rcnn"
MASKRCNN_MODEL_PATH = MASKRCNN_DIR / "maskrcnn_resnet50_fpn_coco.pth"
MASKRCNN_BACKBONE_PATH = MASKRCNN_DIR / "resnet50.pth"

# YOLOv8m detection model
YOLO_DETECTION_PATH = CHECKPOINT_DIR / "yolov8m.pt"

# DeepLabV3 segmentation model
DEEPLAB_DIR = CHECKPOINT_DIR / "deeplabv3"
DEEPLAB_PATH = DEEPLAB_DIR / "deeplabv3_resnet101_coco.pth"

# Evaluation Parameters
IOU_THRESHOLD = 0.5         # IoU threshold for matching predictions to ground truth
CONFIDENCE_THRESHOLD = 0.5  # Threshold for detection confidence (if applicable)
MAX_INSTANCES = 5
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Other evaluation configuration (you can add more as needed)
METRICS = {
    "iou_threshold": IOU_THRESHOLD,
    "confidence_threshold": CONFIDENCE_THRESHOLD
}

# ---------------------------------------------------------------------
# Additional Paths for GT annotations and predictions
# GT Annotations:
GT_MASKS_DIR = PROJECT_ROOT / "metrics_artifacts" / "gt_annotations" / "gt_masks"
GT_JSONS_DIR = PROJECT_ROOT / "metrics_artifacts" / "gt_annotations" / "gt_JSONs"

# Predicted Masks & JSONs (Preprocessed):
PREDICTED_BASE_DIR = PROJECT_ROOT / "metrics_artifacts" / "predictions" / "predicted_masks_preprocessed"

# For YOLO:
YOLO_PREDICTED_MASKS_DIR = PREDICTED_BASE_DIR / "yolo" / "yolo_predicted_masks"
YOLO_PREDICTED_JSONS_DIR = PREDICTED_BASE_DIR / "yolo" / "yolo_predicted_JSONs"

# For Mask R-CNN:
MASKRCNN_PREDICTED_MASKS_DIR = PREDICTED_BASE_DIR / "maskrcnn" / "maskrcnn_predicted_masks"
MASKRCNN_PREDICTED_JSONS_DIR = PREDICTED_BASE_DIR / "maskrcnn" / "maskrcnn_predicted_JSONs"

# For YOLO+DeepLab:
YOLO_DEEPLAB_PREDICTED_MASKS_DIR = PREDICTED_BASE_DIR / "yolo_deeplab" / "yolo_deeplab_predicted_masks"
YOLO_DEEPLAB_PREDICTED_JSONS_DIR = PREDICTED_BASE_DIR / "yolo_deeplab" / "yolo_deeplab_predicted_JSONs"

# Print configuration summary
print("Configuration loaded:")
print(f"  DAVIS raw frames: {DAVIS_RAW_FRAMES_DIR}")

# Print configuration summary
print("Configuration loaded:")
print(f"  DAVIS raw frames: {DAVIS_RAW_FRAMES_DIR}")
print(f"  Raw representative masks: {RAW_MASKS_DIR}")
print(f"  Representative BBox JSON: {REP_BBOX_JSON}")
print(f"  Converted Multi-Object Masks: {REP_MASKS_MULTI}")
print(f"  Converted Single-Object Masks: {REP_MASKS_SINGLE}")
print(f"  Canonical mapping: {CANONICAL_MAPPING_PATH}")
print(f"  YOLO checkpoint: {YOLO_CHECKPOINT}")
print(f"  YOLOv8m detection path: {YOLO_DETECTION_PATH}")
print(f"  DeepLabV3 weights path: {DEEPLAB_PATH}")
print(f"  MaskRCNN model path: {MASKRCNN_MODEL_PATH} ")
print(f"  MaskRCNN backbone path:{MASKRCNN_BACKBONE_PATH}")
print(f"  Device: {DEVICE}")

print("Additional Paths:")
print(f"  GT Masks: {GT_MASKS_DIR}")
print(f"  GT JSONs: {GT_JSONS_DIR}")
print(f"  YOLO Predicted Masks: {YOLO_PREDICTED_MASKS_DIR}")
print(f"  YOLO Predicted JSONs: {YOLO_PREDICTED_JSONS_DIR}")
print(f"  MASKRCNN Predicted Masks: {MASKRCNN_PREDICTED_MASKS_DIR}")
print(f"  MASKRCNN Predicted JSONs: {MASKRCNN_PREDICTED_JSONS_DIR}")
print(f"  YOLO-Deeplab Predicted Masks: {YOLO_DEEPLAB_PREDICTED_MASKS_DIR}")
print(f"  YOLO-Deeplab Predicted JSONs: {YOLO_DEEPLAB_PREDICTED_JSONS_DIR}")