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
MAX_INSTANCES = 15
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Other evaluation configuration (you can add more as needed)
METRICS = {
    "iou_threshold": IOU_THRESHOLD,
    "confidence_threshold": CONFIDENCE_THRESHOLD
}

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