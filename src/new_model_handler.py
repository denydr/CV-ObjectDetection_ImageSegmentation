import os
import sys
import cv2
import numpy as np
from pathlib import Path
import torch

# YOLOv8
from ultralytics import YOLO

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

checkpoint_dir = Path(os.path.join(project_root, "checkpoints"))

checkpoint_path_yolo8n_seg = Path(os.path.join(project_root, checkpoint_dir, "yolov8n-seg.pt"))

class YOLOv8SegmentationModel:
    def __init__(self, model_path="yolov8n-seg.pt", device="cuda"):
        # Use a segmentation-capable model file like "yolov8n-seg.pt"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def predict(self, image):
        # Run inference on the input image
        results = self.model(image)
        # For a single image, the first result contains the predictions
        prediction = results[0]

        # Extract bounding boxes (xyxy format), class labels, and confidence scores
        boxes = prediction.boxes.xyxy.cpu().numpy()
        labels = prediction.boxes.cls.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()

        # Extract segmentation masks if available (each mask as a binary mask or probabilities)
        masks = prediction.masks.data.cpu().numpy() if hasattr(prediction, "masks") else None

        return boxes, labels, scores, masks