import os
import sys
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO

# Import configuration (ensure config.py contains paths and device settings)
from config import YOLO_CHECKPOINT, DEVICE

# Base model interface
class BaseModel:
    def predict(self, image):
        raise NotImplementedError("Subclasses should implement this method.")

# YOLOv8 segmentation model implementation
class YOLOv8SegmentationModel(BaseModel):
    def __init__(self, model_path=None, device=None):
        # Use provided paths or defaults from config
        self.device = torch.device(device if device else DEVICE)
        model_path = model_path if model_path else str(YOLO_CHECKPOINT)
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def predict(self, image):
        # Run inference on the input image
        results = self.model(image)
        # For a single image, take the first result
        prediction = results[0]

        # Extract bounding boxes, labels, and scores
        boxes = prediction.boxes.xyxy.cpu().numpy()
        labels = prediction.boxes.cls.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()

        # Extract segmentation masks (if available)
        masks = prediction.masks.data.cpu().numpy() if hasattr(prediction, "masks") else None

        return boxes, labels, scores, masks

# (Optional) Factory function to instantiate the correct model type later.
def get_model(model_type="yolo"):
    if model_type.lower() == "yolo":
        return YOLOv8SegmentationModel()
    # elif model_type.lower() == "maskrcnn":
    #     return MaskRCNNModel()  # Placeholder for future model.
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
        from config import DAVIS_RAW_FRAMES_DIR  # Ensure you're using the correct absolute path

        sequence_name = "bike-packing"
        test_image_path = DAVIS_RAW_FRAMES_DIR / sequence_name / "00000.png"

        if not test_image_path.exists():
            print(f"Test image not found at {test_image_path}")
        else:
            model = get_model("yolo")
            test_img = cv2.imread(str(test_image_path))
            if test_img is None:
                print("Failed to load test image (check image integrity and format).")
            else:
                boxes, labels, scores, masks = model.predict(test_img)
                print("Boxes:", boxes)
                print("Labels:", labels)
                print("Scores:", scores)
                # Optionally visualize or process the masks