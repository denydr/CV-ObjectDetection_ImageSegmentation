import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pathlib import Path
from ultralytics import YOLO

from config import (
    YOLO_CHECKPOINT,
    MASKRCNN_MODEL_PATH,
    MASKRCNN_BACKBONE_PATH,
    DEVICE
)

# -------------------------
# Base class
# -------------------------
class BaseModel:
    def predict(self, image):
        raise NotImplementedError("Subclasses should implement this method.")

# -------------------------
# YOLOv8 Segmentation
# -------------------------
class YOLOv8SegmentationModel(BaseModel):
    def __init__(self, model_path=None, device=None):
        self.device = torch.device(device if device else DEVICE)
        model_path = model_path if model_path else str(YOLO_CHECKPOINT)
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def predict(self, image):
        results = self.model(image)
        prediction = results[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        labels = prediction.boxes.cls.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()

        masks = (prediction.masks.data.cpu().numpy() > 0.5).astype(np.uint8) if hasattr(prediction, "masks") and prediction.masks is not None else None
        return boxes, labels, scores, masks

# -------------------------
# Mask R-CNN
# -------------------------
class MaskRCNNModel(BaseModel):
    def __init__(self, device=None):
        self.device = torch.device(device if device else DEVICE)

        # Ensure directory exists
        MASKRCNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Download and save Mask R-CNN model weights if not found
        if not MASKRCNN_MODEL_PATH.exists():
            print("📥 Downloading Mask R-CNN weights...")
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
            weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            model_pretrained = maskrcnn_resnet50_fpn(weights=weights)
            torch.save(model_pretrained.state_dict(), MASKRCNN_MODEL_PATH)
            print(f"✅ Saved Mask R-CNN weights to {MASKRCNN_MODEL_PATH}")

        # Download and save ResNet-50 backbone separately (optional)
        if not MASKRCNN_BACKBONE_PATH.exists():
            print("📥 Downloading ResNet-50 backbone...")
            backbone_weights = torchvision.models.resnet50(pretrained=True).state_dict()
            torch.save(backbone_weights, MASKRCNN_BACKBONE_PATH)
            print(f"✅ Saved ResNet-50 backbone to {MASKRCNN_BACKBONE_PATH}")

        # Load the model with weights from file
        self.model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        self.model.load_state_dict(torch.load(MASKRCNN_MODEL_PATH, map_location=self.device))
        self.model.to(self.device).eval()

    def predict(self, image):
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)[0]

        boxes = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        masks = (outputs["masks"].squeeze(1).cpu().numpy() > 0.5).astype(np.uint8) if "masks" in outputs else None
        return boxes, labels, scores, masks


# -------------------------
# Factory function
# -------------------------
def get_model(model_type="yolo"):
    model_type = model_type.lower()
    if model_type == "yolo":
        return YOLOv8SegmentationModel()
    elif model_type == "maskrcnn":
        return MaskRCNNModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# -------------------------
# Run test for model loading
# -------------------------
# if __name__ == "__main__":
#     print("🔍 Loading Mask R-CNN model for testing...")
#     model = get_model("maskrcnn")
#     print("✅ Mask R-CNN model loaded and ready.")
if __name__ == "__main__":
    print("🔍 Testing Mask2Former auto-download and loading...")
    model = get_model("mask2former")
    print("✅ Mask2Former is ready for inference.")