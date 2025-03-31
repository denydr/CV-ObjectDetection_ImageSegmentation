import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import urllib.request
from pathlib import Path
from ultralytics import YOLO

from config import (
    YOLO_CHECKPOINT,
    MASKRCNN_MODEL_PATH,
    MASKRCNN_BACKBONE_PATH,
    YOLO_DETECTION_PATH,
    DEEPLAB_PATH,
    DEEPLAB_DIR,
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
        image_tensor = to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)[0]

        boxes = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        masks = (outputs["masks"].squeeze(1).cpu().numpy() > 0.5).astype(np.uint8) if "masks" in outputs else None
        return boxes, labels, scores, masks


# -------------------------
# Unified YOLOv8m + DeepLabV3 Pipeline
# -------------------------
class YOLOv8_DeepLabV3_PipelineModel(BaseModel):
    def __init__(self, device=None):
        self.device = torch.device(device if device else DEVICE)

        # -----------------------
        # Load YOLOv8m for detection (boxes + labels)
        # -----------------------
        YOLO_DETECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not YOLO_DETECTION_PATH.exists():
            print("📥 Downloading YOLOv8m detection model...")
            yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt"
            urllib.request.urlretrieve(yolo_url, YOLO_DETECTION_PATH)
            print(f"✅ YOLOv8m saved to {YOLO_DETECTION_PATH}")
        self.yolo = YOLO(str(YOLO_DETECTION_PATH))
        self.yolo.to(self.device)
        self.names = self.yolo.model.names

        # -----------------------
        # Load DeepLabV3 for segmentation (masks)
        # -----------------------
        DEEPLAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not DEEPLAB_PATH.exists():
            print("📥 Downloading DeepLabV3 weights...")
            deeplab_url = "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth"
            urllib.request.urlretrieve(deeplab_url, DEEPLAB_PATH)
            print(f"✅ DeepLabV3 weights saved to {DEEPLAB_PATH}")

        self.deeplab = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        state_dict = torch.load(DEEPLAB_PATH, map_location=self.device)
        self.deeplab.load_state_dict(state_dict)
        self.deeplab.to(self.device).eval()

    def predict(self, image):
        # ------------------ Detection with YOLO ------------------
        yolo_results = self.yolo(image)
        prediction = yolo_results[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        labels = prediction.boxes.cls.cpu().numpy().astype(int)
        scores = prediction.boxes.conf.cpu().numpy()

        # ------------------ Instance Segmentation with DeepLabV3 ------------------
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                masks.append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
                continue

            crop_tensor = to_tensor(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.deeplab(crop_tensor)['out']
            pred_mask = output.squeeze(0).argmax(0).cpu().numpy()

            # Resize and place mask back in original image size
            full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            resized_mask = cv2.resize(pred_mask.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            full_mask[y1:y2, x1:x2] = resized_mask
            masks.append(full_mask)

        masks = np.stack(masks) if masks else None
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
    elif model_type == "yolo_deeplab":
        return YOLOv8_DeepLabV3_PipelineModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# -------------------------
# Run test for model loading
# -------------------------
if __name__ == "__main__":
    print("🔍 Loading YOLO-DeepLab model for testing...")
    model = get_model("yolo_deeplab")
    print("✅ YOLO_DEEPLAB model loaded and ready.")
