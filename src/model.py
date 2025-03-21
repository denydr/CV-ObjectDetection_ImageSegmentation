import os
import sys

import cv2
import numpy as np
from pathlib import Path
import torch
import torchvision.models.detection as models

# YOLOv8
from ultralytics import YOLO

# Mask R-CNN
import torchvision
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# SAM (Segment Anything Model)
from segment_anything import sam_model_registry, SamPredictor

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

checkpoint_dir = Path(os.path.join(project_root, "checkpoints"))

checkpoint_path_sam_vit = Path(os.path.join(project_root, checkpoint_dir, "sam_vit_h_4b8939.pth"))
check_maskrcnn_resnet50_fpn = Path(os.path.join(project_root, checkpoint_dir, "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"))


class YOLOv8Model:
    def __init__(self, model_path="yolov8n.pt", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def predict(self, image):
        results = self.model(image)
        return results


class MaskRCNNModel:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Tell PyTorch where to look for the resnet backbone
        os.environ['TORCH_HOME'] = str(checkpoint_dir)
        # Path to specific Mask R-CNN checkpoint file
        maskrcnn_path = checkpoint_dir / "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"  # Adjust filename if different
        resnet_path = checkpoint_dir / "resnet50-0676ba61.pth"

        # 💡 Avoid downloading anything by disabling all weights
        self.model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

        # Load the saved weights (if available)
        if maskrcnn_path.exists():
            checkpoint = torch.load(maskrcnn_path, map_location=self.device)  # Load from local file
            self.model.load_state_dict(checkpoint)  # Load weights into the model
            print(f"✅ Loaded Mask R-CNN weights from: {maskrcnn_path}")
        else:
            print(f"⚠️ Warning: No checkpoint found at {maskrcnn_path}. Using default untrained model.")

        # Manually load backbone if needed
        if resnet_path.exists():
            try:
                backbone_weights = torch.load(resnet_path, map_location=self.device)
                self.model.backbone.body.load_state_dict(backbone_weights, strict=False)
                print(f"✅ Loaded ResNet-50 backbone from: {resnet_path}")
            except Exception as e:
                print(f"⚠️ Failed to load backbone weights: {e}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        return outputs


class SAMModel:
    def __init__(self, model_type="vit_h", checkpoint_path=checkpoint_path_sam_vit, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Ensure checkpoint path is a string and exists
        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided for SAMModel.")

        checkpoint_path_str = str(Path(checkpoint_path))

        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device)
        self.predictor = SamPredictor(self.model)
        self.model.eval()

    def predict(self, image):
        self.predictor.set_image(image)
        return self.predictor


class HybridSAMYOLOModel:
    def __init__(self, sam_checkpoint=checkpoint_path_sam_vit, yolo_model="yolov8n.pt", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sam = SAMModel(checkpoint_path=sam_checkpoint, device=device)
        self.yolo = YOLOv8Model(model_path=yolo_model, device=device)

    def predict(self, image):
        yolo_results = self.yolo.predict(image)
        sam_results = self.sam.predict(image)
        return yolo_results, sam_results