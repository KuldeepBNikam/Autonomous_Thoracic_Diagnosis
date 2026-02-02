import torch
import cv2
import numpy as np
from typing import Dict

from HyperLung_XR.constant import *
from HyperLung_XR.ml.model.arch import HybridCNNTransformer

from HyperLung_XR.explainability.utils import overlay_heatmap
from HyperLung_XR.constant.training_pipeline import PREDICTION_LABEL
from HyperLung_XR.logger import logging
from torchvision import transforms

class Predictor:
    

    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        # Load model
        self.model = HybridCNNTransformer(num_classes=2)
        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        # Unfreeze top CNN layers ONLY for Grad-CAM
        self.model.unfreeze_top_layers()
        self.model.to(device)
        self.model.eval()


    def predict(self, image_tensor: torch.Tensor, original_image: np.ndarray):

        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True

        #  Forward pass
        output, feature_maps = self.model(image_tensor)
        feature_maps.retain_grad()

        temperature = 1.5 #last change
        probs = torch.softmax(output / temperature, dim=1)

        
        confidence, pred_idx = torch.max(probs, dim=1)

        diagnosis = PREDICTION_LABEL[pred_idx.item()]
        confidence = round(min(confidence.item(), 0.98), 2)


        #  Backward pass for Grad-CAM
        self.model.zero_grad()
        score = output[:, pred_idx.item()]
        score.backward(retain_graph=True)


        #  Grad-CAM computation (MANUAL)
        gradients = feature_maps.grad          # (B, C, H, W)

        # Global average pooling on gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of feature maps
        cam = (weights * feature_maps).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize CAM to [0, 1]
        
        cam = cv2.resize(
            cam,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, lung_mask = cv2.threshold(
            gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = np.ones((7, 7), np.uint8)
        lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, kernel)
        cam = cam * lung_mask

        overlay = overlay_heatmap(original_image, cam)

        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 4),
            "heatmap_overlay": overlay
        }

