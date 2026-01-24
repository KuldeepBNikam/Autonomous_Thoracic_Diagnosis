import torch
import cv2
import numpy as np
from typing import Dict

from HyperLung_XR.ml.model.arch import HybridCNNTransformer
from HyperLung_XR.explainability.gradcam import GradCAM
from HyperLung_XR.explainability.utils import overlay_heatmap
from HyperLung_XR.constant.training_pipeline import PREDICTION_LABEL
from HyperLung_XR.logger import logging


class Predictor:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device

        # Load model
        self.model = HybridCNNTransformer(num_classes=2)
        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.model.to(device)
        self.model.eval()

        # Grad-CAM target layer (DenseNet last conv)
        self.target_layer = self.model.cnn[-1]
        self.gradcam = GradCAM(self.model, self.target_layer)

    def predict(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray
    ) -> Dict:
        """
        image_tensor: (1, 3, 224, 224) torch.Tensor
        original_image: (H, W, 3) numpy array (BGR or RGB)
        """

        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)

            output = self.model(image_tensor)

            probs = torch.softmax(output, dim=1)
            confidence, prediction_idx = torch.max(probs, dim=1)

        # Convert prediction
        prediction_idx = prediction_idx.item()
        confidence = confidence.item()

        diagnosis = PREDICTION_LABEL[prediction_idx]

        # Generate Grad-CAM (NO torch.no_grad here)
        cam = self.gradcam.generate(
            image_tensor=image_tensor,
            class_idx=prediction_idx
        )

        overlay = overlay_heatmap(original_image, cam)

        logging.info(
            f"Prediction: {diagnosis}, Confidence: {confidence:.4f}"
        )

        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 4),
            "heatmap_overlay": overlay
        }
