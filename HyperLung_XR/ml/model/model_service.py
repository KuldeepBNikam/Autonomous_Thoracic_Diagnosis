import io
import cv2
import base64
import numpy as np
import torch
from PIL import Image as PILImage
import bentoml

from bentoml.io import Image, Text, JSON

from HyperLung_XR.explainability.gradcam import GradCAM
from HyperLung_XR.explainability.utils import overlay_heatmap
from HyperLung_XR.ml.reporting.medical_report import MedicalReportGenerator
from HyperLung_XR.ml.reporting.llm_client import LLMClient

from HyperLung_XR.constant.training_pipeline import *
from HyperLung_XR.ml.model.arch import HybridCNNTransformer


bento_model = bentoml.pytorch.get(BENTOML_MODEL_NAME)

model = HybridCNNTransformer(num_classes=2)
state_dict = bento_model.load_model()
model.load_state_dict(state_dict)
model.eval()

runner = bento_model.to_runner()

svc = bentoml.Service(name=BENTOML_SERVICE_NAME, runners=[runner])

# Initialize Grad-CAM (DenseNet last conv layer)

target_layer = model.cnn[-1]
gradcam = GradCAM(model, target_layer)

# Initialize LLM
llm_client = LLMClient(openai_client=None)  # plug real client later
report_generator = MedicalReportGenerator(llm_client)


@svc.api(input=Image(allowed_mime_types=["image/jpeg"]), output=JSON())
async def predict(img):
    # Convert image to bytes
    buffer = io.BytesIO()
    img.save(buffer, "jpeg")
    image_bytes = buffer.getvalue()

    # Load transforms
    transforms = bento_model.custom_objects.get(TRAIN_TRANSFORMS_KEY)

    # PIL image
    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transforms(image).unsqueeze(0).cpu()

    # --------------------
    # MODEL PREDICTION
    # --------------------
    output = await runner.async_run(image_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, pred_idx = torch.max(probs, dim=1)

    diagnosis = PREDICTION_LABEL[pred_idx.item()]
    confidence = round(confidence.item(), 4)

    # --------------------
    # GRAD-CAM
    # --------------------
    cam = gradcam.generate(
        input_tensor=image_tensor,
        class_idx=pred_idx.item()
    )

    original_np = np.array(image)
    overlay = overlay_heatmap(original_np, cam)

    _, buffer = cv2.imencode(".png", overlay)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    # --------------------
    # LLM MEDICAL REPORT
    # --------------------
    report = report_generator.generate_report({
        "diagnosis": diagnosis,
        "confidence": confidence
    })

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "gradcam": heatmap_base64,
        "report": report
    }