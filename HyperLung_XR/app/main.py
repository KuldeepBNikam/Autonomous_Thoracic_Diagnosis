from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import io
import os
import base64
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

from HyperLung_XR.pipeline.predictor import Predictor
from HyperLung_XR.ml.reporting.medical_report import MedicalReportGenerator
from HyperLung_XR.ml.reporting.llm_client import LLMClient
from HyperLung_XR.ml.chatbot.chat_service import MedicalChatBot
from HyperLung_XR.constant.training_pipeline import DEVICE
from HyperLung_XR.utils.model_utils import get_latest_model_path
from HyperLung_XR.utils.report_cache import (
    get_report,
    is_report_ready,
    reset_report
)


app = FastAPI(title="Autonomous Thoracic Diagnosis")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


predictor = Predictor(
    model_path=get_latest_model_path(),
    device=DEVICE
)

llm = LLMClient(model="mistral:latest")
report_generator = MedicalReportGenerator(llm)
chatbot = MedicalChatBot()

LAST_PREDICTION_CONTEXT = {}

class ChatRequest(BaseModel):
    messages: list


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    global LAST_PREDICTION_CONTEXT
    LAST_PREDICTION_CONTEXT = {}   
    reset_report()

    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted image")

    image_np = np.array(image)
    image_tensor = predictor.transform(image).unsqueeze(0)

    result = predictor.predict(image_tensor, image_np)

    LAST_PREDICTION_CONTEXT["diagnosis"] = result["diagnosis"]
    LAST_PREDICTION_CONTEXT["confidence"] = result["confidence"]


    success, buffer = cv2.imencode(".png", result["heatmap_overlay"])
    if not success:
        raise RuntimeError("Failed to encode Grad-CAM")

    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    background_tasks.add_task(
    report_generator.generate_report,
    {
        "diagnosis": result["diagnosis"],
        "confidence": result["confidence"],
        "gradcam_summary": (
            "No significant abnormal activation in lung fields"
            if result["diagnosis"] == "NORMAL"
            else "Increased activation in suspected lung regions"
        )
    }
)

    return {
        "diagnosis": result["diagnosis"],
        "confidence": result["confidence"],
        "gradcam_image_base64": gradcam_base64,
        "report": "Report is being generated. Please wait a few seconds."
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        reply = chatbot.reply(
            request.messages,
            context=LAST_PREDICTION_CONTEXT
        )
        return {"reply": reply}
    except Exception:
        return {
            "reply": "The assistant is temporarily unavailable."
        }

@app.get("/report")
async def get_report_endpoint():
    
    if not is_report_ready():
        return {"status": "processing", "report": None}

    return {
        "status": "ready",
        "report": get_report()
    }