🫁 Autonomous Thoracic Diagnosis

AI-powered system for detecting pneumonia from chest X-ray images using deep learning, explainable AI, and automated report generation.

The goal of this project is to simulate a mini AI-assisted radiology workflow where a model:

Detects pneumonia from X-ray images

Highlights suspicious lung regions

Generates a medical-style diagnostic report

Allows users to interact with an AI medical chatbot

🖼️ Project Preview
<img src="images/ui-preview.png" width="800">

(Upload X-ray → AI Prediction → Heatmap → Medical Report → Chatbot)

⚙️ What This System Does
🔍 Pneumonia Detection

The model classifies chest X-ray images into:

Normal

Pneumonia

using a hybrid deep learning architecture.

🧠 Hybrid CNN + Transformer Model
<img src="images/model-architecture.png" width="700">

The system combines:

DenseNet (CNN backbone)
captures fine lung textures and abnormalities.

Swin Transformer
captures global spatial relationships in the X-ray.

Both features are fused before the final classification layer.

This hybrid design helps the model learn both:

local radiological patterns

global lung structure

🔥 Explainable AI with Grad-CAM
<img src="images/gradcam-example.png" width="600">

To make the model interpretable, Grad-CAM is used to visualize which regions influenced the prediction.

The heatmap highlights areas in the lungs that contributed most to the decision.

This helps users understand why the model predicted pneumonia.

📝 AI Medical Report Generation
<img src="images/report-example.png" width="700">

Once the model makes a prediction, the system automatically generates a structured medical-style report using an LLM.

The report includes:

Observations

Findings

Impression

Recommendations

The LLM runs locally through Ollama, so the system works offline.

💬 AI Medical Chatbot
<img src="images/chatbot-preview.png" width="600">

Users can ask follow-up questions such as:

What is pneumonia?

Is this condition serious?

What should be the next step?

The chatbot uses the same local LLM backend.

🏗️ System Architecture
<img src="images/system-architecture.png" width="800">

Pipeline overview:

X-ray Image
   │
   ▼
Image Preprocessing
(CLAHE + normalization)
   │
   ▼
Hybrid CNN + Swin Transformer
   │
   ├─ Prediction
   ├─ Confidence Score
   ├─ Grad-CAM Heatmap
   │
   ▼
Prediction Data
   │
   ▼
LLM Medical Report Generator
   │
   ▼
Frontend Interface
   ├ Diagnosis Result
   ├ Heatmap Visualization
   ├ Generated Report
   └ AI Chatbot
🧰 Tech Stack
Deep Learning

PyTorch

TIMM

DenseNet

Swin Transformer

Image Processing

OpenCV

CLAHE enhancement

Explainability

Grad-CAM

LLM Integration

Ollama

Mistral

Backend

Python

FastAPI

Frontend

HTML

CSS

JavaScript

📂 Project Structure
Autonomous_Thoracic_Diagnosis
│
├── artifacts/
│   ├── models
│   ├── evaluation
│   └── predictions
│
├── configs/
│
├── HyperLung_XR/
│   ├── components
│   ├── pipeline
│   ├── entity
│   ├── utils
│   └── logger
│
├── frontend/
├── research/
│
├── app.py
├── main.py
└── requirements.txt
🚀 Running the Project

Clone the repository

git clone https://github.com/KuldeepBNikam/Autonomous_Thoracic_Diagnosis.git
cd Autonomous_Thoracic_Diagnosis

Create environment

conda create -p venv python=3.10 -y
conda activate venv

Install dependencies

pip install -r requirements.txt

Run the backend

python app.py

Run the UI

python main.py

Open in browser

http://localhost:8000

Upload an X-ray image to see the full pipeline in action.

📊 Example Output

The system produces:

✔ Prediction (Normal / Pneumonia)
✔ Confidence score
✔ Grad-CAM heatmap
✔ AI-generated medical report
✔ Interactive chatbot

🔮 Future Improvements

Ideas for improving the system:

Multi-disease classification (TB, COVID-19, fibrosis)

Better Grad-CAM layer tuning

DICOM image support

Cloud deployment

Radiologist-grade reporting templates

Model monitoring & drift detection

⚠️ Disclaimer

This project is intended for educational and research purposes only.

It should not be used for real medical diagnosis without professional medical supervision.

👨‍💻 Author

Kuldeep Nikam

GitHub
https://github.com/KuldeepBNikam
