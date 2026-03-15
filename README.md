Autonomous Thoracic Diagnosis 🫁

Deep Learning–based system for automatic pneumonia detection from Chest X-ray images with explainable AI visualization and AI-generated medical reports.

This project builds an end-to-end diagnostic pipeline combining computer vision, transformers, and LLMs to simulate a simplified AI-assisted radiology workflow.

The system classifies X-ray images as:

Normal

Pneumonia

It also provides:

Grad-CAM heatmaps for model explainability

LLM-generated radiology reports

Interactive medical chatbot for follow-up queries

Why I Built This

Chest X-ray interpretation requires experience and can be time-consuming.

The goal of this project is to explore how deep learning + explainable AI + LLMs can assist medical professionals by:

Automatically detecting pneumonia

Highlighting suspicious lung regions

Generating structured diagnostic reports

This project is also part of my effort to build a complete AI product pipeline, not just a model.

Project Highlights
Hybrid Deep Learning Model

A custom CNN + Transformer architecture designed for medical imaging.

DenseNet Backbone → captures local lung patterns

Swin Transformer → learns global contextual features

Fusion Layer → combines CNN + transformer features

This combination improves the model's ability to capture both:

fine-grained textures

global structural patterns

Explainable AI

Medical AI must be interpretable.

This project integrates Grad-CAM visualization to highlight areas in the X-ray that influenced the model's prediction.

This helps:

build trust in the model

provide visual diagnostic support

AI Medical Report Generation

After prediction, a Large Language Model generates a structured medical report including:

Observation

Findings

Impression

Recommendations

The system uses a local LLM via Ollama to keep the pipeline lightweight and offline-capable.

Medical Chatbot

An AI chatbot allows users to ask questions such as:

What does pneumonia mean?

What should be the next medical step?

Is the infection severe?

This is powered by the same LLM backend.

System Architecture
Chest X-ray Image
        │
        ▼
Image Preprocessing
(CLAHE + normalization)
        │
        ▼
Hybrid CNN + Swin Transformer Model
        │
        ├── Prediction (Normal / Pneumonia)
        │
        ├── Grad-CAM Heatmap
        │
        ▼
Prediction Data
        │
        ▼
LLM Medical Report Generator
        │
        ▼
Interactive Web Interface
        │
        ├── Diagnosis Result
        ├── Heatmap Visualization
        ├── Generated Medical Report
        └── AI Chatbot
Tech Stack
Deep Learning

PyTorch

TIMM

DenseNet

Swin Transformer

Image Processing

OpenCV

CLAHE enhancement

AI Explainability

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

Dataset

Chest X-ray dataset used for training:

~2.3 GB dataset

Two classes:

Normal

Pneumonia

Typical preprocessing includes:

CLAHE contrast enhancement

resizing

normalization

Project Structure
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
│   ├── components/
│   ├── pipeline/
│   ├── entity/
│   ├── utils/
│   └── logger
│
├── frontend/
│
├── research/
│
├── main.py
├── app.py
└── requirements.txt
Installation

Clone the repository

git clone https://github.com/KuldeepBNikam/Autonomous_Thoracic_Diagnosis.git
cd Autonomous_Thoracic_Diagnosis

Create environment

conda create -p venv python=3.10 -y
conda activate venv

Install dependencies

pip install -r requirements.txt
Running the Project

Start backend server

python app.py

Run frontend

python main.py

Open browser

http://localhost:8000

Upload an X-ray image to receive:

Prediction

Grad-CAM visualization

AI medical report

Chatbot assistance

Example Output

System returns:

Prediction: Pneumonia

Confidence Score

Heatmap highlighting infected lung regions

AI-generated diagnostic report

Future Improvements

Planned upgrades:

Multi-disease detection (TB, COVID-19, fibrosis)

Better Grad-CAM layer selection

DICOM support

Cloud deployment

Real radiologist-style report templates

Model monitoring and drift detection

Disclaimer

This project is for research and educational purposes only.

It should not be used for real medical diagnosis without professional supervision.

Author

Kuldeep Nikam

GitHub
https://github.com/KuldeepBNikam
