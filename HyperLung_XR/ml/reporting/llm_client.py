import requests
import random
import threading

_ollama_lock = threading.Lock()


class LLMClient:
    def __init__(self, model="mistral:latest"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        self.session = requests.Session()

        self.fallback_templates = {
            "normal": [
                "Lung fields appear clear without focal consolidation.",
                "No significant pulmonary abnormality is identified."
            ],
            "pneumonia": [
                "Patchy pulmonary opacities may be present.",
                "Ill-defined increased lung opacities are noted."
            ]
        }


    def generate(self, prompt: str) -> str:
        try:
            with _ollama_lock:
                response = self.session.post(
                    self.url,
                    json={
                        "model": self.model,
                        "prompt": (
                            "You are a radiology report assistant. "
                            "Be concise, cautious, and clinical.\n\n"
                            + prompt
                        ),
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 250
                        }
                    },
                    timeout=90
                )

            response.raise_for_status()
            return response.json()["response"]

        except Exception:
            diagnosis = "normal"
            if "pneumonia" in prompt.lower():
                diagnosis = "pneumonia"

            findings = random.choice(self.fallback_templates[diagnosis])
            return (
                "Findings\n"
                f"{findings}\n\n"
                "Impression\n"
                "No definitive radiographic evidence of acute pneumonia.\n\n"
                "Severity Assessment\n"
                "Mild severity.\n\n"
                "Recommendation\n"
                "Clinical correlation is advised.\n\n"
                "Disclaimer\n"
                "AI-generated clinical decision support only."
            )
