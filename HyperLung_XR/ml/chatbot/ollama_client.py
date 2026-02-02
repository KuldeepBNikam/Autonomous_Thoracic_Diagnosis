import requests
import threading

_ollama_lock = threading.Lock()

class OllamaClient:
    def __init__(self, model="mistral:latest"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        self.session = requests.Session()

    def chat(self, messages):
        # Convert chat → single prompt (best practice for generate)
        prompt = ""
        for m in messages:
            prompt += f"{m['role'].upper()}: {m['content']}\n"

        with _ollama_lock:
            response = self.session.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 200
                    }
                },
                timeout=60
            )

            response.raise_for_status()
            return response.json()["response"]
