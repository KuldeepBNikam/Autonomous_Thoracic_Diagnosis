from typing import List
from HyperLung_XR.ml.chatbot.ollama_client import OllamaClient
from HyperLung_XR.utils.report_cache import get_report, is_report_ready


class MedicalChatBot:
    def __init__(self):
        self.client = OllamaClient(model="mistral:latest")

        self.system_prompt = (
            "You are a medical assistant.\n"
            "Explain findings in simple language.\n"
            "Give general precautions only.\n"
            "Do NOT diagnose or suggest medication.\n"
            "Keep answers short."
        )

    def reply(self, chat_history: List[dict], context: dict | None = None):
    # Trust report state, not fragile globals
        if not is_report_ready():
            return "Please wait until the medical report is ready."

        report = get_report()
        if not report:
            return "Report is still being generated. Please wait."

        context = context or {}
        context_text = (
            f"Diagnosis: {context.get('diagnosis', 'Unknown')}\n"
            f"Confidence: {round(context.get('confidence', 0) * 100, 2)}%\n"
            f"Report Summary: {report.splitlines()[0]}"
        )

        messages = [
            {
                "role": "system",
                "content": self.system_prompt + "\n\n" + context_text
            }
        ]

        if chat_history:
            messages.append(chat_history[-1])

        try:
            return self.client.chat(messages)
        except Exception as e:
            print("Ollama error:", e)
            return "The assistant is temporarily unavailable."