from typing import Dict
from HyperLung_XR.logger import logging


class MedicalReportGenerator:
    def __init__(self, llm_client):
        """
        llm_client: OpenAI / Azure / local LLM wrapper
        """
        self.llm = llm_client

    def generate_report(self, prediction_data: Dict) -> str:
        """
        prediction_data example:
        {
            "diagnosis": "Pneumonia",
            "confidence": 0.96
        }
        """

        prompt = self._build_prompt(prediction_data)

        logging.info("Generating medical report using LLM")

        response = self.llm.generate(prompt)

        return response

    def _build_prompt(self, prediction_data: Dict) -> str:
        diagnosis = prediction_data["diagnosis"]
        confidence = prediction_data["confidence"]

        prompt = f"""
            You are an AI clinical assistant supporting a radiologist.

            Based on the following AI analysis of a chest X-ray, generate a structured medical report.

            AI Findings:
            - Predicted condition: {diagnosis}
            - Confidence score: {confidence:.2f}

            Write the report using the following sections ONLY:
            1. Findings
            2. Impression
            3. Severity Assessment
            4. Recommendation
            5. Disclaimer

            Guidelines:
            - Use cautious, clinical language.
            - Do NOT make definitive diagnoses.
            - Do NOT mention AI, model, Grad-CAM, or algorithms.
            - Assume this is a decision-support tool.
            - Keep it concise and professional.
            """
        return prompt.strip()
