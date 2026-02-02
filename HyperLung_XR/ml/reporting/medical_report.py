from typing import Dict
from HyperLung_XR.logger import logging
from HyperLung_XR.utils.report_cache import save_report


class MedicalReportGenerator:
    def __init__(self, llm_client):
        """
        llm_client: OpenAI / Azure / local LLM wrapper
        """
        self.llm = llm_client

    def generate_report(self, prediction_data: Dict) -> str:
        logging.info(" Report generation started")
        prompt = self._build_prompt(prediction_data)

        try:
            response = self.llm.generate(prompt)
            save_report(response)
            return response

        except Exception as e:
            logging.error(f"LLM failed, retrying once: {e}")

            try:
                #  Retry ONCE after short delay
                import time
                time.sleep(2)

                response = self.llm.generate(prompt)
                logging.info(" Saving report to cache")

                save_report(response)
                return response

            except Exception:
                diagnosis = prediction_data["diagnosis"]

                if diagnosis.lower() == "normal":
                    fallback = (
                        "Findings\n"
                        "Lung fields appear clear with no focal consolidation.\n\n"
                        "Impression\n"
                        "No radiographic evidence of acute pneumonia.\n\n"
                        "Severity Assessment\n"
                        "No significant abnormality detected.\n\n"
                        "Recommendation\n"
                        "Routine clinical correlation advised if symptoms persist.\n\n"
                        "Disclaimer\n"
                        "This is an AI-generated clinical decision support report."
                    )
                else:
                    fallback = (
                        "Findings\n"
                        "Patchy pulmonary opacities may be present.\n\n"
                        "Impression\n"
                        "Findings are suggestive but not diagnostic of pneumonia.\n\n"
                        "Severity Assessment\n"
                        "Moderate severity based on imaging features.\n\n"
                        "Recommendation\n"
                        "Clinical correlation and follow-up advised.\n\n"
                        "Disclaimer\n"
                        "This is an AI-generated clinical decision support report."
                    )

                save_report(fallback)
                return fallback



    def _build_prompt(self, prediction_data: Dict) -> str:
        diagnosis = prediction_data["diagnosis"]
        confidence = prediction_data["confidence"]

        if confidence >= 0.9:
            confidence_band = "high confidence"
        elif confidence >= 0.75:
            confidence_band = "moderate confidence"
        else:
            confidence_band = "low confidence"

        if diagnosis.lower() == "pneumonia":
            severity_hint = "possible inflammatory or infectious lung involvement"
        else:
            severity_hint = "no obvious acute pulmonary abnormality"

        prompt = f"""
        Generate a concise chest X-ray report.

        Finding: {diagnosis}
        Confidence: {confidence_band} ({confidence:.2f})

        Context: {severity_hint}
        Diagnosis Constraint:
        - If finding is NORMAL, do NOT mention opacities, infiltrates, pneumonia, or pathology.

        Rules:
        - Use cautious radiology language
        - No definitive diagnosis
        - Do not mention AI or automation

        Sections:
        Findings
        Impression
        Severity Assessment
        Recommendation
        Disclaimer
        """
        return prompt.strip()

