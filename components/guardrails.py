import re


class Guardrails:
    @staticmethod
    def validate_output(answer: str):
        # Basic check: Prevent the model from apologizing or mentioning it's an AI
        banned_phrases = ["I am an AI", "As a language model", "I apologize"]
        for phrase in banned_phrases:
            if phrase.lower() in answer.lower():
                return False, "Output triggered identity guardrail."

        # Check for hallucination markers or empty citations
        if "Source:" not in answer and "[" not in answer:
            return False, "Output missing citations."

        return True, answer