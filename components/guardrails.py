import logging
import re

from utils import timer

logger = logging.getLogger(__name__)


class Guardrails:
    @staticmethod
    @timer(name="Output validation")
    def validate_output(answer: str) -> tuple:
        if not answer:
            logger.warning("Guardrail triggered: empty answer")
            return False, "Output was empty."

        try:
            banned_phrases = ["I am an AI", "As a language model", "I apologize"]
            for phrase in banned_phrases:
                if phrase.lower() in answer.lower():
                    logger.warning("Guardrail triggered: banned phrase '%s'", phrase)
                    return False, "Output triggered identity guardrail."

            if "Source:" not in answer and "[" not in answer:
                logger.warning("Guardrail triggered: missing citations")
                return False, "Output missing citations."

            logger.info("Guardrails passed")
            return True, answer
        except Exception as e:
            logger.error("Guardrail validation error: %s", e)
            return False, f"Guardrail error: {e}"
