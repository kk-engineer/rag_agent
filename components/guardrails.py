import logging
import re
from utils import timer

logger = logging.getLogger(__name__)


class Guardrails:
    @staticmethod
    @timer
    def validate_output(answer: str):
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