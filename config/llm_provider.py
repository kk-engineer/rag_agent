import logging
import os
from langchain_openai import ChatOpenAI
from utils import timer

logger = logging.getLogger(__name__)


class LLMProvider:
    @staticmethod
    @timer
    def get_nvidia_llm(model_name="meta/llama-3.3-70b-instruct", temperature=0.2):
        logger.info("Configuring NVIDIA LLM: %s (temperature=%.1f)", model_name, temperature)
        api_key = os.environ.get("nvidia_api_key")
        if not api_key:
            raise ValueError("nvidia_api_key not found in environment variables")

        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",
            temperature=temperature,
            max_tokens=1024
        )