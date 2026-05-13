import logging
import os
from langchain_openai import ChatOpenAI
from config.llm_local import LocalLLM
from utils import timer

logger = logging.getLogger(__name__)

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.3-70b-instruct"


class LLMProvider:
    @staticmethod
    @timer(name="Initialize LLM")
    def get_llm(model_name: str = MODEL_NAME, use_local: bool = True, temperature: float = 0.2):
        if use_local:
            try:
                return LocalLLM.get_local_llm()
            except Exception as e:
                logger.error("Local LLM init failed: %s", e)
                raise

        logger.info("Configuring Nvidia LLM (model=%s)", model_name)
        api_key = os.environ.get("nvidia_api_key")
        if not api_key:
            raise ValueError("nvidia_api_key not found in environment variables")

        try:
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=BASE_URL,
                temperature=temperature,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error("Nvidia LLM init failed: %s", e)
            raise
