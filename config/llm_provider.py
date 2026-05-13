import logging
import os
from langchain_openai import ChatOpenAI

from config.llm_local import LocalLLM
from utils import timer

logger = logging.getLogger(__name__)

BASE_URL="https://integrate.api.nvidia.com/v1"
MODEL_NAME="meta/llama-3.3-70b-instruct"
USE_LOCAL_LLM=True

class LLMProvider:

    @staticmethod
    @timer
    def get_llm(model_name=MODEL_NAME, temperature=0.2):
        if USE_LOCAL_LLM:
            return LocalLLM.get_local_llm()
        else:
            logger.info("Configuring Nvidia LLM.")
            api_key = os.environ.get("nvidia_api_key")
            if not api_key:
                raise ValueError("nvidia_api_key not found in environment variables")

            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=BASE_URL,
                temperature=temperature,
                max_tokens=1024
            )