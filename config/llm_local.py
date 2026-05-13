import logging

from langchain_openai import ChatOpenAI
from openai import APIConnectionError, OpenAI

from utils import timer

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "mistral-nemo-12b.gguf"


class LocalLLM:
    @staticmethod
    @timer(name="Initialize local LLM")
    def get_local_llm(model_name: str = MODEL_NAME, temperature: float = 0.0):
        logger.info("Configuring Local LLM (model=%s, url=%s)", model_name, BASE_URL)
        try:
            return ChatOpenAI(
                model=model_name,
                openai_api_key="not-needed",
                openai_api_base=BASE_URL,
                temperature=temperature,
                max_tokens=8192,
                timeout=300,
                max_retries=3,
            )
        except Exception as e:
            logger.error("Local LLM ChatOpenAI init failed: %s", e)
            raise


def empty_stream():
    yield from []


client = OpenAI(base_url=BASE_URL, api_key="not-needed")


def chat_stream(messages, temperature=0.1):
    logger.info("Streaming local LLM (%s) with %d messages", MODEL_NAME, len(messages))
    try:
        return client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
    except APIConnectionError:
        logger.error("Local LLM server not running at %s", BASE_URL)
        return empty_stream()
    except Exception as e:
        logger.error("Local LLM streaming request failed: %s", e)
        return empty_stream()
