import logging
from openai import OpenAI, APIConnectionError

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000/v1"
MISTRAL_MODEL = "Mistral-Small-3bit-MLX"


def empty_stream():
    yield from []


client = OpenAI(
    base_url=BASE_URL,
    api_key="not-needed"
)


def chat_stream(messages, temperature=0.1):
    logger.info("Calling local LLM (%s) with %d messages", MISTRAL_MODEL, len(messages))
    try:
        return client.chat.completions.create(
            model=MISTRAL_MODEL,
            messages=messages,
            temperature=temperature,
            stream=True
        )
    except APIConnectionError:
        logger.error("Local LLM server not running at %s", BASE_URL)
        return empty_stream()
    except Exception as e:
        logger.error("Local LLM request failed: %s", e)
        return empty_stream()