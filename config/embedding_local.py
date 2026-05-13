import logging

from langchain_openai import OpenAIEmbeddings

from utils import timer

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8001/v1"
MODEL_NAME = "nomic-embed"


class LocalEmbedding:

    @staticmethod
    @timer(name="Initialize local embeddings")
    def get_local_embedding(
        model_name: str = MODEL_NAME
    ):

        return OpenAIEmbeddings(
            model=model_name,
            api_key="sk-no-key-required",
            base_url=BASE_URL,
            check_embedding_ctx_length=False,
            chunk_size=1024,
            tiktoken_enabled=False,
        )