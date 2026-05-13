import logging
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from config.embedding_local import LocalEmbedding
from utils import timer

logger = logging.getLogger(__name__)

USE_LOCAL_EMBEDDINGS = True

class EmbeddingProvider:
    @staticmethod
    @timer
    def get_embedding(model_name="nvidia/nv-embedqa-e5-v5"):
        if USE_LOCAL_EMBEDDINGS:
            return LocalEmbedding.get_local_embedding(model_name)
        logger.info("Configuring embedding ", model_name)
        api_key = os.environ.get("nvidia_api_key")
        if not api_key:
            raise ValueError("nvidia_api_key not found in environment variables")

        return NVIDIAEmbeddings(
            model=model_name,
            nvidia_api_key=api_key,
        )