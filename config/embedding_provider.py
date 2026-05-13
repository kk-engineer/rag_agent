import logging
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from config.embedding_local import LocalEmbedding
from utils import timer

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    @staticmethod
    @timer(name="Initialize embeddings")
    def get_embedding(model_name: str = "nvidia/nv-embedqa-e5-v5", use_local: bool = True):
        if use_local:
            try:
                return LocalEmbedding.get_local_embedding(model_name)
            except Exception as e:
                logger.error("Local embedding init failed: %s", e)
                raise

        logger.info("Configuring cloud embedding: %s", model_name)
        api_key = os.environ.get("nvidia_api_key")
        if not api_key:
            raise ValueError("nvidia_api_key not found in environment variables")

        try:
            return NVIDIAEmbeddings(
                model=model_name,
                nvidia_api_key=api_key,
            )
        except Exception as e:
            logger.error("NVIDIA embedding init failed: %s", e)
            raise
