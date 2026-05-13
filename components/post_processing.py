import logging

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from utils import timer

logger = logging.getLogger(__name__)


class ContextOptimizer:
    def __init__(self, base_retriever):
        logger.info("Initializing FlashRank reranker")
        try:
            self.compressor = FlashrankRerank()
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=base_retriever,
            )
        except Exception as e:
            logger.error("FlashRank reranker init failed: %s", e)
            raise

    @timer(name="Rerank & compress")
    def get_optimized_docs(self, query: str) -> list:
        logger.info("Reranking and compressing results for query")
        try:
            docs = self.compression_retriever.invoke(query)
            logger.info("Returning %d optimized documents", len(docs))
            return docs
        except Exception as e:
            logger.error("Reranking failed for query '%s': %s", query[:50], e)
            raise
