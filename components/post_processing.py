import logging
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from utils import timer

logger = logging.getLogger(__name__)


class ContextOptimizer:
    def __init__(self, base_retriever):
        logger.info("Initializing FlashRank reranker")
        self.compressor = FlashrankRerank()
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )

    @timer
    def get_optimized_docs(self, query):
        logger.info("Reranking and compressing results for query")
        docs = self.compression_retriever.invoke(query)
        logger.info("Returning %d optimized documents", len(docs))
        return docs