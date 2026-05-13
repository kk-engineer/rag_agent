import logging
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.embedding_provider import EmbeddingProvider
from utils import timer

logger = logging.getLogger(__name__)


class SemanticProcessor:
    def __init__(self):
        logger.info("Initializing embeddings for semantic chunking")
        self.embeddings = EmbeddingProvider.get_embedding()
        self.safety_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        self.semantic_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile"
        )

    @timer
    def split(self, documents):
        valid_docs = [d for d in documents if d.page_content and d.page_content.strip()]

        if not valid_docs:
            logger.warning("No valid text content found to split")
            return []

        logger.info("Splitting %d documents into chunks", len(valid_docs))
        pre_split_docs = self.safety_splitter.split_documents(valid_docs)
        logger.info("Safety pre-split produced %d chunks", len(pre_split_docs))

        final_input = [d for d in pre_split_docs if d.page_content.strip()]
        return final_input
        try:
            semantic_docs = self.semantic_splitter.split_documents(final_input)
            logger.info("Semantic split produced %d chunks", len(semantic_docs))
            return semantic_docs
        except Exception as e:
            logger.warning("Semantic split failed (%s), falling back to recursive chunks", e)
            return final_input