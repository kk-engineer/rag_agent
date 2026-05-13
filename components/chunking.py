import logging
from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.embedding_provider import EmbeddingProvider
from config.settings import USE_SEMANTIC_CHUNKING
from utils import timer

logger = logging.getLogger(__name__)


class SemanticProcessor:
    def __init__(self):
        self.safety_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
        )

        if USE_SEMANTIC_CHUNKING:
            logger.info("Initializing embeddings for semantic chunking")
            try:
                self.embeddings = EmbeddingProvider.get_embedding()
            except Exception as e:
                logger.error("Failed to initialize embeddings: %s", e)
                raise

    @timer(name="Chunking")
    def split(self, documents: List[Document]) -> List[Document]:
        valid_docs = [d for d in documents if d.page_content and d.page_content.strip()]
        if not valid_docs:
            logger.warning("No valid text content found to split")
            return []

        logger.info("Splitting %d documents into chunks", len(valid_docs))

        pre_split = self._run_recursive_split(valid_docs)

        if USE_SEMANTIC_CHUNKING:
            chunks = self._run_semantic_split(pre_split)
        else:
            logger.info("Semantic chunking disabled, using recursive split")
            chunks = pre_split

        self._enrich_metadata(chunks)
        return chunks

    @timer(name="Recursive split")
    def _run_recursive_split(self, valid_docs: List[Document]) -> List[Document]:
        try:
            pre_split = self.safety_splitter.split_documents(valid_docs)
        except Exception as e:
            logger.error("Safety pre-split failed: %s", e)
            raise
        pre_split = [d for d in pre_split if d.page_content.strip()]
        logger.info("Safety pre-split produced %d chunks", len(pre_split))
        return pre_split

    @timer(name="Semantic chunking")
    def _run_semantic_split(self, pre_split: List[Document]) -> List[Document]:
        try:
            semantic_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
            )
            chunks = semantic_splitter.split_documents(pre_split)
            logger.info("Semantic split produced %d chunks", len(chunks))
            return chunks
        except Exception as e:
            logger.warning("SemanticChunker failed (%s), falling back to recursive chunks", e)
            return pre_split

    @staticmethod
    @timer(name="Enrich metadata")
    def _enrich_metadata(chunks: List[Document]) -> None:
        source_groups = {}
        for i, chunk in enumerate(chunks):
            src = chunk.metadata.get("source", "unknown")
            source_groups.setdefault(src, []).append(i)

        for src, indices in source_groups.items():
            total = len(indices)
            for pos, idx in enumerate(indices):
                chunk = chunks[idx]
                chunk.metadata["chunk_index"] = pos
                chunk.metadata["total_chunks"] = total
                if "section_title" not in chunk.metadata or not chunk.metadata["section_title"]:
                    chunk.metadata["section_title"] = ""
                if "heading_path" not in chunk.metadata:
                    chunk.metadata["heading_path"] = ""

        logger.debug(
            "Enriched metadata for %d chunks across %d sources",
            len(chunks),
            len(source_groups),
        )
