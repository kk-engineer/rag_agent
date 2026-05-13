import logging
import re
from collections import Counter
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config.embedding_provider import EmbeddingProvider
from utils import timer

logger = logging.getLogger(__name__)

STOPWORDS = set(
    "the a an is are was were be been have has had do does did will would could should may might must shall can need dare ought used to of in on at for with by from as into through during before after above below between out off over under again further then once here there when where why how all each every both few more most other some any no not only own same so than too very just because but or nor and although unless while if else until".split()
)


class IndexingEngine:
    def __init__(self, persist_dir: str = "./db"):
        logger.info("Initializing indexing engine (persist_dir=%s)", persist_dir)
        try:
            self.embeddings = EmbeddingProvider.get_embedding()
        except Exception as e:
            logger.error("Failed to initialize embeddings for indexing: %s", e)
            raise
        self.persist_dir = persist_dir
        self.vectorstore: Optional[Chroma] = None

    @timer(name="Build vector store")
    def build_vector_store(self, chunks: List[Document]) -> Chroma:
        cleaned = []
        for chunk in chunks:
            text = chunk.page_content.strip()
            if not text:
                continue
            text = text[:2000]
            text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            chunk.page_content = text
            cleaned.append(chunk)

        logger.info("Building vector store from %d chunks (after clean)", len(cleaned))

        self._add_context_windows(cleaned)
        self._extract_keywords(cleaned)

        try:
            self.vectorstore = Chroma.from_documents(
                documents=cleaned,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="nomic_v1_5_768",
            )
        except Exception as e:
            logger.error("Chroma vector store creation failed: %s", e)
            raise

        logger.info("Vector store ready with %d vectors", len(cleaned))
        return self.vectorstore

    @timer(name="Build BM25 index")
    def build_bm25_index(self, chunks: List[Document]) -> BM25Retriever:
        logger.info("Building BM25 keyword index from %d chunks", len(chunks))
        try:
            retriever = BM25Retriever.from_documents(chunks)
        except Exception as e:
            logger.error("BM25 index build failed: %s", e)
            raise
        logger.info("BM25 index ready")
        return retriever

    @staticmethod
    def _add_context_windows(chunks: List[Document]) -> None:
        source_map = {}
        for i, c in enumerate(chunks):
            src = c.metadata.get("source", "unknown")
            source_map.setdefault(src, []).append(i)

        for indices in source_map.values():
            for pos, idx in enumerate(indices):
                chunk = chunks[idx]
                chunk.metadata["context_before"] = (
                    chunks[indices[pos - 1]].page_content[:300] if pos > 0 else ""
                )
                chunk.metadata["context_after"] = (
                    chunks[indices[pos + 1]].page_content[:300]
                    if pos < len(indices) - 1
                    else ""
                )

        logger.debug("Added context windows for %d chunks", len(chunks))

    @staticmethod
    def _extract_keywords(chunks: List[Document], top_n: int = 8) -> None:
        for chunk in chunks:
            try:
                words = re.findall(r"[a-zA-Z]{4,}", chunk.page_content.lower())
                words = [w for w in words if w not in STOPWORDS]
                if not words:
                    chunk.metadata["keywords"] = ""
                    continue
                common = [w for w, _ in Counter(words).most_common(top_n)]
                chunk.metadata["keywords"] = ", ".join(common)
            except Exception as e:
                logger.warning("Keyword extraction failed for a chunk: %s", e)
                chunk.metadata["keywords"] = ""
