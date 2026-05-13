import logging
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from config.embedding_provider import EmbeddingProvider
from utils import timer

logger = logging.getLogger(__name__)


class IndexingEngine:
    def __init__(self, persist_dir="./db"):
        logger.info("Initializing indexing engine (persist_dir=%s)", persist_dir)
        self.embeddings = EmbeddingProvider.get_embedding()
        self.persist_dir = persist_dir
        self.vectorstore = None

    @timer
    def build_vector_store(self, chunks):

        logger.info(
            "Building vector store from %d chunks",
            len(chunks)
        )

        cleaned_chunks = []

        for i, chunk in enumerate(chunks):
            text = chunk.page_content.strip()
            if not text:
                continue

            # hard truncate for safety
            text = text[:4000]

            # utf8 safety
            text = (
                text
                .encode(
                    "utf-8",
                    errors="ignore"
                )
                .decode(
                    "utf-8",
                    errors="ignore"
                )
            )

            chunk.page_content = text

            cleaned_chunks.append(chunk)

        logger.info(
            "Adding %d cleaned chunks",
            len(cleaned_chunks)
        )

        self.vectorstore = Chroma.from_documents(
            documents=cleaned_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="nomic_v1_5_768",
        )

        logger.info(
            "Vector store ready"
        )

        return self.vectorstore

    @timer
    def build_bm25_index(self, chunks):
        logger.info("Building BM25 keyword index from %d chunks", len(chunks))
        retriever = BM25Retriever.from_documents(chunks)
        logger.info("BM25 index ready")
        return retriever