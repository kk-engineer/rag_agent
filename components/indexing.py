import logging
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from utils import timer

logger = logging.getLogger(__name__)


class IndexingEngine:
    def __init__(self, persist_dir="./db"):
        logger.info("Initializing indexing engine (persist_dir=%s)", persist_dir)
        self.embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            nvidia_api_key=os.environ.get("nvidia_api_key")
        )
        self.persist_dir = persist_dir
        self.vectorstore = None

    @timer
    def build_vector_store(self, chunks):
        logger.info("Building Chroma vector store from %d chunks", len(chunks))
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        logger.info("Vector store ready at %s", self.persist_dir)
        return self.vectorstore

    @timer
    def build_bm25_index(self, chunks):
        logger.info("Building BM25 keyword index from %d chunks", len(chunks))
        retriever = BM25Retriever.from_documents(chunks)
        logger.info("BM25 index ready")
        return retriever