from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
import os

class IndexingEngine:
    def __init__(self, persist_dir="./db"):
        print("Initializing IndexingEngine")
        self.embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            nvidia_api_key=os.environ.get("nvidia_api_key")
        )
        self.persist_dir = persist_dir
        self.vectorstore = None

    def build_vector_store(self, chunks):
        print("Building vector store")
        self.vectorstore =  Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        return self.vectorstore

    def load_vectorstore(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        return self.vectorstore

    def build_bm25_index(self, chunks):
        print("Building bm25 index")
        return BM25Retriever.from_documents(chunks)