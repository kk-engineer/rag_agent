from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

class ContextOptimizer:
    def __init__(self, base_retriever):
        print("ContextOptimizer")
        # Reranking
        self.compressor = FlashrankRerank()
        # Compression
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )

    def get_optimized_docs(self, query):
        return self.compression_retriever.invoke(query)