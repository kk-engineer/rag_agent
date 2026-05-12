import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticProcessor:
    def __init__(self, embedding_model=None):
        """
        Initializes the semantic chunker.
        If no embedding_model is provided, it configures it for NVIDIA by default.
        """
        print("Initializing SemanticProcessor")
        self.embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            nvidia_api_key=os.environ.get("nvidia_api_key")
        )
        # Step 1: Create a safety splitter to ensure no text exceeds the token limit
        # 1 token is roughly 4 characters, so 512 tokens ~ 2000 chars.
        # We use 1500 to be safe.
        self.safety_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        # Step 2: The Semantic Chunker for meaningful splits
        self.semantic_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile"
        )

    def split(self, documents):
        # 1. Filter out documents with empty or whitespace-only content
        valid_docs = [d for d in documents if d.page_content and d.page_content.strip()]

        if not valid_docs:
            print("⚠️ No valid text content found to split. Skipping chunking.")
            return []

        # 2. Safety pre-split to stay under NVIDIA's 512 token limit
        pre_split_docs = self.safety_splitter.split_documents(valid_docs)
        return pre_split_docs

        # 3. Final check: ensure pre_split didn't result in empty strings
        final_input = [d for d in pre_split_docs if d.page_content.strip()]

        try:
            return self.semantic_splitter.split_documents(final_input)
        except Exception as e:
            print(f"❌ NVIDIA API Error during semantic split: {e}")
            return final_input  # Fallback to recursive chunks


# from langchain_text_splitters import RecursiveCharacterTextSplitter
#
#
# def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#     )
#
#     chunks = splitter.split_documents(documents)
#
#     return chunks