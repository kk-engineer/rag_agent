from langchain_community.document_loaders import PyPDFLoader
import os

class DocumentIngestor:
    def __init__(self, file_path):
        print("Ingestion started")
        self.file_path = file_path

    def load_and_clean(self):
        print("Load and clean document")
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()
        cleaned_docs = []
        for doc in docs:
            # 1. Strip whitespace
            text = doc.page_content.strip()
            # 2. Only keep if there's actual text content
            if text:
                doc.page_content = text
                cleaned_docs.append(doc)

        return cleaned_docs