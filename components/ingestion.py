import logging
from langchain_community.document_loaders import PyPDFLoader
from utils import timer

logger = logging.getLogger(__name__)


class DocumentIngestor:
    def __init__(self, file_path):
        self.file_path = file_path

    @timer
    def load_and_clean(self):
        logger.info("Loading PDF: %s", self.file_path)
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()
        cleaned_docs = []
        for doc in docs:
            text = doc.page_content.strip()
            if text:
                doc.page_content = text
                cleaned_docs.append(doc)

        logger.info("Loaded %d pages, %d non-empty", len(docs), len(cleaned_docs))
        return cleaned_docs