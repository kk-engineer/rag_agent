import logging
import re
import unicodedata
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from utils import timer

logger = logging.getLogger(__name__)

HEADING_PATTERNS = [
    re.compile(r'^(CHAPTER|PART|SECTION|APPENDIX|CHAPTER)\s+\w+', re.IGNORECASE),
    re.compile(r'^[A-Z][A-Z\s]{2,50}$'),
    re.compile(r'^\d+\.\s+[A-Z]'),
    re.compile(r'^[IVXLCDM]+\.\s+[A-Z]'),
]


class DocumentIngestor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    @timer(name="Load PDF & clean")
    def load_and_clean(self) -> List[Document]:
        try:
            logger.info("Loading PDF: %s", self.file_path)
            loader = PyPDFLoader(self.file_path)
            docs = loader.load()
        except Exception as e:
            logger.error("Failed to load PDF %s: %s", self.file_path, e)
            raise

        cleaned_docs = []
        current_section = ""
        seen_headings = []

        for doc in docs:
            content = doc.page_content
            page_num = doc.metadata.get("page", 0)

            content = self._clean_text(content)
            if len(content) <= 10:
                continue

            heading = self._detect_heading(content)
            if heading:
                current_section = heading
                if not seen_headings or seen_headings[-1] != heading:
                    seen_headings.append(heading)

            doc.page_content = content
            doc.metadata["source"] = self.file_path
            doc.metadata["page"] = page_num
            doc.metadata["section_title"] = current_section
            doc.metadata["heading_path"] = " > ".join(seen_headings[-5:])

            cleaned_docs.append(doc)

        heading_count = len(
            {d.metadata["section_title"] for d in cleaned_docs if d.metadata["section_title"]}
        )
        logger.info(
            "Loaded %d pages, %d cleaned, %d unique sections detected",
            len(docs),
            len(cleaned_docs),
            heading_count,
        )
        return cleaned_docs

    @staticmethod
    def _clean_text(content: str) -> str:
        try:
            content = unicodedata.normalize("NFKC", content)
            content = content.replace("\x00", "")
            content = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
            content = content.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Text cleaning failed on a page: %s", e)
            content = ""
        return content

    @staticmethod
    def _detect_heading(text: str) -> str:
        lines = text.split("\n")
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
            if 3 <= len(line) <= 80 and any(p.search(line) for p in HEADING_PATTERNS):
                logger.debug("Detected heading: %s", line[:60])
                return line
        return ""
