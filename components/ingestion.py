import logging
import re
import unicodedata

from langchain_community.document_loaders import (
    PyPDFLoader
)

from utils import timer

logger = logging.getLogger(__name__)


class DocumentIngestor:

    def __init__(self, file_path):
        self.file_path = file_path

    @timer
    def load_and_clean(self):

        logger.info(
            "Loading PDF: %s",
            self.file_path
        )

        loader = PyPDFLoader(
            self.file_path
        )

        docs = loader.load()

        cleaned_docs = []

        for doc in docs:

            content = doc.page_content

            # normalize unicode safely
            content = unicodedata.normalize(
                "NFKC",
                content
            )

            # remove null bytes
            content = content.replace(
                "\x00",
                ""
            )

            # remove control chars only
            content = re.sub(
                r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]",
                " ",
                content
            )

            # collapse whitespace
            content = re.sub(
                r"\s+",
                " ",
                content
            ).strip()

            # utf8 safety
            content = (
                content
                .encode(
                    "utf-8",
                    errors="ignore"
                )
                .decode(
                    "utf-8",
                    errors="ignore"
                )
            )

            if len(content) > 10:

                doc.page_content = content

                cleaned_docs.append(doc)

        logger.info(
            "Loaded %d pages, %d cleaned",
            len(docs),
            len(cleaned_docs)
        )

        return cleaned_docs