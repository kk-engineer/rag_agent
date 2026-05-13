import logging

from langchain_core.prompts import ChatPromptTemplate
from opensmith import trace

from utils import timer

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_template(
            """
        Answer the question using ONLY the context below.
        Provide citations in the format [Source, Page].

        Context: {context}
        Question: {question}
        """
        )

    @timer(name="LLM generation")
    @trace(tags=["rag", "generation"])
    def generate(self, query: str, docs: list) -> str:
        logger.info("Generating answer from %d context documents", len(docs))
        try:
            context_lines = []
            for d in docs:
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page", "?")
                section = d.metadata.get("section_title", "")
                kw = d.metadata.get("keywords", "")

                header = f"[Source: {src}, Page: {page}"
                if section:
                    header += f", Section: {section}"
                if kw:
                    kw_list = kw.split(", ") if isinstance(kw, str) else kw
                    header += f", Keywords: {', '.join(kw_list[:3])}"
                header += "]"
                context_lines.append(f"{header}\n{d.page_content}")

            context = "\n\n".join(context_lines)
            chain = self.prompt_template | self.llm
            response = chain.invoke({"context": context, "question": query})
            logger.info("Generated response (%d chars)", len(response.content))
            return response.content
        except Exception as e:
            logger.error("Generation failed: %s", e)
            raise
