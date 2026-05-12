import logging
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from utils import timer

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_template("""
        Answer the question using ONLY the context below. 
        Provide citations in the format [Source, Page].

        Context: {context}
        Question: {question}
        """)

    @timer
    @traceable
    def generate(self, query, docs):
        logger.info("Generating answer from %d context documents", len(docs))
        context = "\n\n".join([f"[{d.metadata['source']}] {d.page_content}" for d in docs])
        chain = self.prompt_template | self.llm
        response = chain.invoke({"context": context, "question": query})
        logger.info("Generated response (%d chars)", len(response.content))
        return response.content