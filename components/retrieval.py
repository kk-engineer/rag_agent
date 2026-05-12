import logging
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from utils import timer

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, vectorstore, bm25_retriever, llm):
        self.llm = llm

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. Provide these alternative questions separated by newlines. 
            Do not include any empty lines or numbers.
            Original question: {question}""",
        )

        logger.info("Setting up multi-query retriever (k=5)")
        self.mq_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            llm=self.llm,
            prompt=QUERY_PROMPT
        )

        logger.info("Creating ensemble retriever (dense=0.7, sparse=0.3)")
        self.ensemble = EnsembleRetriever(
            retrievers=[self.mq_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )

    @timer
    def retrieve(self, query: str):
        logger.info("Retrieving documents for: %s", query[:80])
        return self.ensemble.invoke(query)