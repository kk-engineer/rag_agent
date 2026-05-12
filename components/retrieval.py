from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import PromptTemplate


class HybridRetriever:
    def __init__(self, vectorstore, bm25_retriever, llm):
        self.llm = llm

        # 1. Custom Prompt to prevent NVIDIA 'Empty String' errors
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. Provide these alternative questions separated by newlines. 
            Do not include any empty lines or numbers.
            Original question: {question}""",
        )

        # 2. Multi-Query Retriever
        self.mq_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            llm=self.llm,
            prompt=QUERY_PROMPT
        )

        # 3. Create the Ensemble (Hybrid) Retriever
        # We name this 'ensemble' so the orchestrator can find it
        self.ensemble = EnsembleRetriever(
            retrievers=[self.mq_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )

    def retrieve(self, query: str):
        # High-level method to invoke the ensemble
        return self.ensemble.invoke(query)