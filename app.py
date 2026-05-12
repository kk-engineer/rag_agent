from components.ingestion import DocumentIngestor
from components.chunking import SemanticProcessor
from components.indexing import IndexingEngine
from components.retrieval import HybridRetriever
from components.post_processing import ContextOptimizer
from components.generation import Generator
from components.guardrails import Guardrails
from evals.ragas_eval import RagasEvaluator
from config.llm_provider import LLMProvider
import os


class RAGPipeline:
    def __init__(self, file_path):
        # 1. Initialize LLM
        self.llm = LLMProvider.get_nvidia_llm(model_name="meta/llama-3.3-70b-instruct")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"  # LangSmith
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = os.environ.get("langchain_api_key")
        os.environ["LANGCHAIN_PROJECT"] = "rag_agent"

        # Init components...
        self.ingestor = DocumentIngestor(file_path)
        self.processor = SemanticProcessor()
        self.indexer = IndexingEngine()

        raw_docs = self.ingestor.load_and_clean()
        print(f"Total documents loaded: {len(raw_docs)}")
        for i, d in enumerate(raw_docs):
            print(f"Doc {i} length: {len(d.page_content)}")
        chunks = self.processor.split(raw_docs)
        self.vector_store = self.indexer.build_vector_store(chunks)
        self.bm25 = self.indexer.build_bm25_index(chunks)

        self.retriever_engine = HybridRetriever(self.vector_store, self.bm25, llm=self.llm)
        self.optimizer = ContextOptimizer(self.retriever_engine.ensemble)
        self.generator = Generator(self.llm)

    def run_query(self, query, ground_truth=None):
        # 1. Retrieval & Optimization
        retrieved_docs = self.optimizer.get_optimized_docs(query)
        print(f"Retrieved docs: {len(retrieved_docs)}")

        # 2. Generation
        raw_answer = self.generator.generate(query, retrieved_docs)
        print(f"Raw answer: {raw_answer}")

        # 3. Guardrails
        is_valid, final_answer = Guardrails.validate_output(raw_answer)
        print(f"Valid answer: {is_valid}")

        # Format for RAGAS
        return {
            "question": query,
            "answer": final_answer if is_valid else "Blocked by Guardrails",
            "contexts": [doc.page_content for doc in retrieved_docs],
            "ground_truth": ground_truth
        }


if __name__ == "__main__":
    pipeline = RAGPipeline("Mastery_by_Robert_Greene.pdf")

    # Define test cases for evaluation
    eval_queries = [
        {"q": "What is the greatest obstacle to pursuit of mastery?", "gt": "The greatest obstacle to our pursuit of mastery comes from the emotional drain we experience \
            in dealing with the resistance and manipulations of the people around us."},
        {"q": "What is Original Mind?", "gt": "It is the mind looked at the world more directly—not through words and received ideas."}
    ]

    results = []
    for item in eval_queries:
        res = pipeline.run_query(item["q"], item["gt"])
        results.append(res)

    # RUN RAGAS EVALUATION
    ragas_eval = RagasEvaluator()
    print(f"Ragas evaluation: {ragas_eval}")
    summary_df = ragas_eval.run_evaluation(results)

    print("--- RAGAS Scores ---")
    print(summary_df)