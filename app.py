import logging
import os
from components.ingestion import DocumentIngestor
from components.chunking import SemanticProcessor
from components.indexing import IndexingEngine
from components.retrieval import HybridRetriever
from components.post_processing import ContextOptimizer
from components.generation import Generator
from components.guardrails import Guardrails
from evals.ragas_eval import RagasEvaluator
from config.llm_provider import LLMProvider
from utils import setup_logging, timer

logger = logging.getLogger(__name__)


class RAGPipeline:
    @timer
    def __init__(self, file_path):
        logger.info("=" * 50)
        logger.info("RAG Pipeline Initialization")
        logger.info("=" * 50)

        self.llm = LLMProvider.get_nvidia_llm(model_name="meta/llama-3.3-70b-instruct")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = os.environ.get("langchain_api_key")
        os.environ["LANGCHAIN_PROJECT"] = "rag_agent"

        self.ingestor = DocumentIngestor(file_path)
        self.processor = SemanticProcessor()
        self.indexer = IndexingEngine()

        raw_docs = self.ingestor.load_and_clean()
        chunks = self.processor.split(raw_docs)
        self.vector_store = self.indexer.build_vector_store(chunks)
        self.bm25 = self.indexer.build_bm25_index(chunks)

        self.retriever_engine = HybridRetriever(self.vector_store, self.bm25, llm=self.llm)
        self.optimizer = ContextOptimizer(self.retriever_engine.ensemble)
        self.generator = Generator(self.llm)

        logger.info("Pipeline ready")

    @timer
    def run_query(self, query, ground_truth=None):
        logger.info("Processing query: %s", query[:100])

        retrieved_docs = self.optimizer.get_optimized_docs(query)
        raw_answer = self.generator.generate(query, retrieved_docs)
        is_valid, final_answer = Guardrails.validate_output(raw_answer)

        result = {
            "question": query,
            "answer": final_answer if is_valid else "Blocked by Guardrails",
            "contexts": [doc.page_content for doc in retrieved_docs],
            "ground_truth": ground_truth
        }

        logger.info("Query result: valid=%s | answer=%s...", is_valid, final_answer[:80])
        return result


if __name__ == "__main__":
    setup_logging(logging.INFO)

    pipeline = RAGPipeline("Mastery_by_Robert_Greene.pdf")

    eval_queries = [
        {"q": "What is the greatest obstacle to pursuit of mastery?", "gt": "The greatest obstacle to our pursuit of mastery comes from the emotional drain we experience in dealing with the resistance and manipulations of the people around us."},
        {"q": "What is Original Mind?", "gt": "It is the mind looked at the world more directly—not through words and received ideas."}
    ]

    results = []
    for item in eval_queries:
        logger.info("--- Query: %s ---", item["q"])
        res = pipeline.run_query(item["q"], item["gt"])
        results.append(res)

    ragas_eval = RagasEvaluator()
    summary_df = ragas_eval.run_evaluation(results)

    logger.info("=" * 50)
    logger.info("RAGAS Evaluation Scores")
    logger.info("=" * 50)
    print(summary_df.to_string())