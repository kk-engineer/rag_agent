import logging
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from config.embedding_provider import EmbeddingProvider
from config.llm_provider import LLMProvider
from utils import timer

logger = logging.getLogger(__name__)


class RagasEvaluator:
    def __init__(self):
        logger.info("Initializing RAGAS evaluator")
        self.eval_llm = LLMProvider.get_llm()
        self.eval_embeddings = EmbeddingProvider.get_embedding()
        self.metrics = [
            #faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]
        logger.info("Metrics: %s", [m.name for m in self.metrics])

    @timer
    def run_evaluation(self, results_list: list):
        logger.info("Running RAGAS evaluation on %d results", len(results_list))
        dataset = Dataset.from_list(results_list)

        score = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings
        )

        df = score.to_pandas()
        logger.info("RAGAS evaluation complete")
        return df