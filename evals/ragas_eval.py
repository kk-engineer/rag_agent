import logging

from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall, faithfulness,
)

from config.embedding_provider import EmbeddingProvider
from config.llm_provider import LLMProvider
from utils import timer

logger = logging.getLogger(__name__)


class RagasEvaluator:
    def __init__(self):
        logger.info("Initializing RAGAS evaluator")
        try:
            self.eval_llm = LLMProvider.get_llm()
            self.eval_embeddings = EmbeddingProvider.get_embedding()
        except Exception as e:
            logger.error("RAGAS evaluator init failed: %s", e)
            raise

        self.metrics = [
            #answer_relevancy,
            #faithfulness,
            context_recall,
            context_precision,
        ]
        logger.info("Metrics: %s", [m.name for m in self.metrics])
        self.run_config = RunConfig(max_workers=8, timeout=300)

    @timer(name="RAGAS evaluation")
    def run_evaluation(self, results_list: list):
        logger.info("Running RAGAS evaluation on %d results", len(results_list))
        if not results_list:
            logger.warning("No results to evaluate")
            import pandas as pd
            return pd.DataFrame()

        try:
            dataset = Dataset.from_list(results_list)
            score = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.eval_llm,
                embeddings=self.eval_embeddings,
                run_config=self.run_config,
            )
            df = score.to_pandas()
            logger.info("RAGAS evaluation complete")
            return df
        except Exception as e:
            logger.error("RAGAS evaluation run failed: %s", e)
            raise
