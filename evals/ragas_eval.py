from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import os
from datasets import Dataset
from config.llm_provider import LLMProvider
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings



class RagasEvaluator:
    def __init__(self):
        # RAGAS uses an LLM and Embeddings to evaluate the outputs
        self.eval_llm = LLMProvider.get_nvidia_llm(model_name="meta/llama-3.3-70b-instruct")
        self.eval_embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            nvidia_api_key=os.environ.get("nvidia_api_key")
        )
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]

    def run_evaluation(self, results_list: list):
        """
        results_list: List of dicts containing:
        {
            "question": str,
            "answer": str,
            "contexts": list[str],
            "ground_truth": str
        }
        """
        # Convert list of dicts to HuggingFace Dataset
        dataset = Dataset.from_list(results_list)

        # Execute RAGAS evaluation
        score = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings
        )

        return score.to_pandas()