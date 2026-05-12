import logging

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.test_cases = [
            {"query": "What is the primary tech stack?", "expected_keyword": "Python"},
            {"query": "How are embeddings generated?", "expected_keyword": "OpenAI"}
        ]

    def run_suite(self):
        logger.info("Starting keyword-based evaluation (%d tests)", len(self.test_cases))
        for case in self.test_cases:
            res = self.pipeline.ask(case["query"])
            passed = case["expected_keyword"].lower() in res.lower()
            status = "PASS" if passed else "FAIL"
            logger.info("[%s] query=%s | expected=%s", status, case["query"], case["expected_keyword"])