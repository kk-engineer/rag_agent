import logging

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.test_cases = [
            {"query": "What is the greatest obstacle to pursuit of mastery?", "expected_keyword": "emotional drain"},
            {
                "query": "What is the apprenticeship phase?",
                "expected_keyword": "learning",
            },
        ]

    def run_suite(self):
        logger.info("Starting keyword-based evaluation (%d tests)", len(self.test_cases))
        for case in self.test_cases:
            try:
                res = self.pipeline.ask(case["query"])
                passed = case["expected_keyword"].lower() in res.lower()
                status = "PASS" if passed else "FAIL"
                logger.info(
                    "[%s] query=%s | expected=%s",
                    status,
                    case["query"],
                    case["expected_keyword"],
                )
            except Exception as e:
                logger.error(
                    "Evaluator failed on query '%s': %s", case["query"], e
                )
