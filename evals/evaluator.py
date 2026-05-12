class RAGEvaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.test_cases = [
            {"query": "What is the primary tech stack?", "expected_keyword": "Python"},
            {"query": "How are embeddings generated?", "expected_keyword": "OpenAI"}
        ]

    def run_suite(self):
        print("--- Starting Evaluation ---")
        for case in self.test_cases:
            res = self.pipeline.ask(case["query"])
            passed = case["expected_keyword"].lower() in res.lower()
            print(f"Query: {case['query']} | Passed: {passed}")