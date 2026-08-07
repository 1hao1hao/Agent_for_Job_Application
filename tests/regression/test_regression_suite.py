from pathlib import Path
import unittest

from intern_rag.evaluation import load_regression_cases, run_regression_suite
from intern_rag.routing import route_query


class ExecutableRegressionTests(unittest.TestCase):
    def test_all_fixed_regressions_pass_and_open_are_excluded(self) -> None:
        cases = load_regression_cases(
            Path("tests/regression/cases_v0.2.jsonl")
        )

        def route_handler(query: str) -> dict[str, object]:
            route = route_query(query)
            return {"intent": route.intent, "sources": route.routed_sources}

        result = run_regression_suite(cases, {"route": route_handler})

        failures = [
            f"{item['case_id']}:{item['failure_type']}"
            for item in result.case_results
            if item["status"] == "fixed" and not item["passed"]
        ]
        self.assertEqual(failures, [], msg=f"regression failures: {failures}")
        self.assertEqual(result.fixed_pass_rate, 1.0)
        self.assertEqual(result.fixed_count, 1)
        self.assertEqual(result.open_count, 4)


if __name__ == "__main__":
    unittest.main()
