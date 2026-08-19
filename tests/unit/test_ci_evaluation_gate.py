import unittest

from intern_rag.evaluation.ci_gate import MetricGate, evaluate_ci_gate


class EvaluationGateTests(unittest.TestCase):
    def test_gate_passes_equal_or_better_candidate(self) -> None:
        result = evaluate_ci_gate(
            {"recall_at_5": 0.7, "p95_ms": 100.0},
            {"recall_at_5": 0.72, "p95_ms": 110.0},
            [
                MetricGate("recall_at_5", "higher_is_better", max_drop=0.01),
                MetricGate("p95_ms", "lower_is_better", max_increase_ratio=1.2),
            ],
            fixed_regression_pass_rate=1.0,
        )
        self.assertTrue(result.passed)

    def test_gate_reports_metric_and_case_failures(self) -> None:
        result = evaluate_ci_gate(
            {"grounding": 0.9}, {"grounding": 0.7},
            [MetricGate("grounding", "higher_is_better", max_drop=0.02)],
            fixed_regression_pass_rate=0.5,
            failed_case_ids=["reg-2", "reg-1"],
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failed_case_ids, ["reg-1", "reg-2"])
        self.assertIn("grounding dropped", result.reasons[0])


if __name__ == "__main__":
    unittest.main()
