import unittest

from intern_rag.evaluation.metrics import (
    calculate_citation_validity,
    calculate_key_point_coverage,
    summarize_end_to_end_results,
)


class EndToEndMetricsTests(unittest.TestCase):
    def test_citation_validity_handles_valid_invalid_empty_and_abstention(self) -> None:
        self.assertEqual(
            calculate_citation_validity(["a", "bad"], ["a"], status="answered"),
            0.5,
        )
        self.assertEqual(
            calculate_citation_validity([], ["a"], status="answered"), 0.0
        )
        self.assertIsNone(
            calculate_citation_validity([], [], status="insufficient_evidence")
        )

    def test_key_point_coverage_uses_normalized_keyword_match(self) -> None:
        coverage, covered = calculate_key_point_coverage(
            "系统实现 Citation Validator 和引用校验。",
            ["Citation Validator", "Citation Validity"],
        )
        self.assertEqual(coverage, 0.5)
        self.assertEqual(covered, ["Citation Validator"])

    def test_summary_uses_protocol_denominators(self) -> None:
        results = [
            {
                "case_id": "answerable",
                "category": "single_source",
                "answerable": True,
                "status": "answered",
                "citation_ids": ["a"],
                "citation_validity": 1.0,
                "key_point_coverage": 1.0,
                "unsupported_answer": False,
                "router_correct": True,
                "recall_at_5": 1.0,
            },
            {
                "case_id": "unanswerable",
                "category": "unanswerable",
                "answerable": False,
                "status": "insufficient_evidence",
                "citation_ids": [],
                "citation_validity": None,
                "key_point_coverage": None,
                "unsupported_answer": None,
                "router_correct": True,
                "recall_at_5": None,
            },
        ]

        summary = summarize_end_to_end_results(
            results, key_point_threshold=0.5
        )

        self.assertEqual(summary["metrics"]["citation_validity"], 1.0)
        self.assertEqual(summary["metrics"]["abstention_accuracy"], 1.0)
        self.assertEqual(summary["metrics"]["unsupported_answer_rate"], 0.0)
        self.assertEqual(summary["metrics"]["end_to_end_success_rate"], 1.0)

    def test_unknown_unsupported_label_makes_e2e_rate_unavailable(self) -> None:
        result = {
            "case_id": "a",
            "category": "single_source",
            "answerable": True,
            "status": "answered",
            "citation_ids": ["a"],
            "citation_validity": 1.0,
            "key_point_coverage": 1.0,
            "unsupported_answer": None,
            "router_correct": True,
            "recall_at_5": 1.0,
        }
        summary = summarize_end_to_end_results([result], key_point_threshold=0.5)

        self.assertIsNone(summary["metrics"]["unsupported_answer_rate"])
        self.assertIsNone(summary["metrics"]["end_to_end_success_rate"])

    def test_answerable_abstention_is_end_to_end_failure(self) -> None:
        result = {
            "case_id": "unexpected-abstention",
            "category": "semantic_rewrite",
            "answerable": True,
            "status": "insufficient_evidence",
            "citation_ids": [],
            "citation_validity": None,
            "key_point_coverage": None,
            "unsupported_answer": None,
            "router_correct": True,
            "recall_at_5": 0.0,
        }

        summary = summarize_end_to_end_results([result], key_point_threshold=0.5)

        self.assertEqual(summary["metrics"]["end_to_end_success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
