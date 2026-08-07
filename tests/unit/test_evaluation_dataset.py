from pathlib import Path
import tempfile
import unittest

from intern_rag.evaluation import (
    EvaluationCase,
    load_evaluation_dataset,
    validate_evaluation_dataset,
)


def _case(
    case_id: str = "single_001",
    *,
    human_reviewed: bool = True,
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        query="岗位要求什么？",
        category="single_source",
        split="dev",
        expected_intent="analyze_jd",
        expected_sources=["jd"],
        relevant_chunk_ids=["jd-1"],
        answerable=True,
        expected_points=["Python"],
        notes="测试标签",
        human_reviewed=human_reviewed,
    )


class EvaluationDatasetTests(unittest.TestCase):
    def test_loader_rejects_handwritten_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.jsonl"
            path.write_text(
                '{"case_id":"x","predicted_intent":"analyze_jd"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "forbidden predictions"):
                load_evaluation_dataset(path)

    def test_validation_checks_duplicate_missing_chunk_and_review(self) -> None:
        cases = [
            _case(human_reviewed=False),
            _case(human_reviewed=False),
        ]

        result = validate_evaluation_dataset(
            cases,
            available_chunk_ids=set(),
            require_full_distribution=False,
            require_human_review=True,
        )

        self.assertFalse(result.is_valid)
        self.assertTrue(any("duplicate case_id" in error for error in result.errors))
        self.assertTrue(
            any("references missing chunks" in error for error in result.errors)
        )
        self.assertTrue(
            any("has not been human reviewed" in error for error in result.errors)
        )

    def test_candidate_validation_allows_pending_review_with_warning(self) -> None:
        result = validate_evaluation_dataset(
            [_case(human_reviewed=False)],
            available_chunk_ids={"jd-1"},
            require_full_distribution=False,
            require_human_review=False,
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.reviewed_case_count, 0)
        self.assertIn("pending human review", result.warnings[0])


if __name__ == "__main__":
    unittest.main()
