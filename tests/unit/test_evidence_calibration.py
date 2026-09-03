import unittest

from intern_rag.evaluation.calibration import calibrate_score_gate
from intern_rag.evaluation.knowledge_dataset import KnowledgeEvaluationCase


def _case(case_id: str, split: str, answerable: bool) -> KnowledgeEvaluationCase:
    return KnowledgeEvaluationCase(
        case_id, "测试问题", "single_source", split, ("jd",),
        ("gold",) if answerable else (), ("要点",) if answerable else (), answerable,
    )


def _row(case_id: str, split: str, score: float, chunk_id: str) -> dict:
    return {
        "case_id": case_id, "split": split,
        "predicted": {"retrieved": [{"chunk_id": chunk_id, "score": score, "details": {}}]},
    }


class CalibrationTests(unittest.TestCase):
    def test_selects_separable_operating_point(self) -> None:
        result = calibrate_score_gate(
            "bm25", [_case("p", "dev", True), _case("n", "dev", False)],
            [_row("p", "dev", 0.9, "gold"), _row("n", "dev", 0.1, "noise")],
        )
        self.assertTrue(result.enabled)
        self.assertLessEqual(result.operating_point["far"], 0.05)
        self.assertEqual(result.split, "dev")

    def test_unseparable_scores_disable_gate(self) -> None:
        result = calibrate_score_gate(
            "dense", [_case("p", "dev", True), _case("n", "dev", False)],
            [_row("p", "dev", 0.1, "gold"), _row("n", "dev", 0.9, "noise")],
        )
        self.assertFalse(result.enabled)
        self.assertEqual(result.status, "raw_score_not_separable")

    def test_rejects_test_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "dev"):
            calibrate_score_gate("bm25", [_case("x", "test", True)], [])


if __name__ == "__main__":
    unittest.main()
