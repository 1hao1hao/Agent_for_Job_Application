from pathlib import Path
import tempfile
import unittest

from intern_rag.evaluation import (
    EvaluationCase,
    EvaluationRunConfig,
    run_keyword_evaluation,
    save_run_artifacts,
)
from intern_rag.ingestion import Chunk


def _case(*, human_reviewed: bool = True) -> EvaluationCase:
    return EvaluationCase(
        case_id="single_001",
        query="分析岗位 Python 要求",
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


def _config(*, candidate_run: bool = False) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="test-run",
        dataset_version="test-v1",
        split="dev",
        retriever_name="keyword",
        top_k=5,
        chunk_max_chars=800,
        git_commit="test-commit",
        command="test command",
        candidate_run=candidate_run,
    )


class EvaluationRunnerTests(unittest.TestCase):
    def test_runner_generates_predictions_metrics_and_artifacts(self) -> None:
        chunk = Chunk(
            id="jd-1",
            source_type="jd",
            source_path="data/raw/jd/test.md",
            title="测试岗位",
            text="岗位要求熟悉 Python。",
            metadata={"source_type": "jd"},
        )

        result = run_keyword_evaluation([_case()], [chunk], _config())

        self.assertEqual(result.summary["case_count"], 1)
        self.assertEqual(result.summary["metrics"]["router_accuracy"], 1.0)
        self.assertEqual(result.summary["metrics"]["recall_at_3"], 1.0)
        self.assertEqual(result.summary["metrics"]["mrr"], 1.0)
        self.assertEqual(result.summary["metrics"]["ndcg_at_5"], 1.0)
        self.assertNotIn(
            "Candidate runs use labels pending human review and cannot be "
            "reported as formal evaluation.",
            result.summary["limitations"],
        )
        self.assertEqual(
            result.case_results[0]["predicted"]["intent"],
            "analyze_jd",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            save_run_artifacts(result, run_dir)
            artifact_names = {path.name for path in run_dir.iterdir()}

        self.assertEqual(
            artifact_names,
            {
                "run_config.json",
                "summary.json",
                "case_results.jsonl",
                "failures.jsonl",
            },
        )

    def test_formal_runner_rejects_unreviewed_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "human reviewed"):
            run_keyword_evaluation(
                [_case(human_reviewed=False)],
                [],
                _config(candidate_run=False),
            )

    def test_candidate_runner_marks_summary(self) -> None:
        result = run_keyword_evaluation(
            [_case(human_reviewed=False)],
            [],
            _config(candidate_run=True),
        )

        self.assertEqual(
            result.summary["report_status"],
            "candidate_not_human_verified",
        )
        self.assertIn(
            "Candidate runs use labels pending human review and cannot be "
            "reported as formal evaluation.",
            result.summary["limitations"],
        )


if __name__ == "__main__":
    unittest.main()
