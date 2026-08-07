from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from intern_rag.evaluation import (
    ClaimJudgment,
    EvidenceSpan,
    FakeGroundingGrader,
    FakeKeyPointGrader,
    GroundingGrade,
    KeyPointGrade,
    PointJudgment,
    SemanticAuditConfig,
    run_saved_prediction_audit,
    save_semantic_audit_artifacts,
)


class SemanticAuditIntegrationTests(unittest.TestCase):
    def test_saved_predictions_are_audited_without_running_pipeline(self) -> None:
        cases = [
            {
                "case_id": "answerable",
                "query": "我做过语义检索吗？",
                "category": "semantic_paraphrase",
                "split": "dev",
                "answerable": True,
                "expected_points": ["向量检索"],
                "status": "answered",
                "answer": "实现过 embedding 语义召回。",
                "citation_ids": ["project-1"],
                "context_ids": ["project-1"],
                "citation_validity": 1.0,
                "key_point_coverage": 0.0,
                "covered_points": [],
                "unsupported_answer": None,
                "router_correct": True,
                "recall_at_5": 1.0,
            },
            {
                "case_id": "unanswerable",
                "query": "薪资多少？",
                "category": "unanswerable",
                "split": "dev",
                "answerable": False,
                "expected_points": [],
                "status": "insufficient_evidence",
                "answer": "证据不足。",
                "citation_ids": [],
                "context_ids": [],
                "citation_validity": None,
                "key_point_coverage": None,
                "covered_points": [],
                "unsupported_answer": None,
                "router_correct": True,
                "recall_at_5": None,
            },
        ]
        traces = [
            {
                "request_id": "answerable",
                "retrieved_chunks": [{
                    "chunk_id": "project-1",
                    "text": "项目实现了 embedding 语义召回。",
                }],
            },
            {"request_id": "unanswerable", "retrieved_chunks": []},
        ]
        point_grader = FakeKeyPointGrader(responses=[KeyPointGrade(
            status="completed",
            judgments=[PointJudgment(
                "向量检索", "covered", "语义等价", "embedding 语义召回"
            )],
            grader_name="fake-point",
            grader_version="v1",
        )])
        grounding_grader = FakeGroundingGrader(responses=[GroundingGrade(
            status="completed",
            claims=[ClaimJudgment(
                "实现过 embedding 语义召回",
                "supported",
                ["project-1"],
                [EvidenceSpan("embedding 语义召回", "project-1")],
                "证据明确支持",
            )],
            grader_name="fake-grounding",
            grader_version="v1",
        )])
        config = SemanticAuditConfig(
            run_id="audit-test",
            dataset_version="evalrag_test",
            split="dev",
            source_prediction_run_id="source-run",
            model="fake",
            temperature=0.0,
            key_point_prompt_version="kp-v1",
            grounding_prompt_version="ground-v1",
            key_point_threshold=0.5,
            input_cache_hit_usd_per_million=0.0,
            input_cache_miss_usd_per_million=0.0,
            output_usd_per_million=0.0,
            pricing_source="test",
            pricing_checked_at="2026-08-04",
            command="test",
        )

        result = run_saved_prediction_audit(
            cases,
            traces,
            config,
            point_grader,
            grounding_grader,
        )

        self.assertEqual(result.summary["metrics"]["key_point_coverage"], 1.0)
        self.assertEqual(result.summary["metrics"]["unsupported_answer_rate"], 0.0)
        self.assertEqual(result.summary["metrics"]["end_to_end_success_rate"], 1.0)
        self.assertEqual(
            result.summary["coverage_comparison"]["improved_case_count"], 1
        )
        self.assertEqual(len(point_grader.calls), 1)
        self.assertEqual(len(grounding_grader.calls), 1)
        self.assertEqual(result.summary["grader_usage"]["call_count"], 0)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            save_semantic_audit_artifacts(result, output)
            self.assertTrue((output / "point_verdicts.jsonl").exists())
            self.assertTrue((output / "claim_verdicts.jsonl").exists())
            self.assertTrue((output / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
