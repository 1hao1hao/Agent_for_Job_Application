import json
import unittest

from intern_rag.agent import FakeLlmClient, LlmTimeoutError
from intern_rag.evaluation import (
    EvidenceSpan,
    GroundingGrade,
    LexicalKeyPointGrader,
    LlmGroundingGrader,
    LlmKeyPointGrader,
    ClaimJudgment,
)


class SemanticGradingTests(unittest.TestCase):
    def test_semantic_grader_recognizes_paraphrase_missed_by_lexical(self) -> None:
        answer = "我实现了基于 embedding 的语义召回。"
        point = "向量检索"
        lexical = LexicalKeyPointGrader().grade(answer, [point])
        client = FakeLlmClient(responses=[json.dumps({
            "judgments": [{
                "point": point,
                "verdict": "covered",
                "answer_evidence": "基于 embedding 的语义召回",
                "reason": "两者表达相同的向量语义检索能力",
            }]
        }, ensure_ascii=False)])
        semantic = LlmKeyPointGrader(
            client,
            model="fake",
            temperature=0.0,
            prompt_version="test-v1",
        ).grade(answer, [point])

        self.assertEqual(lexical.coverage, 0.0)
        self.assertEqual(semantic.coverage, 1.0)
        self.assertEqual(semantic.status, "completed")

    def test_invalid_key_point_json_becomes_unknown_unavailable(self) -> None:
        grader = LlmKeyPointGrader(
            FakeLlmClient(responses=["not-json"]),
            model="fake",
            temperature=0.0,
            prompt_version="test-v1",
        )

        grade = grader.grade("回答", ["要点"])

        self.assertEqual(grade.status, "unavailable")
        self.assertEqual(grade.judgments[0].verdict, "unknown")
        self.assertIsNone(grade.coverage)
        self.assertEqual(grade.error_type, "invalid_json")

    def test_missing_key_point_field_is_controlled_unavailable(self) -> None:
        grader = LlmKeyPointGrader(
            FakeLlmClient(responses=[json.dumps({
                "judgments": [{"point": "要点", "verdict": "covered"}]
            }, ensure_ascii=False)]),
            model="fake",
            temperature=0.0,
            prompt_version="test-v1",
        )

        grade = grader.grade("回答包含要点", ["要点"])

        self.assertEqual(grade.status, "unavailable")
        self.assertEqual(grade.error_type, "invalid_grader_output")
        self.assertIsNone(grade.coverage)

    def test_timeout_is_controlled_unavailable(self) -> None:
        class TimeoutClient:
            last_token_usage = None

            def generate(self, prompt, *, model, temperature):
                del prompt, model, temperature
                raise LlmTimeoutError("timeout")

        grader = LlmGroundingGrader(
            TimeoutClient(),  # type: ignore[arg-type]
            model="fake",
            temperature=0.0,
            prompt_version="test-v1",
        )

        grade = grader.grade("岗位要求 Python。", {"jd-1": "要求 Python"})

        self.assertEqual(grade.status, "unavailable")
        self.assertEqual(grade.error_type, "llm_timeout")
        self.assertIsNone(grade.unsupported_answer)

    def test_supported_claim_requires_real_citation_and_evidence_span(self) -> None:
        answer = "岗位要求熟悉 Python。"
        context = {"jd-1": "岗位要求：熟悉 Python 和 Git。"}
        client = FakeLlmClient(responses=[json.dumps({
            "claims": [{
                "claim": "岗位要求熟悉 Python",
                "verdict": "supported",
                "citation_ids": ["jd-1"],
                "evidence": [{"chunk_id": "jd-1", "text": "熟悉 Python"}],
                "reason": "引用证据明确包含该要求",
            }]
        }, ensure_ascii=False)])
        grader = LlmGroundingGrader(
            client,
            model="fake",
            temperature=0.0,
            prompt_version="test-v1",
        )

        grade = grader.grade(answer, context)

        self.assertEqual(grade.status, "completed")
        self.assertFalse(grade.unsupported_answer)
        self.assertEqual(grade.claims[0].evidence[0].chunk_id, "jd-1")

    def test_fabricated_evidence_span_makes_grade_unavailable(self) -> None:
        client = FakeLlmClient(responses=[json.dumps({
            "claims": [{
                "claim": "候选人熟悉 Kubernetes",
                "verdict": "supported",
                "citation_ids": ["resume-1"],
                "evidence": [{"chunk_id": "resume-1", "text": "熟悉 Kubernetes"}],
                "reason": "证据支持",
            }]
        }, ensure_ascii=False)])
        grader = LlmGroundingGrader(
            client,
            model="fake",
            temperature=0.0,
            prompt_version="test-v1",
        )

        grade = grader.grade(
            "候选人熟悉 Kubernetes。",
            {"resume-1": "候选人熟悉 Python。"},
        )

        self.assertEqual(grade.status, "unavailable")
        self.assertIsNone(grade.unsupported_answer)

    def test_missing_evidence_is_a_valid_unsupported_verdict(self) -> None:
        client = FakeLlmClient(responses=[json.dumps({
            "claims": [{
                "claim": "候选人熟悉 Kubernetes",
                "verdict": "unsupported",
                "citation_ids": [],
                "evidence": [],
                "reason": "引用证据中不存在 Kubernetes 经历",
            }]
        }, ensure_ascii=False)])
        grader = LlmGroundingGrader(
            client,
            model="fake",
            temperature=0.0,
            prompt_version="test-v1",
        )

        grade = grader.grade(
            "候选人熟悉 Kubernetes。",
            {"resume-1": "候选人熟悉 Python。"},
        )

        self.assertEqual(grade.status, "completed")
        self.assertTrue(grade.unsupported_answer)

    def test_unsupported_and_unknown_are_not_treated_as_supported(self) -> None:
        unsupported = GroundingGrade(
            status="completed",
            claims=[ClaimJudgment(
                claim="候选人熟悉 Kubernetes",
                verdict="unsupported",
                citation_ids=[],
                evidence=[],
                reason="引用中不存在该事实",
            )],
            grader_name="fake",
            grader_version="v1",
        )
        unknown = GroundingGrade(
            status="completed",
            claims=[ClaimJudgment(
                claim="候选人经验丰富",
                verdict="unknown",
                citation_ids=["resume-1"],
                evidence=[EvidenceSpan("参与项目", "resume-1")],
                reason="经验丰富缺少明确标准",
            )],
            grader_name="fake",
            grader_version="v1",
        )

        self.assertTrue(unsupported.unsupported_answer)
        self.assertIsNone(unknown.unsupported_answer)


if __name__ == "__main__":
    unittest.main()
