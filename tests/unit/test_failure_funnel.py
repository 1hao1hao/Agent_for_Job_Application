import unittest

from intern_rag.agent import Citation, RagResponse
from intern_rag.evaluation.failure_funnel import (
    classify_failure_stage,
    summarize_failure_funnel,
)
from intern_rag.evaluation.knowledge_dataset import KnowledgeEvaluationCase
from intern_rag.routing import RouteDecision
from intern_rag.tracing import build_agent_trace


def _case(*, answerable: bool = True) -> KnowledgeEvaluationCase:
    return KnowledgeEvaluationCase(
        case_id="case-1",
        query="测试问题",
        category="single_source",
        split="dev",
        expected_sources=("jd",),
        relevant_chunk_ids=("gold-1",) if answerable else (),
        expected_points=("测试要点",),
        answerable=answerable,
    )


def _response(status: str, *, citations: list[Citation] | None = None):
    return RagResponse(
        request_id="case-1",
        trace_id="trace-1",
        answer="回答",
        citations=citations or [],
        routed_sources=["jd"],
        status=status,  # type: ignore[arg-type]
        latency_ms=10.0,
    )


def _trace(*, retrieved: list[str], context: list[str], evidence: str, generation=None):
    trace = build_agent_trace(
        "测试问题",
        RouteDecision("analyze_jd", ["jd"], []),
        [],
        {"total": 10.0},
        request_id="case-1",
        trace_id="trace-1",
        retrieval={"chunk_ids": retrieved},
        context={"used_chunk_ids": context},
        evidence={"status": evidence, "reason": "test_reason"},
        generation=generation or {},
        attempts=[{"type": "initial_retrieval", "retrieved_chunk_ids": retrieved}],
    )
    return trace


class FailureFunnelTests(unittest.TestCase):
    def test_distinguishes_retrieval_miss_from_gate_reject(self) -> None:
        miss = classify_failure_stage(
            _case(), _response("insufficient_evidence"),
            _trace(retrieved=["noise"], context=[], evidence="insufficient"),
        )
        rejected = classify_failure_stage(
            _case(), _response("insufficient_evidence"),
            _trace(retrieved=["gold-1"], context=[], evidence="insufficient"),
        )

        self.assertEqual(miss.terminal_stage, "retrieval_miss")
        self.assertEqual(rejected.terminal_stage, "gate_reject")

    def test_detects_context_drop_and_generator_abstention(self) -> None:
        dropped = classify_failure_stage(
            _case(), _response("insufficient_evidence"),
            _trace(retrieved=["gold-1"], context=["noise"], evidence="sufficient"),
        )
        generated = classify_failure_stage(
            _case(), _response("insufficient_evidence"),
            _trace(
                retrieved=["gold-1"], context=["gold-1"], evidence="sufficient",
                generation={"sufficient": False, "reason": "模型拒答"},
            ),
        )

        self.assertEqual(dropped.terminal_stage, "context_evidence_dropped")
        self.assertEqual(generated.terminal_stage, "generator_abstain")

    def test_summary_uses_explicit_denominators(self) -> None:
        answered = classify_failure_stage(
            _case(),
            _response(
                "answered",
                citations=[
                    Citation(
                        "gold-1", "data/raw/jd/a.md", "jd", "证据", 1, 0.9
                    )
                ],
            ),
            _trace(retrieved=["gold-1"], context=["gold-1"], evidence="sufficient"),
        )
        abstained = classify_failure_stage(
            _case(answerable=False), _response("insufficient_evidence"),
            _trace(retrieved=[], context=[], evidence="insufficient"),
        )

        summary = summarize_failure_funnel([answered, abstained])

        self.assertEqual(summary["answerable_e2e_success"], 1.0)
        self.assertEqual(summary["unanswerable_abstention_accuracy"], 1.0)
        self.assertEqual(summary["citation_validity_answered"], 1.0)


if __name__ == "__main__":
    unittest.main()
