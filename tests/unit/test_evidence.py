import json
from pathlib import Path
import tempfile
import unittest

from intern_rag.agent.evidence import (
    EvidenceConfig,
    ScoreGateConfig,
    check_evidence,
    load_evidence_config,
)
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult
from intern_rag.routing import RouteDecision


def _result(chunk_id: str, source_type: str, score: float) -> RetrievalResult:
    chunk = Chunk(
        id=chunk_id,
        source_type=source_type,
        source_path=f"data/raw/{source_type}/test.md",
        title="测试证据",
        text="用于 Evidence Gate 测试的证据。",
        metadata={"source_type": source_type},
    )
    return RetrievalResult(chunk_id, score, 1, chunk)


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.route = RouteDecision("match_resume", ["jd", "resume"], [])
        self.config = EvidenceConfig(min_scores={"keyword": 0.2})

    def test_empty_results_are_retryable_before_limit(self) -> None:
        decision = check_evidence(
            self.route, [], retriever_name="keyword",
            retry_count=0, max_retries=1, config=self.config,
        )
        self.assertEqual(decision.status, "retryable")
        self.assertEqual(decision.reason, "empty_retrieval")

    def test_legacy_weak_score_stops_after_retry(self) -> None:
        decision = check_evidence(
            self.route,
            [
                _result("jd-1", "jd", 0.1),
                _result("resume-1", "resume", 0.1),
            ],
            retriever_name="keyword", retry_count=1, max_retries=1,
            config=self.config,
        )
        self.assertEqual(decision.status, "insufficient")
        self.assertEqual(decision.reason, "weak_retrieval_score")

    def test_calibrated_low_score_triggers_retry_only_once(self) -> None:
        config = EvidenceConfig(
            min_scores={}, config_version="gate-v2.1",
            calibrated_scores={
                "bm25": ScoreGateConfig(
                    False, None, "raw_score_not_separable", 10.0
                )
            },
            require_source_coverage=False,
        )
        first = check_evidence(
            self.route, [_result("jd-1", "jd", 2.0)],
            retriever_name="adaptive", retry_count=0, max_retries=1,
            config=config, evidence_requirement={"need_type": "exact_fact"},
            retrieval_trace={"selected_strategy": "bm25"},
        )
        final = check_evidence(
            self.route, [_result("jd-1", "jd", 2.0)],
            retriever_name="adaptive", retry_count=1, max_retries=1,
            config=config, evidence_requirement={"need_type": "exact_fact"},
            retrieval_trace={"selected_strategy": "bm25"},
        )
        self.assertEqual(first.status, "retryable")
        self.assertEqual(first.reason, "low_retrieval_confidence")
        self.assertEqual(final.status, "sufficient")
        self.assertTrue(final.low_confidence_after_retry)

    def test_loads_retry_only_threshold_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gate.json"
            path.write_text(json.dumps({
                "config_version": "gate-v2.1",
                "retrievers": {
                    "dense": {
                        "enabled": False,
                        "threshold": None,
                        "low_confidence_retry_threshold": 0.66,
                        "status": "retry_only",
                    }
                },
            }), encoding="utf-8")
            config = load_evidence_config(path)

        score = config.calibrated_scores["dense"]
        self.assertFalse(score.enabled)
        self.assertEqual(score.low_confidence_retry_threshold, 0.66)

    def test_sufficient_results_cover_required_sources(self) -> None:
        decision = check_evidence(
            self.route,
            [_result("jd-1", "jd", 0.8), _result("resume-1", "resume", 0.7)],
            retriever_name="keyword", retry_count=0, max_retries=1,
            config=self.config,
        )
        self.assertEqual(decision.status, "sufficient")

    def test_missing_cross_source_is_reported(self) -> None:
        decision = check_evidence(
            self.route, [_result("jd-1", "jd", 0.8)],
            retriever_name="keyword", retry_count=0, max_retries=1,
            config=self.config,
        )
        self.assertEqual(decision.status, "retryable")
        self.assertEqual(decision.missing_sources, ["resume"])

    def test_unknown_route_is_normal_unanswerable(self) -> None:
        decision = check_evidence(
            RouteDecision("unknown", [], []), [], retriever_name="keyword",
            retry_count=0, max_retries=1, config=self.config,
        )
        self.assertEqual(decision.status, "insufficient")
        self.assertEqual(decision.reason, "unanswerable_route")

    def test_single_source_need_does_not_require_all_router_sources(self) -> None:
        decision = check_evidence(
            self.route, [_result("jd-1", "jd", 0.8)], retriever_name="adaptive",
            retry_count=0, max_retries=1, config=EvidenceConfig(min_scores={}),
            evidence_requirement={"need_type": "exact_fact"},
            retrieval_trace={"strategy": "bm25"},
        )
        self.assertEqual(decision.status, "sufficient")

    def test_disabled_unseparable_score_gate_uses_structure(self) -> None:
        config = EvidenceConfig(
            min_scores={}, config_version="gate-v2",
            calibrated_scores={
                "dense": ScoreGateConfig(False, None, "raw_score_not_separable")
            },
            require_source_coverage=False,
        )
        decision = check_evidence(
            RouteDecision("interview_prep", ["interview"], []),
            [_result("i-1", "interview", -9.0)], retriever_name="adaptive",
            retry_count=0, max_retries=1, config=config,
            evidence_requirement={"need_type": "semantic_explanation"},
            retrieval_trace={"strategy": "dense"},
        )
        self.assertEqual(decision.status, "sufficient")
        self.assertEqual(decision.threshold_status, "raw_score_not_separable")

    def test_relation_need_requires_graph_path(self) -> None:
        decision = check_evidence(
            self.route, [_result("jd-1", "jd", 0.8)], retriever_name="adaptive",
            retry_count=0, max_retries=1, config=EvidenceConfig(min_scores={}),
            evidence_requirement={"need_type": "relation_reasoning"},
            retrieval_trace={"strategy": "graph_hybrid"},
        )
        self.assertEqual(decision.status, "retryable")
        self.assertEqual(decision.reason, "graph_evidence_missing")


if __name__ == "__main__":
    unittest.main()
