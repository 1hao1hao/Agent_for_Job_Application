import unittest

from intern_rag.agent.evidence import EvidenceConfig, check_evidence
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

    def test_weak_results_stop_after_retry_is_exhausted(self) -> None:
        decision = check_evidence(
            self.route, [_result("jd-1", "jd", 0.1)],
            retriever_name="keyword", retry_count=1, max_retries=1,
            config=self.config,
        )
        self.assertEqual(decision.status, "insufficient")
        self.assertEqual(decision.reason, "weak_retrieval_score")

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


if __name__ == "__main__":
    unittest.main()
