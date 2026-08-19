import unittest

from intern_rag.ingestion import Chunk
from intern_rag.retrieval import (
    AdaptiveRetriever,
    AdaptiveRetrieverConfig,
    FakeRerankScorer,
    QueryAnalyzer,
    RetrievalResult,
)


def _chunk(chunk_id: str, source_type: str = "jd") -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type=source_type,
        source_path=f"data/raw/{source_type}/{chunk_id}.md",
        title=chunk_id,
        text=f"{chunk_id} 的证据文本",
        metadata={"source_type": source_type},
    )


class FixedRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[int, set[str] | None]] = []

    def __call__(self, query, chunks, top_k=5, source_types=None):
        del query, chunks
        self.calls.append((top_k, source_types))
        return self.results[:top_k]


class AdaptiveRetrieverTests(unittest.TestCase):
    def _retrievers(self, hybrid_results=None):
        base = [
            RetrievalResult("c1", 0.9, 1, _chunk("c1")),
            RetrievalResult("c2", 0.5, 2, _chunk("c2", "resume")),
            RetrievalResult("c3", 0.3, 3, _chunk("c3")),
        ]
        return {
            "bm25": FixedRetriever(base),
            "dense": FixedRetriever(base),
            "hybrid": FixedRetriever(hybrid_results or base),
        }

    def test_query_analyzer_selects_all_three_strategies(self) -> None:
        analyzer = QueryAnalyzer()

        exact = analyzer.analyze("RRF 是什么", {"interview"})
        semantic = analyzer.analyze("换句话说明召回偏科", {"interview"})
        literal = analyzer.analyze("岗位职责", {"jd"})

        self.assertEqual(analyzer.choose_strategy(exact)[0], "hybrid")
        self.assertEqual(analyzer.choose_strategy(semantic)[0], "dense")
        self.assertEqual(analyzer.choose_strategy(literal)[0], "bm25")

    def test_high_confidence_hybrid_skips_reranker_and_keeps_source_filter(self) -> None:
        hybrid = [
            RetrievalResult(
                "c1", 0.9, 1, _chunk("c1"),
                details={"keyword_rank": 1, "dense_rank": 1},
            ),
            RetrievalResult(
                "c2", 0.4, 2, _chunk("c2"),
                details={"keyword_rank": 2, "dense_rank": 3},
            ),
        ]
        retrievers = self._retrievers(hybrid)
        scorer = FakeRerankScorer({"c1 的证据文本": 0.1})
        retriever = AdaptiveRetriever(retrievers, scorer)

        results = retriever("RRF 检索", [], top_k=2, source_types={"jd"})
        decision = retriever.get_last_trace()

        self.assertEqual([item.chunk_id for item in results], ["c1", "c2"])
        self.assertFalse(decision["rerank_invoked"])
        self.assertEqual(retrievers["hybrid"].calls, [(20, {"jd"})])
        self.assertEqual(scorer.calls, [])

    def test_low_confidence_multi_source_invokes_reranker_once(self) -> None:
        hybrid = [
            RetrievalResult(
                "c1", 0.03, 1, _chunk("c1"),
                details={"keyword_rank": 1, "dense_rank": None},
            ),
            RetrievalResult(
                "c2", 0.029, 2, _chunk("c2", "resume"),
                details={"keyword_rank": 2, "dense_rank": None},
            ),
        ]
        scorer = FakeRerankScorer({"c1 的证据文本": 0.1, "c2 的证据文本": 0.9})
        retriever = AdaptiveRetriever(
            self._retrievers(hybrid),
            scorer,
            config=AdaptiveRetrieverConfig(
                confidence_threshold=0.8,
                original_rank_weight=0.5,
                rerank_rank_weight=1.0,
            ),
        )

        results = retriever(
            "结合岗位和简历分析", [], top_k=2, source_types={"jd", "resume"}
        )
        decision = retriever.get_last_trace()

        self.assertEqual([item.chunk_id for item in results], ["c2", "c1"])
        self.assertTrue(decision["rerank_invoked"])
        self.assertTrue(decision["rerank_applied"])
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(results[0].details["original_rank"], 2)

    def test_empty_source_route_returns_without_calling_any_retriever(self) -> None:
        retrievers = self._retrievers()
        scorer = FakeRerankScorer({})
        retriever = AdaptiveRetriever(retrievers, scorer)

        results = retriever("未知问题", [], source_types=set())

        self.assertEqual(results, [])
        self.assertEqual(sum(len(item.calls) for item in retrievers.values()), 0)
        self.assertEqual(retriever.get_last_trace()["candidate_count"], 0)

    def test_rerank_ties_keep_original_order_stable(self) -> None:
        retriever = AdaptiveRetriever(
            self._retrievers(),
            FakeRerankScorer({
                "c1 的证据文本": 0.5,
                "c2 的证据文本": 0.5,
                "c3 的证据文本": 0.5,
            }),
            config=AdaptiveRetrieverConfig(confidence_threshold=1.0),
        )

        results = retriever("岗位职责", [], top_k=3, source_types={"jd"})

        self.assertEqual([item.chunk_id for item in results], ["c1", "c2", "c3"])

    def test_rerank_policy_never_skips_low_confidence_candidates(self) -> None:
        hybrid = [
            RetrievalResult(
                "c1", 0.03, 1, _chunk("c1"),
                details={"keyword_rank": 1, "dense_rank": None},
            )
        ]
        scorer = FakeRerankScorer({"c1 的证据文本": 1.0})
        retriever = AdaptiveRetriever(
            self._retrievers(hybrid),
            scorer,
            config=AdaptiveRetrieverConfig(
                confidence_threshold=1.0,
                rerank_policy="never",
                force_strategy="hybrid",
            ),
        )

        retriever("任意问题", [], top_k=1, source_types={"jd"})

        self.assertFalse(retriever.get_last_trace()["rerank_invoked"])
        self.assertEqual(scorer.calls, [])
        self.assertIn("forced hybrid", retriever.get_last_trace()["reason"])

    def test_rerank_policy_always_invokes_for_high_confidence_candidates(self) -> None:
        hybrid = [
            RetrievalResult(
                "c1", 0.9, 1, _chunk("c1"),
                details={"keyword_rank": 1, "dense_rank": 1},
            )
        ]
        scorer = FakeRerankScorer({"c1 的证据文本": 1.0})
        retriever = AdaptiveRetriever(
            self._retrievers(hybrid),
            scorer,
            config=AdaptiveRetrieverConfig(
                rerank_policy="always",
                force_strategy="hybrid",
            ),
        )

        retriever("任意问题", [], top_k=1, source_types={"jd"})

        self.assertTrue(retriever.get_last_trace()["rerank_invoked"])
        self.assertEqual(len(scorer.calls), 1)


if __name__ == "__main__":
    unittest.main()
