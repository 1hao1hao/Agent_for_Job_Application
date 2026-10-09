import unittest

from intern_rag.ingestion import Chunk
from intern_rag.retrieval import (
    AdaptiveRetriever,
    AdaptiveRetrieverConfig,
    EvidenceGapConfig,
    EvidenceGapGuidedRetriever,
    FakeRerankScorer,
    RetrievalResult,
    assess_evidence_gap,
)


def _chunk(chunk_id: str, source_type: str, title: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type=source_type,
        source_path=f"data/raw/{source_type}/{chunk_id}.md",
        title=title,
        text=text,
        metadata={"source_type": source_type},
    )


class CorpusRetriever:
    """测试用确定性 Retriever：按给定顺序返回并遵守来源过滤。"""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls = 0

    def __call__(self, query, chunks, top_k=5, source_types=None):
        del query, chunks
        self.calls += 1
        return [
            item
            for item in self.results
            if source_types is None or item.chunk.source_type in source_types
        ][:top_k]


class EvidenceGapRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.noise = _chunk("jd-noise", "jd", "通用后端岗位", "负责接口开发。")
        self.gold = _chunk(
            "project-gold", "project_logs", "RAGFlow MCP Client Examples",
            "项目实现了 MCP Client 示例与检索链路。",
        )
        initial = RetrievalResult(
            self.noise.id, 1.0, 1, self.noise, "bm25",
            {"bm25_score": 1.0, "bm25_rank": 1},
        )
        rescued = RetrievalResult(
            self.gold.id, 0.9, 1, self.gold, "dense",
            {"dense_score": 0.9, "dense_rank": 1},
        )
        self.bm25 = CorpusRetriever([initial])
        self.dense = CorpusRetriever([rescued])
        base = AdaptiveRetriever(
            {"bm25": self.bm25, "dense": self.dense, "hybrid": self.bm25},
            FakeRerankScorer({self.noise.text: 0.1, self.gold.text: 1.0}),
            config=AdaptiveRetrieverConfig(rerank_policy="never", candidate_k=5),
        )
        self.retriever = EvidenceGapGuidedRetriever(
            base,
            {"bm25": self.bm25, "dense": self.dense},
            config=EvidenceGapConfig(candidate_k=5),
        )

    def test_quoted_title_gap_is_detected_without_gold_label(self) -> None:
        results = self.retriever(
            "《RAGFlow MCP Client Examples》介绍了什么？",
            [self.noise, self.gold],
            top_k=1,
            source_types={"jd"},
        )

        assessment = assess_evidence_gap(
            "《RAGFlow MCP Client Examples》介绍了什么？",
            results,
            self.retriever.get_last_trace(),
        )

        self.assertIn("exact_anchor_missing", assessment.gaps)
        self.assertNotIn("relevant_ids", self.retriever.get_last_trace())

    def test_rescue_uses_complementary_route_and_preserves_provenance(self) -> None:
        query = "《RAGFlow MCP Client Examples》介绍了什么？"
        initial = self.retriever(
            query, [self.noise, self.gold], top_k=1, source_types={"jd"}
        )
        rescued = self.retriever.rescue(
            query,
            [self.noise, self.gold],
            initial_results=initial,
            top_k=1,
            evidence_requirement={"need_type": "exact_fact"},
            initial_trace=self.retriever.get_last_trace(),
        )

        trace = self.retriever.get_last_trace()
        self.assertEqual(rescued[0].chunk_id, "project-gold")
        self.assertTrue(trace["rescue_invoked"])
        self.assertIn("dense", trace["rescue_paths"])
        self.assertEqual(trace["rescue_call_count"], 2)
        self.assertIn("rescue_provenance", rescued[0].details)


if __name__ == "__main__":
    unittest.main()
