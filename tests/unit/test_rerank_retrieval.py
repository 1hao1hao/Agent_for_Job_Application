import unittest
from unittest.mock import patch

from intern_rag.ingestion import Chunk
from intern_rag.retrieval import (
    ChineseTokenOverlapScorer,
    CrossEncoderRerankScorer,
    FakeRerankScorer,
    RerankRetriever,
    RetrievalResult,
)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type="jd",
        source_path=f"data/raw/jd/{chunk_id}.md",
        title=chunk_id,
        text=text,
        metadata={"source_type": "jd"},
    )


class RerankRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            _chunk("c1", "弱相关"),
            _chunk("c2", "强相关"),
            _chunk("c3", "中等相关"),
        ]

    def _candidate_retriever(self, query, chunks, top_k=5, source_types=None):
        del query, source_types
        return [
            RetrievalResult(chunk.id, 1.0 / rank, rank, chunk)
            for rank, chunk in enumerate(chunks[:top_k], start=1)
        ]

    def test_fake_scorer_reorders_candidates_and_keeps_original_rank(self) -> None:
        scorer = FakeRerankScorer({
            "弱相关": 0.1,
            "强相关": 0.9,
            "中等相关": 0.5,
        })
        retriever = RerankRetriever(
            self._candidate_retriever, scorer, candidate_k=3
        )

        results = retriever("测试", self.chunks, top_k=2)

        self.assertEqual([result.chunk_id for result in results], ["c2", "c3"])
        self.assertEqual(results[0].details["original_rank"], 2)
        self.assertEqual(results[0].details["rerank_score"], 0.9)
        self.assertEqual(len(scorer.calls), 1)

    def test_empty_candidates_return_empty_without_scoring(self) -> None:
        scorer = FakeRerankScorer({})
        retriever = RerankRetriever(
            lambda query, chunks, top_k=5, source_types=None: [],
            scorer,
        )

        self.assertEqual(retriever("query", self.chunks), [])
        self.assertEqual(scorer.calls, [])

    def test_score_count_mismatch_is_rejected(self) -> None:
        class BrokenScorer:
            name = "broken"
            version = "v1"

            def score(self, query, documents):
                del query, documents
                return [0.1]

        retriever = RerankRetriever(
            self._candidate_retriever, BrokenScorer(), candidate_k=3
        )

        with self.assertRaises(ValueError):
            retriever("query", self.chunks)

    def test_chinese_token_overlap_scorer_prefers_matching_document(self) -> None:
        scorer = ChineseTokenOverlapScorer()

        scores = scorer.score("Python 检索", ["熟悉 Python 检索", "市场运营"])

        self.assertGreater(scores[0], scores[1])

    def test_cross_encoder_adapter_passes_revision_and_predict_options(self) -> None:
        class FakeCrossEncoder:
            init_args = None
            predict_args = None

            def __init__(self, model_name, **kwargs):
                FakeCrossEncoder.init_args = (model_name, kwargs)

            def predict(self, pairs, **kwargs):
                FakeCrossEncoder.predict_args = (pairs, kwargs)
                return [0.8, 0.2]

        with patch("sentence_transformers.CrossEncoder", FakeCrossEncoder):
            scorer = CrossEncoderRerankScorer(
                "example/chinese-reranker",
                "fixed-revision",
                local_files_only=True,
                batch_size=2,
            )
            scores = scorer.score("查询", ["相关", "无关"])

        self.assertEqual(scores, [0.8, 0.2])
        self.assertEqual(
            FakeCrossEncoder.init_args,
            (
                "example/chinese-reranker",
                {
                    "revision": "fixed-revision",
                    "device": "cpu",
                    "local_files_only": True,
                },
            ),
        )
        self.assertEqual(
            FakeCrossEncoder.predict_args,
            (
                [("查询", "相关"), ("查询", "无关")],
                {"batch_size": 2, "show_progress_bar": False},
            ),
        )


if __name__ == "__main__":
    unittest.main()
