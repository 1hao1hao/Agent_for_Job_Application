import json
from pathlib import Path
import tempfile
import unittest

from intern_rag.ingestion import Chunk
from intern_rag.retrieval import (
    BM25Retriever,
    HybridRetriever,
    build_bm25_index,
    load_bm25_index,
    save_bm25_index,
    tokenize_bm25,
)


def _chunk(chunk_id: str, source_type: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type=source_type,
        source_path=f"data/raw/{source_type}/{chunk_id}.md",
        title=chunk_id,
        text=text,
        metadata={"source_type": source_type},
    )


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            _chunk("short", "jd", "Python RAG 检索"),
            _chunk("long", "jd", "Python Python RAG 检索 评测 Agent Context"),
            _chunk("resume", "resume", "简历包含 Python 项目"),
        ]
        self.index = build_bm25_index(self.chunks, "fixture-v1")
        self.retriever = BM25Retriever(self.index)

    def test_tokenizer_preserves_frequency_for_bm25(self) -> None:
        tokens = tokenize_bm25("Python Python 向量检索")

        self.assertEqual(tokens.count("python"), 2)
        self.assertIn("向量", tokens)
        self.assertIn("检索", tokens)

    def test_ranking_source_filter_top_k_and_details(self) -> None:
        results = self.retriever(
            "Python RAG 检索", self.chunks, top_k=1, source_types={"jd"}
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].chunk.source_type, "jd")
        self.assertGreater(results[0].score, 0)
        self.assertIn("bm25_score", results[0].details)

    def test_empty_query_and_non_positive_top_k(self) -> None:
        self.assertEqual(self.retriever("", self.chunks), [])
        self.assertEqual(self.retriever("Python", self.chunks, top_k=0), [])

    def test_index_round_trip_and_invalid_array_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "index.json"
            save_bm25_index(self.index, path)
            loaded = load_bm25_index(path)
            self.assertEqual(loaded.chunk_ids, self.index.chunk_ids)

            payload = loaded.to_dict()
            payload["document_lengths"] = []
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_bm25_index(path)

    def test_bm25_dense_rrf_preserves_bm25_rank(self) -> None:
        hybrid = HybridRetriever(
            self.retriever,
            self.retriever,
            lexical_name="bm25",
            rrf_k=10,
        )

        results = hybrid("Python", self.chunks, top_k=2)

        self.assertIn("bm25_rank", results[0].details)
        self.assertIn("dense_rank", results[0].details)
        self.assertNotIn("keyword_rank", results[0].details)


if __name__ == "__main__":
    unittest.main()
