import unittest

from intern_rag.ingestion import Chunk
from intern_rag.retrieval import HybridRetriever, RetrievalResult


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type="jd",
        source_path=f"data/raw/jd/{chunk_id}.md",
        title=chunk_id,
        text=chunk_id,
        metadata={"source_type": "jd"},
    )


def _fixed_retriever(ids: list[str]):
    chunks = {chunk_id: _chunk(chunk_id) for chunk_id in ids}

    def retrieve(query, all_chunks, top_k=5, source_types=None):
        del query, all_chunks, source_types
        return [
            RetrievalResult(chunk_id, 1.0 / rank, rank, chunks[chunk_id])
            for rank, chunk_id in enumerate(ids[:top_k], start=1)
        ]

    return retrieve


class HybridRetrieverTests(unittest.TestCase):
    def test_rrf_deduplicates_and_preserves_route_ranks(self) -> None:
        retriever = HybridRetriever(
            _fixed_retriever(["shared", "keyword-only"]),
            _fixed_retriever(["shared", "dense-only"]),
            rrf_k=10,
        )

        results = retriever("query", [], top_k=3)

        self.assertEqual([result.chunk_id for result in results], ["shared", "dense-only", "keyword-only"])
        self.assertEqual(len({result.chunk_id for result in results}), 3)
        self.assertEqual(results[0].details["keyword_rank"], 1)
        self.assertEqual(results[0].details["dense_rank"], 1)

    def test_single_empty_route_and_top_k(self) -> None:
        retriever = HybridRetriever(
            _fixed_retriever([]),
            _fixed_retriever(["b", "a"]),
            rrf_k=10,
        )

        results = retriever("query", [], top_k=1)

        self.assertEqual([result.chunk_id for result in results], ["b"])
        self.assertIsNone(results[0].details["keyword_rank"])

    def test_stable_sort_uses_chunk_id_for_ties(self) -> None:
        retriever = HybridRetriever(
            _fixed_retriever(["b"]),
            _fixed_retriever(["a"]),
            rrf_k=10,
        )

        self.assertEqual(
            [result.chunk_id for result in retriever("query", [], top_k=2)],
            ["a", "b"],
        )


if __name__ == "__main__":
    unittest.main()
