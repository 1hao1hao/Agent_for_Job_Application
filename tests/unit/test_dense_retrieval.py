from pathlib import Path
import tempfile
import unittest

from intern_rag.ingestion import Chunk
from intern_rag.retrieval import (
    DenseIndex,
    DenseIndexMetadata,
    DenseRetriever,
    SentenceTransformerEmbedder,
    build_dense_index,
    load_dense_index,
    save_dense_index,
)


class FakeEmbeddingModel:
    """测试专用向量模型，自动化测试不下载真实模型。"""

    name = "fake-embedding"
    version = "v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = {"语义查询": [1.0, 0.0], "jd": [1.0, 0.0], "resume": [0.0, 1.0]}
        return [vectors.get(text, [0.0, 0.0]) for text in texts]


def _chunk(chunk_id: str, source_type: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type=source_type,
        source_path=f"data/raw/{source_type}/{chunk_id}.md",
        title=chunk_id,
        text=text,
        metadata={"source_type": source_type},
    )


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [_chunk("jd-1", "jd", "jd"), _chunk("resume-1", "resume", "resume")]
        self.index = DenseIndex(
            metadata=DenseIndexMetadata(
                dataset_version="test-v1",
                embedding_name="fake-embedding",
                embedding_version="v1",
                dimensions=2,
                chunk_count=2,
                created_at="2026-08-01T00:00:00+00:00",
            ),
            chunk_ids=["jd-1", "resume-1"],
            vectors=[[1.0, 0.0], [0.0, 1.0]],
        )

    def test_dense_ranking_filter_and_top_k(self) -> None:
        retriever = DenseRetriever(self.index, FakeEmbeddingModel())

        results = retriever("语义查询", self.chunks, top_k=1, source_types={"jd"})

        self.assertEqual([result.chunk_id for result in results], ["jd-1"])
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].details["dense_score"], 1.0)

    def test_index_metadata_mismatch_is_rejected(self) -> None:
        class WrongModel(FakeEmbeddingModel):
            version = "v2"

        with self.assertRaisesRegex(ValueError, "version"):
            DenseRetriever(self.index, WrongModel())

    def test_empty_query_returns_empty_results(self) -> None:
        retriever = DenseRetriever(self.index, FakeEmbeddingModel())
        self.assertEqual(retriever("", self.chunks), [])

    def test_real_index_can_be_saved_loaded_and_queried(self) -> None:
        chunks = [
            _chunk("jd-rag", "jd", "岗位要求建设 RAG 检索评测系统"),
            _chunk("resume-api", "resume", "使用 Python 开发接口服务"),
            _chunk("interview-trace", "interview", "Trace 用于排查阶段错误"),
        ]
        index, model = build_dense_index(
            chunks,
            dataset_version="test-v1",
            dimensions=2,
            max_features=128,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            save_dense_index(index, model, index_dir)
            loaded_index, loaded_model = load_dense_index(index_dir)
            results = DenseRetriever(loaded_index, loaded_model)(
                "RAG 检索", chunks, top_k=1, source_types={"jd"}
            )

        self.assertEqual(loaded_index.metadata.dataset_version, "test-v1")
        self.assertEqual([result.chunk_id for result in results], ["jd-rag"])

    def test_sentence_transformer_adapter_uses_model_encode(self) -> None:
        class FakeSentenceModel:
            def encode(self, texts, normalize_embeddings, show_progress_bar):
                self.call = (texts, normalize_embeddings, show_progress_bar)
                return [[0.6, 0.8]]

        backend = FakeSentenceModel()
        embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
        embedder.model = backend

        vectors = embedder.encode(["中文查询"])

        self.assertEqual(vectors, [[0.6, 0.8]])
        self.assertEqual(backend.call, (["中文查询"], True, False))


if __name__ == "__main__":
    unittest.main()
