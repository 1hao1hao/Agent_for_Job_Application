import unittest

from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult, retrieve_top_k, tokenize_text


def _chunk(chunk_id: str, source_type: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type=source_type,
        source_path=f"data/raw/{source_type}/{chunk_id}.md",
        title=chunk_id,
        text=text,
        metadata={"source_type": source_type},
    )


class RetrievalTests(unittest.TestCase):
    def test_retrieve_returns_ranked_results_with_required_fields(self) -> None:
        chunks = [
            _chunk("resume-1", "resume", "我熟悉 Python 和单元测试。"),
            _chunk("jd-1", "jd", "岗位要求熟悉 Python、RAG、向量检索和评测。"),
            _chunk("interview-1", "interview", "RAG 面试常问 Recall@k 和 citation。"),
        ]

        results = retrieve_top_k("Python RAG 检索", chunks, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result, RetrievalResult) for result in results))
        self.assertEqual(results[0].chunk_id, "jd-1")
        self.assertEqual(results[0].rank, 1)
        self.assertGreaterEqual(results[0].score, results[1].score)
        self.assertEqual(results[0].chunk.id, "jd-1")
        self.assertIsNotNone(results[0].reason)

    def test_source_type_filter_limits_candidates(self) -> None:
        chunks = [
            _chunk("resume-1", "resume", "简历中包含 RAG 项目经历。"),
            _chunk("jd-1", "jd", "JD 要求了解 RAG 和 Agent。"),
        ]

        results = retrieve_top_k("RAG", chunks, source_types={"resume"})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.source_type, "resume")
        self.assertEqual(results[0].chunk_id, "resume-1")

    def test_top_k_limits_result_count(self) -> None:
        chunks = [
            _chunk("jd-1", "jd", "Python RAG 检索"),
            _chunk("jd-2", "jd", "Python RAG"),
            _chunk("jd-3", "jd", "Python"),
        ]

        results = retrieve_top_k("Python RAG 检索", chunks, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].chunk_id, "jd-1")

    def test_empty_query_or_non_positive_top_k_returns_empty_results(self) -> None:
        chunks = [_chunk("jd-1", "jd", "Python RAG 检索")]

        self.assertEqual(retrieve_top_k("", chunks), [])
        self.assertEqual(retrieve_top_k("Python", chunks, top_k=0), [])

    def test_tokenize_text_supports_chinese_bigrams_and_english_words(self) -> None:
        tokens = tokenize_text("Python 向量检索")

        self.assertIn("python", tokens)
        self.assertIn("向量", tokens)
        self.assertIn("检索", tokens)


if __name__ == "__main__":
    unittest.main()
