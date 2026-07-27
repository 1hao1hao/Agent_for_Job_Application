import unittest

from intern_rag.agent import AnswerResult, Citation, compose_answer
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult


def _result(
    chunk_id: str,
    text: str,
    rank: int = 1,
    source_type: str = "jd",
    score: float = 0.8,
) -> RetrievalResult:
    chunk = Chunk(
        id=chunk_id,
        source_type=source_type,
        source_path=f"data/raw/{source_type}/{chunk_id}.md",
        title=f"{source_type}-{chunk_id}",
        text=text,
        metadata={"source_type": source_type},
    )
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        chunk=chunk,
        reason="rag, 检索",
    )


class AnswerTests(unittest.TestCase):
    def test_compose_answer_uses_only_retrieved_chunk_text(self) -> None:
        result = _result(
            "jd-1",
            "岗位职责包括参与企业知识库问答、RAG、Agent 工具调用等大模型应用开发。",
        )

        answer_result = compose_answer("这个岗位做什么", [result])

        self.assertIsInstance(answer_result, AnswerResult)
        self.assertTrue(answer_result.is_evidence_sufficient)
        self.assertIn("企业知识库问答", answer_result.answer)
        self.assertIn("Agent 工具调用", answer_result.answer)
        self.assertNotIn("负责线上部署", answer_result.answer)
        self.assertEqual(answer_result.used_chunk_ids, ["jd-1"])

    def test_citation_points_to_chunk_id_and_source_path(self) -> None:
        result = _result("resume-1", "我熟悉 Python 和单元测试。", source_type="resume")

        answer_result = compose_answer("简历有哪些技能", [result])

        self.assertEqual(len(answer_result.citations), 1)
        citation = answer_result.citations[0]
        self.assertIsInstance(citation, Citation)
        self.assertEqual(citation.chunk_id, "resume-1")
        self.assertEqual(citation.source_path, "data/raw/resume/resume-1.md")
        self.assertEqual(citation.source_type, "resume")
        self.assertEqual(citation.rank, 1)
        self.assertEqual(citation.score, 0.8)

    def test_insufficient_evidence_returns_clear_uncertainty(self) -> None:
        answer_result = compose_answer("这个岗位要求什么", [])

        self.assertFalse(answer_result.is_evidence_sufficient)
        self.assertEqual(answer_result.citations, [])
        self.assertEqual(answer_result.used_chunk_ids, [])
        self.assertIn("当前证据不足", answer_result.answer)

    def test_max_chunks_limits_answer_and_citations(self) -> None:
        results = [
            _result("jd-1", "第一条证据", rank=1),
            _result("jd-2", "第二条证据", rank=2),
            _result("jd-3", "第三条证据", rank=3),
        ]

        answer_result = compose_answer("总结岗位", results, max_chunks=2)

        self.assertEqual(answer_result.used_chunk_ids, ["jd-1", "jd-2"])
        self.assertEqual(len(answer_result.citations), 2)
        self.assertIn("第一条证据", answer_result.answer)
        self.assertIn("第二条证据", answer_result.answer)
        self.assertNotIn("第三条证据", answer_result.answer)

    def test_non_positive_limits_return_insufficient_answer(self) -> None:
        result = _result("jd-1", "岗位要求熟悉 Python。")

        answer_result = compose_answer("岗位要求", [result], max_chunks=0)

        self.assertFalse(answer_result.is_evidence_sufficient)
        self.assertEqual(answer_result.citations, [])


if __name__ == "__main__":
    unittest.main()
