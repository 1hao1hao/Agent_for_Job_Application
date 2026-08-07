import unittest

from intern_rag.agent import (
    BuiltContext,
    Citation,
    RagRequest,
    RagResponse,
    build_context,
    context_item_from_result,
    format_context_item,
)
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult


def _retrieval_result(
    chunk_id: str,
    text: str,
    rank: int,
    source_type: str = "jd",
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
        score=1.0 / rank,
        rank=rank,
        chunk=chunk,
        reason="测试证据",
    )


class AgentContextTests(unittest.TestCase):
    def test_rag_contract_reuses_existing_citation(self) -> None:
        request = RagRequest(
            query="这个岗位需要哪些能力？",
            request_id="req-1",
            top_k=3,
        )
        citation = Citation(
            chunk_id="jd-1",
            source_path="data/raw/jd/jd-1.md",
            source_type="jd",
            title="测试岗位",
            rank=1,
            score=0.9,
        )
        response = RagResponse(
            request_id=request.request_id,
            trace_id="trace-1",
            answer="岗位要求熟悉 Python。",
            citations=[citation],
            routed_sources=["jd"],
            status="answered",
            latency_ms=12.5,
        )

        self.assertEqual(request.retriever, "keyword")
        self.assertEqual(response.citations, [citation])
        self.assertIsInstance(response.citations[0], Citation)

    def test_rag_request_rejects_empty_query_and_non_positive_top_k(self) -> None:
        with self.assertRaises(ValueError):
            RagRequest(query="  ")
        with self.assertRaises(ValueError):
            RagRequest(query="岗位要求", top_k=0)

    def test_build_context_keeps_fields_and_follows_rank_order(self) -> None:
        results = [
            _retrieval_result(
                "resume-1",
                "我熟悉 Python 和单元测试。",
                rank=2,
                source_type="resume",
            ),
            _retrieval_result(
                "jd-1",
                "岗位要求熟悉 Python、RAG 和效果评测。",
                rank=1,
            ),
        ]

        context = build_context("分析岗位匹配度", results, max_chars=1000)

        self.assertIsInstance(context, BuiltContext)
        self.assertEqual(context.used_chunk_ids, ["jd-1", "resume-1"])
        self.assertEqual(context.skipped_chunk_ids, [])
        self.assertEqual([item.rank for item in context.items], [1, 2])
        self.assertEqual(context.items[0].source_type, "jd")
        self.assertEqual(
            context.items[0].source_path,
            "data/raw/jd/jd-1.md",
        )
        self.assertEqual(context.items[0].title, "jd-jd-1")
        self.assertEqual(
            context.items[0].text,
            "岗位要求熟悉 Python、RAG 和效果评测。",
        )
        self.assertIn("chunk_id: jd-1", context.text)
        self.assertIn("source_path: data/raw/resume/resume-1.md", context.text)
        self.assertEqual(context.char_count, len(context.text))
        self.assertLessEqual(context.char_count, context.max_chars)
        self.assertFalse(context.is_truncated)

    def test_build_context_returns_empty_context_for_empty_results(self) -> None:
        context = build_context("分析岗位", [], max_chars=100)

        self.assertEqual(context.text, "")
        self.assertEqual(context.items, [])
        self.assertEqual(context.used_chunk_ids, [])
        self.assertEqual(context.skipped_chunk_ids, [])
        self.assertEqual(context.char_count, 0)
        self.assertFalse(context.is_truncated)

    def test_first_chunk_over_budget_is_skipped_without_truncation(self) -> None:
        result = _retrieval_result(
            "jd-long",
            "岗位要求：" + "Python " * 30,
            rank=1,
        )
        full_item_text = format_context_item(context_item_from_result(result))
        max_chars = len(full_item_text) - 1

        context = build_context("岗位要求", [result], max_chars=max_chars)

        self.assertEqual(context.text, "")
        self.assertEqual(context.items, [])
        self.assertEqual(context.used_chunk_ids, [])
        self.assertEqual(context.skipped_chunk_ids, ["jd-long"])
        self.assertTrue(context.is_truncated)

    def test_multiple_chunks_stop_at_first_item_over_budget(self) -> None:
        first_result = _retrieval_result("jd-1", "第一条完整证据。", rank=1)
        second_result = _retrieval_result("jd-2", "第二条完整证据。", rank=2)
        third_result = _retrieval_result("jd-3", "第三条完整证据。", rank=3)
        first_item_text = format_context_item(
            context_item_from_result(first_result)
        )

        context = build_context(
            "总结岗位",
            [third_result, second_result, first_result],
            max_chars=len(first_item_text),
        )

        self.assertEqual(context.text, first_item_text)
        self.assertEqual(context.used_chunk_ids, ["jd-1"])
        self.assertEqual(context.skipped_chunk_ids, ["jd-2", "jd-3"])
        self.assertNotIn("第二条完整证据", context.text)
        self.assertNotIn("第三条完整证据", context.text)
        self.assertLessEqual(context.char_count, context.max_chars)
        self.assertTrue(context.is_truncated)

    def test_build_context_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            build_context("", [], max_chars=100)
        with self.assertRaises(ValueError):
            build_context("岗位要求", [], max_chars=0)


if __name__ == "__main__":
    unittest.main()
