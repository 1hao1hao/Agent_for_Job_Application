from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from intern_rag.agent import (
    ContextBudgetError,
    ContextEngine,
    ContextEngineConfig,
    ConversationMessage,
    MemoryItem,
    ProfileFact,
    UserProfile,
)
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult


class FakeSummarizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def summarize(self, messages):
        if self.fail:
            raise RuntimeError("summary unavailable")
        return "用户已确认每周可实习四天。"


class FailingCompressor:
    def compress(self, text: str, query: str) -> str:
        del text, query
        raise RuntimeError("compression unavailable")


class ShortCompressor:
    def compress(self, text: str, query: str) -> str:
        del text, query
        return "需要 Python"


def _result(chunk_id: str, text: str, rank: int = 1) -> RetrievalResult:
    chunk = Chunk(chunk_id, "jd", "data/jd.md", "岗位", text, {})
    return RetrievalResult(chunk_id, 1.0 / rank, rank, chunk, "fixture")


class ContextEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.history = tuple(
            ConversationMessage(
                f"m{index}", "s1", "u1", "user" if index % 2 else "assistant",
                f"第 {index} 轮内容", (self.now + timedelta(minutes=index)).isoformat(),
            )
            for index in range(1, 6)
        )

    def test_managed_context_keeps_required_segments_and_deduplicates_evidence(self) -> None:
        engine = ContextEngine()
        result = _result("c1", "需要 Python 和 RAG")
        duplicate = _result("c2", "需要 Python 和 RAG", rank=2)

        context = engine.build(
            query="岗位要求是什么？",
            system_prompt="仅根据证据回答。",
            retrieved_results=[result, duplicate],
            config=ContextEngineConfig(token_budget=200, mode="no_memory"),
        )

        self.assertIn("[system:system]", context.text)
        self.assertIn("[query:query]", context.text)
        self.assertEqual(context.evidence.used_chunk_ids, ["c1"])
        self.assertLessEqual(context.token_count, context.token_budget)

    def test_summary_recent_and_compression_failure_have_controlled_fallback(self) -> None:
        engine = ContextEngine(
            summarizer=FakeSummarizer(), evidence_compressor=FailingCompressor()
        )
        context = engine.build(
            query="我的时间条件是什么？",
            system_prompt="只回答资料中的事实。",
            retrieved_results=[_result("c1", "岗位每周要求四天")],
            config=ContextEngineConfig(
                token_budget=200, mode="summary_recent", recent_message_count=2,
                compress_evidence=True,
            ),
            history=self.history,
        )

        self.assertIn("用户已确认每周可实习四天", context.text)
        self.assertIn("m4", context.kept_ids)
        self.assertIn("evidence:c1:compression_error", context.compression_fallbacks)

    def test_semantic_memory_filters_unconfirmed_expired_and_other_profile_data(self) -> None:
        expired = (self.now - timedelta(days=1)).isoformat()
        memories = (
            MemoryItem("ok", "u1", "preference", "优先广州岗位", "confirmed_chat", 1.0, self.now.isoformat()),
            MemoryItem("bad", "u1", "fact", "错误事实", "model_guess", 1.0, self.now.isoformat(), confirmed=False),
            MemoryItem("old", "u1", "decision", "过期决定", "chat", 1.0, self.now.isoformat(), expires_at=expired),
        )
        profile = UserProfile(
            "u1", (ProfileFact("技能", "Python", "confirmed_resume"),), 1, self.now.isoformat()
        )

        context = ContextEngine().build(
            query="我优先哪个城市？",
            system_prompt="仅根据证据回答。",
            retrieved_results=[],
            config=ContextEngineConfig(token_budget=120, mode="semantic_memory"),
            profile=profile,
            memories=memories,
        )

        self.assertEqual(context.recalled_memory_ids, ("ok",))
        self.assertIn("Python", context.text)
        self.assertNotIn("错误事实", context.text)
        self.assertNotIn("过期决定", context.text)

    def test_system_and_query_over_budget_raise_instead_of_silent_truncation(self) -> None:
        with self.assertRaises(ContextBudgetError):
            ContextEngine().build(
                query="很长的问题" * 10,
                system_prompt="系统约束" * 10,
                retrieved_results=[],
                config=ContextEngineConfig(token_budget=2),
            )

    def test_reserved_generation_tokens_are_part_of_total_budget(self) -> None:
        context = ContextEngine().build(
            query="岗位要求？",
            system_prompt="仅根据证据回答。",
            retrieved_results=[_result("c1", "需要 Python")],
            config=ContextEngineConfig(
                token_budget=80,
                reserved_token_count=50,
                mode="no_memory",
            ),
        )

        self.assertEqual(context.reserved_token_count, 50)
        self.assertLessEqual(context.token_count, 80)

    def test_full_history_profile_precedence_and_missing_profile(self) -> None:
        profile = UserProfile(
            "u1", (ProfileFact("稳定城市", "广州", "explicit"),), 1, self.now.isoformat()
        )
        context = ContextEngine().build(
            query="我的稳定城市？",
            system_prompt="稳定画像优先于未确认历史。",
            retrieved_results=[],
            config=ContextEngineConfig(token_budget=80, mode="full_history"),
            profile=profile,
            history=self.history[:2],
        )
        self.assertIn("广州", context.text)
        self.assertIn("m1", context.kept_ids)

        without_profile = ContextEngine().build(
            query="没有画像也要正常工作",
            system_prompt="只依据已有内容。",
            retrieved_results=[],
            config=ContextEngineConfig(token_budget=80, mode="recent_window"),
            profile=None,
        )
        self.assertNotIn("profile:", without_profile.text)

    def test_successful_evidence_compression_preserves_citation_header(self) -> None:
        context = ContextEngine(evidence_compressor=ShortCompressor()).build(
            query="需要什么技能？",
            system_prompt="仅依据证据回答。",
            retrieved_results=[_result("c1", "岗位要求 Python，并包含很多补充说明。")],
            config=ContextEngineConfig(
                token_budget=120, mode="no_memory", compress_evidence=True
            ),
        )
        self.assertIn("chunk_id: c1", context.text)
        self.assertIn("source_type: jd", context.text)
        self.assertIn("需要 Python", context.text)


if __name__ == "__main__":
    unittest.main()
