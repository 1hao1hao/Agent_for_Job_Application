from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from intern_rag.agent import (
    ContextBudgetError,
    ContextEngine,
    ContextEngineConfig,
    ContextPlan,
    ContextSignals,
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


def _result(
    chunk_id: str,
    text: str,
    rank: int = 1,
    source_type: str = "jd",
) -> RetrievalResult:
    chunk = Chunk(chunk_id, source_type, "data/jd.md", "岗位", text, {})
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

    def test_tight_budget_keeps_higher_rank_evidence_before_chunk_id_order(self) -> None:
        engine = ContextEngine()
        results = [
            _result("z-rank-1", "排名第一的关键证据", rank=1),
            _result("a-rank-2", "排名第二的干扰证据", rank=2),
        ]
        full = engine.build(
            query="哪个证据优先？",
            system_prompt="仅根据证据回答。",
            retrieved_results=results,
            config=ContextEngineConfig(token_budget=300, mode="no_memory"),
        )
        budget = sum(
            segment.token_count
            for segment in full.segments
            if segment.segment_id in {"system", "query", "z-rank-1"}
        )

        context = engine.build(
            query="哪个证据优先？",
            system_prompt="仅根据证据回答。",
            retrieved_results=results,
            config=ContextEngineConfig(token_budget=budget, mode="no_memory"),
        )

        self.assertEqual(context.evidence.used_chunk_ids, ["z-rank-1"])
        self.assertIn("a-rank-2", context.evidence.skipped_chunk_ids)

    def test_evidence_first_keeps_required_evidence_before_profile_and_memory(self) -> None:
        engine = ContextEngine()
        result = _result("required-evidence", "岗位要求 Python", rank=1)
        profile = UserProfile(
            user_id="u1",
            version=1,
            facts=(ProfileFact("偏好", "一段很长的用户画像" * 8, "test", True),),
            updated_at=self.now.isoformat(),
        )
        baseline = engine.build(
            query="岗位要求是什么？",
            system_prompt="仅根据证据回答。",
            retrieved_results=[result],
            config=ContextEngineConfig(
                token_budget=100, mode="no_memory", evidence_first=False
            ),
            required_source_types=("jd",),
            profile=profile,
        )
        evidence_first = engine.build(
            query="岗位要求是什么？",
            system_prompt="仅根据证据回答。",
            retrieved_results=[result],
            config=ContextEngineConfig(
                token_budget=100, mode="no_memory", evidence_first=True
            ),
            required_source_types=("jd",),
            profile=profile,
        )

        self.assertNotIn("required-evidence", baseline.evidence.used_chunk_ids)
        self.assertIn("required-evidence", evidence_first.evidence.used_chunk_ids)
        self.assertNotIn("profile:1", evidence_first.kept_ids)

    def test_evidence_first_prioritizes_required_source_within_evidence(self) -> None:
        engine = ContextEngine()
        results = [
            _result("noise-1", "无关来源证据" * 8, 1, "interview"),
            _result("noise-2", "另一条无关证据" * 8, 2, "project_logs"),
            _result("required", "简历中的必要证据" * 8, 3, "resume"),
        ]
        full = engine.build(
            query="简历如何证明匹配？",
            system_prompt="仅根据证据回答。",
            retrieved_results=results,
            config=ContextEngineConfig(token_budget=400, mode="no_memory"),
            required_source_types=("resume",),
        )
        required_segment = next(
            item for item in full.segments if item.segment_id == "required"
        )
        fixed_tokens = sum(
            item.token_count
            for item in full.segments
            if item.segment_id in {"system", "query"}
        )
        context = engine.build(
            query="简历如何证明匹配？",
            system_prompt="仅根据证据回答。",
            retrieved_results=results,
            config=ContextEngineConfig(
                token_budget=fixed_tokens + required_segment.token_count,
                mode="no_memory",
            ),
            required_source_types=("resume",),
        )

        self.assertEqual(context.evidence.used_chunk_ids, ["required"])

    def test_graph_path_group_is_kept_or_dropped_atomically(self) -> None:
        chunks = [
            _result("path-a", "岗位要求 Python", 1, "jd"),
            _result("path-b", "项目使用 Python", 2, "project_logs"),
        ]
        results = [
            RetrievalResult(
                item.chunk_id, item.score, item.rank, item.chunk, item.reason,
                {
                    "path_group_id": "path-1",
                    "path_group_size": 2,
                    "path_valid": 1,
                    "graph_edge_ids": "edge-1|edge-2",
                },
            )
            for item in chunks
        ]
        full = ContextEngine().build(
            query="项目如何证明岗位匹配？",
            system_prompt="仅根据证据回答。",
            retrieved_results=results,
            config=ContextEngineConfig(token_budget=300, mode="no_memory"),
        )
        fixed_tokens = sum(
            item.token_count for item in full.segments
            if item.segment_id in {"system", "query"}
        )
        first_tokens = next(
            item.token_count for item in full.segments if item.segment_id == "path-a"
        )

        tight = ContextEngine().build(
            query="项目如何证明岗位匹配？",
            system_prompt="仅根据证据回答。",
            retrieved_results=results,
            config=ContextEngineConfig(
                token_budget=fixed_tokens + first_tokens, mode="no_memory"
            ),
        )

        self.assertEqual(full.evidence.used_chunk_ids, ["path-a", "path-b"])
        self.assertEqual(tight.evidence.used_chunk_ids, [])
        self.assertTrue(all(
            item["reason"].startswith("path_group_token_budget")
            for item in tight.dropped if item["segment_id"].startswith("path-")
        ))
        incomplete = ContextEngine().build(
            query="项目如何证明岗位匹配？",
            system_prompt="仅根据证据回答。",
            retrieved_results=results[:1],
            config=ContextEngineConfig(token_budget=300, mode="no_memory"),
        )
        self.assertEqual(incomplete.evidence.used_chunk_ids, [])
        self.assertIn("incomplete_path_group", [item["reason"] for item in incomplete.dropped])

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

    def test_adaptive_plan_controls_layers_and_deduplicates_memory(self) -> None:
        memory = MemoryItem(
            "mem1", "u1", "preference", "候选人优先广州岗位", "chat", 0.9,
            self.now.isoformat(), similarity=0.95,
        )
        context = ContextEngine().build(
            query="这个城市的岗位呢？",
            system_prompt="仅依据上下文回答。",
            retrieved_results=[],
            config=ContextEngineConfig(token_budget=120, mode="adaptive"),
            history=self.history[-2:],
            memories=(memory,),
            history_summary="候选人优先广州岗位",
            plan=ContextPlan(
                use_summary=True, recent_history_count=2, memory_top_k=1
            ),
            signals=ContextSignals(0.3, 0.9, 0.8),
        )

        self.assertEqual(context.recalled_memory_ids, ("mem1",))
        self.assertNotIn("history-summary", context.kept_ids)
        self.assertTrue(any(
            item["reason"] == "cross_layer_duplicate:mem1" for item in context.dropped
        ))
        self.assertEqual(context.context_plan.memory_top_k, 1)


if __name__ == "__main__":
    unittest.main()
