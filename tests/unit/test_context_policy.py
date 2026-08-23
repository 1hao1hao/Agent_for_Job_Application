from __future__ import annotations

from datetime import datetime, timezone
import unittest

from intern_rag.agent import (
    ContextInputs,
    ContextPolicy,
    ContextPolicyConfig,
    ContextSignalExtractor,
    ContextSignals,
    ConversationMessage,
    MemoryItem,
)


class FakeEmbeddingModel:
    """用二维向量稳定表达测试中的语义连续与无关文本。"""

    def encode(self, texts):
        output = []
        for text in texts:
            if any(word in text for word in ("广州", "城市", "这个", "岗位")):
                output.append([1.0, 0.0])
            else:
                output.append([0.0, 1.0])
        return output


class ContextPolicyTests(unittest.TestCase):
    def test_signal_extractor_combines_reference_semantics_and_memory(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        inputs = ContextInputs(
            history=(
                ConversationMessage("m1", "s1", "u1", "user", "我优先广州岗位", now),
            ),
            memories=(
                MemoryItem(
                    "mem1", "u1", "preference", "优先广州城市岗位", "chat", 0.8, now
                ),
            ),
        )

        signals = ContextSignalExtractor(FakeEmbeddingModel()).extract(
            "这个城市的岗位呢？", inputs, token_budget=20
        )

        self.assertEqual(signals.reference_score, 1.0)
        self.assertEqual(signals.semantic_continuity, 1.0)
        self.assertEqual(signals.followup_score, 1.0)
        self.assertAlmostEqual(signals.memory_score, 0.8)
        self.assertGreater(signals.history_token_pressure, 0.0)

    def test_policy_selects_summary_recent_and_memory_for_combined_case(self) -> None:
        policy = ContextPolicy(
            ContextPolicyConfig(
                followup_threshold=0.7,
                memory_threshold=0.45,
                long_history_threshold=0.5,
            )
        )

        plan = policy.decide(ContextSignals(0.8, 0.9, 0.7))

        self.assertTrue(plan.use_profile)
        self.assertTrue(plan.use_summary)
        self.assertEqual(plan.recent_history_count, 1)
        self.assertEqual(plan.memory_top_k, 1)
        self.assertEqual(plan.reason["followup_score"], 0.9)

    def test_independent_query_skips_history_summary_and_memory(self) -> None:
        plan = ContextPolicy().decide(ContextSignals(0.02, 0.1, 0.1))

        self.assertTrue(plan.use_profile)
        self.assertFalse(plan.use_summary)
        self.assertEqual(plan.recent_history_count, 0)
        self.assertEqual(plan.memory_top_k, 0)


if __name__ == "__main__":
    unittest.main()
