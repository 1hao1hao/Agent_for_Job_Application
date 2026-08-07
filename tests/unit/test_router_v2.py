import unittest

from intern_rag.routing import (
    HybridRouter,
    HybridRouterConfig,
    SemanticRouter,
    SemanticRouterConfig,
    route_query,
)


class FakeEmbedding:
    """按文本中的意图信号返回可预测向量。"""

    name = "fake-router-embedding"
    version = "v1"

    def encode(self, texts):
        return [self._encode(text) for text in texts]

    @staticmethod
    def _encode(text):
        if "无关" in text or "资料之外" in text or "知识库无法" in text:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        if "面试" in text or "追问" in text:
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if "简历" in text or "候选人" in text:
            return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        if "项目" in text or "架构" in text:
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if "投递" in text or "求职" in text or "用户画像" in text:
            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class RouterV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.semantic = SemanticRouter(
            FakeEmbedding(),
            prototypes={
                "analyze_jd": ("岗位职责",),
                "match_resume": ("简历优势",),
                "interview_prepare": ("面试追问",),
                "project_explain": ("项目架构",),
                "application_plan": ("投递计划",),
                "unknown": ("资料之外",),
            },
            config=SemanticRouterConfig(min_score=0.5, min_margin=0.1),
        )

    def test_semantic_router_matches_interview_intent(self) -> None:
        decision = self.semantic("面试官会怎样追问这个模块？")

        self.assertEqual(decision.intent, "interview_prepare")
        self.assertEqual(decision.strategy, "semantic")
        self.assertGreaterEqual(decision.confidence, 0.5)

    def test_hybrid_router_overrides_weak_generic_rule(self) -> None:
        router = HybridRouter(
            route_query,
            self.semantic,
            config=HybridRouterConfig(
                semantic_override_score=0.5,
                semantic_override_margin=0.1,
                max_weak_rule_keywords=2,
            ),
        )

        decision = router("面试时如何结合岗位回答问题")

        self.assertEqual(decision.intent, "interview_prepare")
        self.assertEqual(decision.reason, "semantic_overrode_weak_rule")
        self.assertEqual(decision.details["rule_intent"], "analyze_jd")

    def test_hybrid_preserves_rule_when_semantic_is_not_confident(self) -> None:
        uncertain_semantic = SemanticRouter(
            FakeEmbedding(),
            prototypes={
                "analyze_jd": ("岗位职责",),
                "match_resume": ("简历优势",),
                "interview_prepare": ("面试追问",),
                "project_explain": ("项目架构",),
                "application_plan": ("投递计划",),
                "unknown": ("资料之外",),
            },
            config=SemanticRouterConfig(min_score=1.1 - 0.1, min_margin=1.1),
        )
        router = HybridRouter(route_query, uncertain_semantic)

        decision = router("分析岗位职责和要求")

        self.assertEqual(decision.intent, "analyze_jd")
        self.assertEqual(decision.reason, "rule_preserved")


if __name__ == "__main__":
    unittest.main()
