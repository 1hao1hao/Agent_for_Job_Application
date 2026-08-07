from __future__ import annotations

from dataclasses import dataclass

from intern_rag.routing.base import Router
from intern_rag.routing.intent_router import RouteDecision


@dataclass(frozen=True)
class HybridRouterConfig:
    """控制语义结果何时可以覆盖规则结果。"""

    semantic_override_score: float = 0.55
    semantic_override_margin: float = 0.04
    max_weak_rule_keywords: int = 2


class HybridRouter:
    """保留强规则的可解释性，并用语义路由补足弱规则和改写 Query。"""

    def __init__(
        self,
        rule_router: Router,
        semantic_router: Router,
        *,
        config: HybridRouterConfig = HybridRouterConfig(),
    ) -> None:
        self.rule_router = rule_router
        self.semantic_router = semantic_router
        self.config = config

    def __call__(self, query: str) -> RouteDecision:
        """融合 Rule 与 Semantic 决策并保存两路理由。"""

        rule = self.rule_router(query)
        semantic = self.semantic_router(query)
        semantic_margin = float(semantic.details.get("margin", 0.0))
        semantic_strong = (
            semantic.intent != "unknown"
            and semantic.confidence is not None
            and semantic.confidence >= self.config.semantic_override_score
            and semantic_margin >= self.config.semantic_override_margin
        )
        rule_is_weak = (
            rule.intent == "unknown"
            or len(rule.matched_keywords) <= self.config.max_weak_rule_keywords
        )
        if rule.intent == semantic.intent and rule.intent != "unknown":
            selected = rule
            reason = "rule_semantic_agree"
        elif semantic_strong and rule_is_weak:
            selected = semantic
            reason = "semantic_overrode_weak_rule"
        else:
            selected = rule
            reason = "rule_preserved"

        return RouteDecision(
            intent=selected.intent,
            routed_sources=list(selected.routed_sources),
            matched_keywords=list(rule.matched_keywords),
            strategy="hybrid",
            confidence=selected.confidence,
            reason=reason,
            details={
                "rule_intent": rule.intent,
                "semantic_intent": semantic.intent,
                "semantic_score": semantic.confidence,
                "semantic_margin": semantic_margin,
                "rule_keyword_count": len(rule.matched_keywords),
            },
        )
