from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from intern_rag.retrieval.dense import EmbeddingModel
from intern_rag.routing.intent_router import INTENT_TO_SOURCES, Intent, RouteDecision


DEFAULT_INTENT_PROTOTYPES: dict[Intent, tuple[str, ...]] = {
    "analyze_jd": (
        "分析岗位职责、任职要求和招聘条件",
        "这个职位需要哪些技术和实习条件",
        "解读公司发布的实习岗位",
    ),
    "match_resume": (
        "结合我的简历分析与岗位的匹配度和差距",
        "我的经历有哪些优势，简历应该怎样修改",
        "候选人的项目经验是否胜任这个职位",
    ),
    "interview_prepare": (
        "准备面试追问、面经和技术问题回答",
        "面试时怎样结合岗位和个人经历作答",
        "这个技术点面试官可能怎样深入追问",
    ),
    "project_explain": (
        "向面试官讲解项目架构、模块、亮点和难点",
        "这个项目的实现过程和技术取舍怎么讲",
        "结合项目日志总结一次工程迭代",
    ),
    "application_plan": (
        "根据目标城市、时间安排和个人偏好制定投递计划",
        "规划下一阶段实习求职和岗位申请",
        "结合用户画像选择适合投递的公司岗位",
    ),
    "unknown": (
        "询问资料之外的实时信息、隐私名单或无关事实",
        "问题与求职、岗位、简历、面试和项目完全无关",
        "当前知识库无法回答的外部信息",
    ),
}


@dataclass(frozen=True)
class SemanticRouterConfig:
    """Semantic Router 的阈值配置，只允许在 dev 上校准。"""

    min_score: float = 0.45
    min_margin: float = 0.02

    def __post_init__(self) -> None:
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")
        if self.min_margin < 0:
            raise ValueError("min_margin must not be negative")


class SemanticRouter:
    """用固定意图原型与 Query 的向量相似度执行语义路由。"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        prototypes: Mapping[Intent, Sequence[str]] = DEFAULT_INTENT_PROTOTYPES,
        config: SemanticRouterConfig = SemanticRouterConfig(),
    ) -> None:
        if not prototypes:
            raise ValueError("prototypes must not be empty")
        self.embedding_model = embedding_model
        self.config = config
        self.prototypes = {
            intent: tuple(texts) for intent, texts in prototypes.items()
        }
        if set(self.prototypes) != set(INTENT_TO_SOURCES):
            raise ValueError("prototypes must cover every supported intent")
        flattened = [
            text
            for intent in INTENT_TO_SOURCES
            for text in self.prototypes[intent]
        ]
        vectors = embedding_model.encode(flattened)
        self._prototype_vectors: dict[Intent, list[list[float]]] = {}
        offset = 0
        for intent in INTENT_TO_SOURCES:
            count = len(self.prototypes[intent])
            if count == 0:
                raise ValueError(f"intent {intent} has no prototype")
            self._prototype_vectors[intent] = vectors[offset : offset + count]
            offset += count

    def __call__(self, query: str) -> RouteDecision:
        """选择与 Query 最相似的意图；低置信度或小间隔时返回 unknown。"""

        if not query.strip():
            return self._decision("unknown", 0.0, 0.0, {})
        query_vector = self.embedding_model.encode([query])[0]
        scores = {
            intent: max(_dot(query_vector, vector) for vector in vectors)
            for intent, vectors in self._prototype_vectors.items()
        }
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_intent, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_score - second_score
        if best_score < self.config.min_score or margin < self.config.min_margin:
            best_intent = "unknown"
        return self._decision(best_intent, best_score, margin, scores)

    @staticmethod
    def _decision(
        intent: Intent,
        score: float,
        margin: float,
        scores: Mapping[Intent, float],
    ) -> RouteDecision:
        return RouteDecision(
            intent=intent,
            routed_sources=INTENT_TO_SOURCES[intent],
            matched_keywords=[],
            strategy="semantic",
            confidence=score,
            reason=(
                "semantic_threshold_not_met"
                if intent == "unknown"
                else "semantic_prototype_matched"
            ),
            details={
                "margin": margin,
                "intent_scores": {
                    key: round(value, 6) for key, value in scores.items()
                },
            },
        )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """计算已归一化向量的点积。"""

    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right))
