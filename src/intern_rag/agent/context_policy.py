from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Protocol, Sequence

from intern_rag.agent.context_engine import (
    ContextInputs,
    MemoryItem,
    MixedTokenEstimator,
    TokenEstimator,
)


class ContextEmbeddingModel(Protocol):
    """Context 信号提取依赖的最小 embedding 接口。"""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """把 Query、历史和 Memory 编码为同一向量空间。"""


@dataclass(frozen=True)
class ContextSignals:
    """ContextPolicy 在一次请求中使用的三个可解释状态信号。"""

    history_token_pressure: float
    followup_score: float
    memory_score: float
    reference_score: float = 0.0
    semantic_continuity: float = 0.0
    history_token_count: int = 0


@dataclass(frozen=True)
class ContextPlan:
    """本轮需要启用哪些记忆层，以及每层最多取多少。"""

    use_profile: bool = True
    use_summary: bool = False
    recent_history_count: int = 4
    memory_top_k: int = 0
    reason: dict[str, float] = field(default_factory=dict)
    policy_version: str = "adaptive-context-policy-v1"

    def __post_init__(self) -> None:
        if self.recent_history_count < 0 or self.memory_top_k < 0:
            raise ValueError("context plan counts must not be negative")


@dataclass(frozen=True)
class ContextPolicyConfig:
    """只在 dev 上校准的自适应上下文阈值。"""

    followup_threshold: float = 0.70
    memory_threshold: float = 0.35
    long_history_threshold: float = 0.50
    summary_pressure_threshold: float = 0.08
    recent_history_high: int = 1
    recent_history_low: int = 0
    memory_top_k: int = 1
    version: str = "adaptive-context-policy-v1"


class ContextSignalExtractor:
    """从 Query 与 ContextInputs 提取历史压力、前文依赖和记忆相关度。

    历史压力是历史 token 数除以上下文预算；前文依赖由指代/省略规则与 Query-
    History 语义连续性各占 50%；长期记忆分数取有效 Memory 的
    `similarity * importance` 最大值。没有 embedding 时使用确定性 token Jaccard
    回退，保证服务受控运行，但正式实验使用固定 BGE embedding。
    """

    REFERENCE_PATTERN = re.compile(
        r"(它|这个|那个|上述|前面|刚才|之前|继续|为什么|还有呢|然后呢|"
        r"这条|那个条件|这个安排|我的稳定条件|最开始|以最新|结合个人条件)"
    )

    def __init__(
        self,
        embedding_model: ContextEmbeddingModel | None = None,
        estimator: TokenEstimator | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.estimator = estimator or MixedTokenEstimator()

    def extract(
        self,
        query: str,
        inputs: ContextInputs,
        *,
        token_budget: int,
    ) -> ContextSignals:
        """计算本轮三个策略信号，分数统一限制在 0 到 1。"""

        if token_budget <= 0:
            raise ValueError("token_budget must be greater than 0")
        history_texts = [item.content for item in inputs.history]
        history_tokens = sum(self.estimator.count(text) for text in history_texts)
        pressure = min(history_tokens / token_budget, 1.0)
        reference_score = 1.0 if self.REFERENCE_PATTERN.search(query) else 0.0
        recent_texts = history_texts[-4:]
        semantic_continuity = max(self.similarities(query, recent_texts), default=0.0)
        followup_score = _clamp(0.5 * reference_score + 0.5 * semantic_continuity)

        available_memories = [item for item in inputs.memories if item.is_available]
        fallback_scores = self.similarities(
            query, [item.content for item in available_memories]
        )
        memory_score = max(
            (
                (item.similarity if item.similarity is not None else similarity)
                * item.importance
                for item, similarity in zip(available_memories, fallback_scores)
            ),
            default=0.0,
        )
        return ContextSignals(
            history_token_pressure=pressure,
            followup_score=followup_score,
            memory_score=_clamp(memory_score),
            reference_score=reference_score,
            semantic_continuity=_clamp(semantic_continuity),
            history_token_count=history_tokens,
        )

    def similarities(self, query: str, texts: Sequence[str]) -> list[float]:
        """返回 Query 与多段文本的余弦相似度，供信号和跨层去重共用。"""

        if not texts:
            return []
        if self.embedding_model is None:
            return [_token_jaccard(query, text) for text in texts]
        vectors = self.embedding_model.encode([query, *texts])
        return [_cosine(vectors[0], vector) for vector in vectors[1:]]


class ContextPolicy:
    """把可解释状态信号转换成 ContextPlan，不负责实际预算裁剪。"""

    def __init__(self, config: ContextPolicyConfig | None = None) -> None:
        self.config = config or ContextPolicyConfig()

    def decide(self, signals: ContextSignals) -> ContextPlan:
        """按前文依赖、历史压力和长期记忆相关度动态选择记忆层。"""

        followup = signals.followup_score >= self.config.followup_threshold
        long_history = signals.history_token_pressure > self.config.long_history_threshold
        use_summary = long_history or (
            followup
            and signals.history_token_pressure >= self.config.summary_pressure_threshold
        )
        recent_count = (
            self.config.recent_history_high
            if followup
            else self.config.recent_history_low
        )
        if long_history:
            recent_count = min(recent_count, 2)
        memory_top_k = (
            self.config.memory_top_k
            if signals.memory_score >= self.config.memory_threshold
            else 0
        )
        return ContextPlan(
            use_profile=True,
            use_summary=use_summary,
            recent_history_count=recent_count,
            memory_top_k=memory_top_k,
            reason={
                "history_token_pressure": signals.history_token_pressure,
                "followup_score": signals.followup_score,
                "memory_score": signals.memory_score,
            },
            policy_version=self.config.version,
        )


def rank_memories(
    query: str,
    memories: Sequence[MemoryItem],
    extractor: ContextSignalExtractor,
) -> tuple[MemoryItem, ...]:
    """为候选 Memory 补充语义分数，并按 similarity * importance 排序。"""

    available = [item for item in memories if item.is_available]
    scores = extractor.similarities(query, [item.content for item in available])
    scored = [
        MemoryItem(**{**item.__dict__, "similarity": _clamp(score)})
        for item, score in zip(available, scores)
    ]
    return tuple(
        sorted(
            scored,
            key=lambda item: (
                -float(item.similarity or 0.0) * item.importance,
                -item.version,
                item.memory_id,
            ),
        )
    )


def _token_jaccard(left: str, right: str) -> float:
    pattern = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
    left_tokens = set(pattern.findall(left.lower()))
    right_tokens = set(pattern.findall(right.lower()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) ** 2 for value in left) ** 0.5
    right_norm = sum(float(value) ** 2 for value in right) ** 0.5
    return _clamp(numerator / (left_norm * right_norm)) if left_norm and right_norm else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
