from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from intern_rag.retrieval import RetrievalResult
from intern_rag.routing import RouteDecision


EvidenceStatus = Literal["sufficient", "retryable", "insufficient"]
EvidenceReason = Literal[
    "sufficient_evidence",
    "unanswerable_route",
    "empty_retrieval",
    "weak_retrieval_score",
    "required_sources_missing",
]


@dataclass(frozen=True)
class EvidenceConfig:
    """Evidence Gate 的可复现阈值配置，只允许在 dev 上调整。"""

    min_results: int = 1
    min_scores: Mapping[str, float] = field(
        default_factory=lambda: {
            "keyword": 0.1,
            "bm25": 0.0,
            "dense": 0.45,
            "hybrid": 0.02,
            "bm25_hybrid": 0.02,
        }
    )
    require_source_coverage: bool = True

    def __post_init__(self) -> None:
        if self.min_results <= 0:
            raise ValueError("min_results must be greater than 0")


@dataclass(frozen=True)
class EvidenceDecision:
    """生成前的证据充分性决定。"""

    status: EvidenceStatus
    reason: EvidenceReason
    message: str
    observed_sources: list[str]
    missing_sources: list[str]
    top_score: float | None
    retry_count: int


def check_evidence(
    route: RouteDecision,
    results: list[RetrievalResult],
    *,
    retriever_name: str,
    retry_count: int,
    max_retries: int,
    config: EvidenceConfig = EvidenceConfig(),
) -> EvidenceDecision:
    """判断证据应进入生成、扩源重试还是明确拒答。"""

    observed_sources = sorted({result.chunk.source_type for result in results})
    required_sources = set(route.routed_sources)
    missing_sources = sorted(required_sources - set(observed_sources))
    top_score = max((result.score for result in results), default=None)

    if route.intent == "unknown" or not route.routed_sources:
        return _decision(
            "insufficient",
            "unanswerable_route",
            "Router 未识别出知识库支持的意图，不调用生成器。",
            observed_sources,
            missing_sources,
            top_score,
            retry_count,
        )
    if len(results) < config.min_results:
        return _retry_or_stop(
            "empty_retrieval",
            "检索结果为空，先扩展到全部来源重试一次。",
            observed_sources,
            missing_sources,
            top_score,
            retry_count,
            max_retries,
        )
    threshold = config.min_scores.get(retriever_name)
    if threshold is not None and top_score is not None and top_score < threshold:
        return _retry_or_stop(
            "weak_retrieval_score",
            f"最高检索分数 {top_score:.6f} 低于 {threshold:.6f}。",
            observed_sources,
            missing_sources,
            top_score,
            retry_count,
            max_retries,
        )
    if config.require_source_coverage and missing_sources:
        return _retry_or_stop(
            "required_sources_missing",
            "检索结果没有覆盖 Router 要求的全部来源。",
            observed_sources,
            missing_sources,
            top_score,
            retry_count,
            max_retries,
        )
    return _decision(
        "sufficient",
        "sufficient_evidence",
        "检索数量、分数和来源覆盖均满足当前门槛。",
        observed_sources,
        [],
        top_score,
        retry_count,
    )


def _retry_or_stop(
    reason: EvidenceReason,
    message: str,
    observed_sources: list[str],
    missing_sources: list[str],
    top_score: float | None,
    retry_count: int,
    max_retries: int,
) -> EvidenceDecision:
    status: EvidenceStatus = (
        "retryable" if retry_count < max_retries else "insufficient"
    )
    return _decision(
        status,
        reason,
        message,
        observed_sources,
        missing_sources,
        top_score,
        retry_count,
    )


def _decision(
    status: EvidenceStatus,
    reason: EvidenceReason,
    message: str,
    observed_sources: list[str],
    missing_sources: list[str],
    top_score: float | None,
    retry_count: int,
) -> EvidenceDecision:
    return EvidenceDecision(
        status=status,
        reason=reason,
        message=message,
        observed_sources=observed_sources,
        missing_sources=missing_sources,
        top_score=top_score,
        retry_count=retry_count,
    )
