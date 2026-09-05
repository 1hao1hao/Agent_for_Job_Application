from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal, Mapping

from intern_rag.retrieval import RetrievalResult
from intern_rag.routing import RouteDecision


EvidenceStatus = Literal["sufficient", "retryable", "insufficient"]
EvidenceReason = Literal[
    "sufficient_evidence", "unanswerable_route", "empty_retrieval",
    "weak_retrieval_score", "low_retrieval_confidence",
    "required_sources_missing", "graph_evidence_missing",
]


@dataclass(frozen=True)
class ScoreGateConfig:
    """某个底层 Retriever 经 dev 校准后的分数配置。

    ``threshold`` 仅保留旧配置兼容；v2.1 使用
    ``low_confidence_retry_threshold`` 触发一次重试，不把 raw score 作为最终拒答条件。
    """

    enabled: bool
    threshold: float | None
    status: str = "calibrated"
    low_confidence_retry_threshold: float | None = None


@dataclass(frozen=True)
class EvidenceConfig:
    """版本化 Evidence Gate 配置；旧的 ``min_scores`` 调用仍受支持。"""

    min_results: int = 1
    min_scores: Mapping[str, float] = field(default_factory=lambda: {
        "keyword": 0.1, "bm25": 0.0, "dense": 0.45,
        "hybrid": 0.02, "bm25_hybrid": 0.02,
    })
    require_source_coverage: bool = True
    config_version: str = "legacy"
    calibrated_scores: Mapping[str, ScoreGateConfig] = field(default_factory=dict)
    need_min_results: Mapping[str, int] = field(default_factory=dict)
    max_frr_for_score_gate: float = 0.25

    def __post_init__(self) -> None:
        if self.min_results <= 0:
            raise ValueError("min_results must be greater than 0")
        if not 0.0 <= self.max_frr_for_score_gate <= 1.0:
            raise ValueError("max_frr_for_score_gate must be between 0 and 1")


@dataclass(frozen=True)
class EvidenceDecision:
    """生成前决定，并携带足以复查门控依据的结构化 Trace。"""

    status: EvidenceStatus
    reason: EvidenceReason
    message: str
    observed_sources: list[str]
    missing_sources: list[str]
    top_score: float | None
    retry_count: int
    evidence_need: str = "legacy"
    effective_retriever: str = "unknown"
    threshold: float | None = None
    threshold_status: str = "legacy"
    calibrated_retry_threshold: float | None = None
    low_confidence: bool = False
    low_confidence_after_retry: bool = False
    relation_evidence_present: bool = False
    structural_checks: Mapping[str, bool] = field(default_factory=dict)
    config_version: str = "legacy"


def load_evidence_config(path: Path) -> EvidenceConfig:
    """从 JSON 读取锁定 Gate 配置；不会静默读取 test 或修改配置。"""

    raw = json.loads(path.read_text(encoding="utf-8"))
    calibrated = {
        str(name): ScoreGateConfig(
            enabled=bool(value.get("enabled", False)),
            threshold=float(value["threshold"]) if value.get("threshold") is not None else None,
            status=str(value.get("status", "unknown")),
            low_confidence_retry_threshold=(
                float(value["low_confidence_retry_threshold"])
                if value.get("low_confidence_retry_threshold") is not None
                else None
            ),
        )
        for name, value in dict(raw.get("retrievers", {})).items()
    }
    return EvidenceConfig(
        min_results=int(raw.get("min_results", 1)),
        min_scores={},
        require_source_coverage=bool(raw.get("legacy_source_coverage", False)),
        config_version=str(raw.get("config_version", "unknown")),
        calibrated_scores=calibrated,
        need_min_results={str(k): int(v) for k, v in dict(raw.get("need_min_results", {})).items()},
        max_frr_for_score_gate=float(raw.get("max_frr_for_score_gate", 0.25)),
    )


def check_evidence(
    route: RouteDecision,
    results: list[RetrievalResult],
    *,
    retriever_name: str,
    retry_count: int,
    max_retries: int,
    config: EvidenceConfig = EvidenceConfig(),
    evidence_requirement: Mapping[str, object] | None = None,
    retrieval_trace: Mapping[str, object] | None = None,
) -> EvidenceDecision:
    """按证据需求检查数量、来源覆盖、图路径和低置信重试信号。

    输入 Router、排序结果、EvidenceRequirement 和 Retriever Trace。函数识别实际
    底层策略；结构证据缺失时重试或拒答。结构条件满足但 raw top-1 score 低于
    本 Retriever 的 dev 阈值时，首次仅触发一次扩源重试；重试后不再因分数偏低
    拒答，而是在 Trace 中留下低置信标记。
    """

    requirement = dict(evidence_requirement or {})
    trace = dict(retrieval_trace or {})
    trace_requirement = trace.get("evidence_requirement")
    if not requirement and isinstance(trace_requirement, dict):
        requirement = dict(trace_requirement)
    need = str(requirement.get("need_type", "legacy"))
    effective = str(trace.get("selected_strategy", trace.get("strategy", retriever_name)))
    effective = {"graph_hybrid": "graph_vector", "adaptive": retriever_name}.get(effective, effective)
    observed_sources = sorted({result.chunk.source_type for result in results})
    required_sources = set(route.routed_sources)
    missing_sources = sorted(required_sources - set(observed_sources))
    top_score = results[0].score if results else None
    min_results = int(config.need_min_results.get(need, config.min_results))
    graph_path_valid = any(
        bool(result.details.get("path_valid")) and bool(result.details.get("graph_edge_ids"))
        for result in results
    )
    needs_source_coverage = (
        need == "multi_source_synthesis"
        or bool(requirement.get("multi_source_required", False))
        or (need == "legacy" and config.require_source_coverage)
    )
    structural_checks = {
        "min_results": len(results) >= min_results,
        "source_coverage": not needs_source_coverage or not missing_sources,
        "graph_path": need != "relation_reasoning" or graph_path_valid,
    }
    score_gate = config.calibrated_scores.get(effective)
    threshold = score_gate.threshold if score_gate and score_gate.enabled else None
    threshold_status = score_gate.status if score_gate else "legacy"
    retry_threshold = (
        score_gate.low_confidence_retry_threshold if score_gate else None
    )
    if score_gate is None:
        threshold = config.min_scores.get(effective, config.min_scores.get(retriever_name))
    low_confidence = bool(
        retry_threshold is not None
        and top_score is not None
        and top_score < retry_threshold
    )

    common = {
        "observed_sources": observed_sources, "missing_sources": missing_sources,
        "top_score": top_score, "retry_count": retry_count, "evidence_need": need,
        "effective_retriever": effective, "threshold": threshold,
        "threshold_status": threshold_status,
        "calibrated_retry_threshold": retry_threshold,
        "low_confidence": low_confidence,
        "low_confidence_after_retry": low_confidence and retry_count >= max_retries,
        "relation_evidence_present": graph_path_valid,
        "structural_checks": structural_checks,
        "config_version": config.config_version,
    }
    if need == "unanswerable" or (
        (route.intent == "unknown" or not route.routed_sources) and not results
    ):
        return _decision("insufficient", "unanswerable_route", "知识库不支持该问题，不调用生成器。", common)
    if len(results) < min_results:
        return _retry_or_stop("empty_retrieval", "检索结果数量不足，扩展来源后最多重试一次。", retry_count, max_retries, common)
    if needs_source_coverage and missing_sources:
        return _retry_or_stop("required_sources_missing", "多来源证据没有覆盖全部必要来源。", retry_count, max_retries, common)
    if need == "relation_reasoning" and not graph_path_valid:
        return _retry_or_stop("graph_evidence_missing", "关系问题缺少有效 Graph path 或关系边证据。", retry_count, max_retries, common)
    if low_confidence and retry_count < max_retries:
        return _decision(
            "retryable", "low_retrieval_confidence",
            f"结构证据满足，但最高分 {top_score:.6f} 低于本检索器重试阈值 {retry_threshold:.6f}。",
            common,
        )
    if low_confidence:
        return _decision(
            "sufficient", "sufficient_evidence",
            "结构证据满足；扩源后仍为低分，仅记录低置信，不据此拒答。",
            common,
        )
    if threshold is not None and top_score is not None and top_score < threshold:
        # 旧 min_scores 配置保持原行为；新 calibration 配置不会启用 hard gate。
        return _retry_or_stop(
            "weak_retrieval_score",
            f"最高检索分数 {top_score:.6f} 低于旧配置门槛 {threshold:.6f}。",
            retry_count,
            max_retries,
            common,
        )
    return _decision("sufficient", "sufficient_evidence", "数量、可用门槛和结构证据均满足要求。", common)


def _retry_or_stop(
    reason: EvidenceReason,
    message: str,
    retry_count: int,
    max_retries: int,
    fields: dict[str, object],
) -> EvidenceDecision:
    status: EvidenceStatus = "retryable" if retry_count < max_retries else "insufficient"
    return _decision(status, reason, message, fields)


def _decision(
    status: EvidenceStatus,
    reason: EvidenceReason,
    message: str,
    fields: dict[str, object],
) -> EvidenceDecision:
    return EvidenceDecision(status=status, reason=reason, message=message, **fields)  # type: ignore[arg-type]
