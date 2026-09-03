from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
import re
from typing import Literal, Mapping

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.base import RetrievalResult, Retriever
from intern_rag.retrieval.rerank import RerankScorer


RetrievalStrategy = Literal["none", "bm25", "dense", "hybrid", "graph_hybrid"]
RerankPolicy = Literal["never", "always", "low_confidence"]
EvidenceNeedType = Literal[
    "unanswerable",
    "exact_fact",
    "semantic_explanation",
    "relation_reasoning",
    "multi_source_synthesis",
    "balanced_retrieval",
]


@dataclass(frozen=True)
class QueryFeatures:
    """四个核心证据需求信号，以及实体和路由两个上下文信息。"""

    needs_exact_match: bool
    needs_semantic_match: bool
    needs_multi_source: bool
    requires_entity_reasoning: bool
    entity_types: tuple[str, ...]
    is_unanswerable_route: bool
    has_conflicting_signals: bool = False


@dataclass(frozen=True)
class QueryAnalyzerConfig:
    """Query Analyzer v2 的少量强信号与版本号。"""

    version: str = "query-analyzer-v2.0"
    semantic_markers: tuple[str, ...] = (
        "换句话", "同义", "通俗", "口语", "改写", "概括", "转述",
        "如何", "怎么", "为什么", "原理", "解释", "提升", "降低", "减少", "优化",
    )
    multi_source_markers: tuple[str, ...] = (
        "结合", "对比", "比较", "综合", "分别", "共同分析",
    )
    relation_markers: tuple[str, ...] = (
        "哪些项目", "哪个项目", "哪些经历", "哪段经历", "能否证明", "是否证明",
        "如何证明", "对应关系", "关联起来", "根据经历推荐", "关系路径", "关系链",
        "跳联系", "多跳", "共同体现", "为什么符合", "为何符合",
    )
    weak_relation_markers: tuple[str, ...] = (
        "匹配", "适合这个岗位", "符合这个岗位", "关联",
    )
    exact_fact_markers: tuple[str, ...] = (
        "是什么", "有哪些", "多少", "何时", "哪里", "职责", "要求", "版本",
    )
    exact_terms: tuple[str, ...] = (
        "bm25", "rrf", "redis stream", "pgvector", "neo4j", "crossencoder",
        "cross-encoder", "fastapi", "docker compose", "citation", "trace id",
    )
    unanswerable_markers: tuple[str, ...] = (
        "unpublished-", "内部薪资审批名单", "量子芯片驱动的十亿节点生产图",
    )
    latin_token_is_exact: bool = False
    entity_markers: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: {
        "job": ("岗位", "职位", "jd", "招聘"),
        "skill": ("技能", "能力", "要求", "技术栈"),
        "project": ("项目", "作品", "系统"),
        "experience": ("经历", "简历", "经验", "候选人", "个人背景"),
        "company": ("公司", "企业", "部门"),
    })


@dataclass(frozen=True)
class StrategySelectionConfig:
    """证据需求到检索策略的版本化映射。"""

    version: str = "strategy-map-v2.0"
    semantic_strategy: Literal["dense", "hybrid"] = "hybrid"


@dataclass(frozen=True)
class EvidenceRequirement:
    """描述当前问题需要哪类证据，而不是直接描述某个检索算法。"""

    need_type: EvidenceNeedType
    lexical_required: bool
    semantic_required: bool
    graph_required: bool
    multi_source_required: bool
    reason: str


@dataclass(frozen=True)
class RetrievalDecision:
    """记录一次自适应检索选择、置信度和重排决策。"""

    strategy: RetrievalStrategy
    confidence: float
    rerank_invoked: bool
    rerank_applied: bool
    candidate_count: int
    reason: str
    reranker_name: str | None = None
    reranker_version: str | None = None
    query_features: dict[str, object] | None = None
    evidence_requirement: dict[str, object] | None = None
    selection_rule: str = ""
    config_version: str = "legacy"
    fallback_reason: str | None = None
    escalation_reason: str | None = None

    def to_trace(self) -> dict[str, object]:
        """转换为可写入 AgentTrace/评测工件的普通字典。"""

        value = asdict(self)
        value["selected_strategy"] = self.strategy
        return value


@dataclass(frozen=True)
class AdaptiveRetrieverConfig:
    """自适应策略、置信度门槛与保守重排融合配置。"""

    confidence_threshold: float = 0.55
    candidate_k: int = 20
    rerank_rrf_k: int = 60
    original_rank_weight: float = 2.0
    rerank_rank_weight: float = 1.0
    rerank_policy: RerankPolicy = "low_confidence"
    force_strategy: RetrievalStrategy | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.candidate_k <= 0 or self.rerank_rrf_k <= 0:
            raise ValueError("candidate_k and rerank_rrf_k must be positive")
        if self.original_rank_weight <= 0 or self.rerank_rank_weight <= 0:
            raise ValueError("rerank weights must be positive")
        if self.rerank_policy not in {"never", "always", "low_confidence"}:
            raise ValueError(f"unknown rerank policy: {self.rerank_policy}")


class QueryAnalyzer:
    """将 Query 转成可解释特征，再判断它真正需要哪类证据。

    输入 Query 和 Router 已限定的来源，先提取精确术语、语义解释、实体类型、
    关系推理和多来源综合信号，再生成 `EvidenceRequirement`。策略选择只消费
    证据需求，因此不会再用“Query 很短”这类弱信号直接决定检索算法。
    """

    def __init__(
        self,
        config: QueryAnalyzerConfig | None = None,
        strategy_config: StrategySelectionConfig | None = None,
    ) -> None:
        self.config = config or QueryAnalyzerConfig()
        self.strategy_config = strategy_config or StrategySelectionConfig()

    def analyze(
        self,
        query: str,
        source_types: set[str] | None,
    ) -> QueryFeatures:
        """提取词面、语义、实体关系和来源需求信号。"""

        normalized = query.strip().lower()
        sources = source_types or set()
        entity_types = tuple(
            entity_type
            for entity_type, markers in self.config.entity_markers.items()
            if any(marker in normalized for marker in markers)
        )
        strong_relations = tuple(
            marker for marker in self.config.relation_markers if marker in normalized
        )
        weak_relations = tuple(
            marker for marker in self.config.weak_relation_markers if marker in normalized
        )
        needs_exact_match = (
            any(term in normalized for term in self.config.exact_terms)
            or any(marker in normalized for marker in self.config.exact_fact_markers)
            or (
                self.config.latin_token_is_exact
                and bool(re.search(r"[A-Za-z][A-Za-z0-9_.+-]+", query))
            )
        )
        needs_semantic_match = any(
            marker in normalized for marker in self.config.semantic_markers
        )
        needs_multi_source = (
            len(sources) >= 2
            or any(marker in normalized for marker in self.config.multi_source_markers)
        )
        requires_entity_reasoning = bool(strong_relations) or (
            bool(weak_relations) and len(entity_types) >= 3
        )
        return QueryFeatures(
            needs_exact_match=needs_exact_match,
            needs_semantic_match=needs_semantic_match,
            needs_multi_source=needs_multi_source,
            requires_entity_reasoning=requires_entity_reasoning,
            entity_types=entity_types,
            is_unanswerable_route=(
                (source_types is not None and not source_types)
                or any(marker in normalized for marker in self.config.unanswerable_markers)
            ),
            has_conflicting_signals=needs_exact_match and needs_semantic_match,
        )

    def classify_evidence_need(
        self,
        features: QueryFeatures,
    ) -> EvidenceRequirement:
        """把 Query 特征归纳为事实、语义、关系或综合证据需求。"""

        if features.is_unanswerable_route:
            return EvidenceRequirement(
                "unanswerable", False, False, False, False,
                "router returned no searchable sources",
            )
        if features.requires_entity_reasoning:
            return EvidenceRequirement(
                "relation_reasoning", True, True, True,
                features.needs_multi_source or len(features.entity_types) >= 2,
                "query needs an explicit relation path between entities",
            )
        if features.needs_multi_source:
            return EvidenceRequirement(
                "multi_source_synthesis", True, True, False, True,
                "query needs evidence synthesis or comparison across sources",
            )
        if features.needs_semantic_match:
            return EvidenceRequirement(
                "semantic_explanation", False, True, False, False,
                "query asks for semantic explanation or paraphrased evidence",
            )
        if features.needs_exact_match:
            return EvidenceRequirement(
                "exact_fact", True, False, False, False,
                "query asks for an exact term or literal fact",
            )
        return EvidenceRequirement(
            "balanced_retrieval", True, True, False, False,
            "evidence need is ambiguous, so lexical and semantic recall are retained",
        )

    def choose_strategy(
        self,
        requirement: QueryFeatures | EvidenceRequirement,
        *,
        graph_enabled: bool = False,
    ) -> tuple[RetrievalStrategy, str]:
        """把证据需求映射到可用 Retriever，并对缺失图能力受控降级。"""

        evidence = (
            self.classify_evidence_need(requirement)
            if isinstance(requirement, QueryFeatures)
            else requirement
        )
        if evidence.need_type == "unanswerable":
            return "none", evidence.reason
        if evidence.graph_required:
            if graph_enabled:
                return "graph_hybrid", evidence.reason
            return "hybrid", f"{evidence.reason}; graph unavailable, use hybrid fallback"
        if evidence.need_type == "exact_fact":
            return "bm25", evidence.reason
        if evidence.need_type == "semantic_explanation":
            return self.strategy_config.semantic_strategy, evidence.reason
        return "hybrid", evidence.reason


class AdaptiveRetriever:
    """按 Query 特征选择召回策略，并仅在低置信度时执行一次保守重排。

    输入与其他 Retriever 一致。先由 QueryAnalyzer 选择 BM25、Dense 或 Hybrid，
    再根据结果数量、首位 margin、来源覆盖和 Hybrid 双路一致性计算置信度。
    低置信度候选只调用一次 scorer，并用加权 RRF 融合原始 rank 与 rerank rank，
    避免直接用 CrossEncoder 分数覆盖已验证的召回排序。输出仍是
    `list[RetrievalResult]`，每条结果附带决策信息；空结果的决策可由
    `get_last_trace()` 读取。
    """

    def __init__(
        self,
        retrievers: dict[RetrievalStrategy, Retriever],
        scorer: RerankScorer,
        *,
        analyzer: QueryAnalyzer | None = None,
        config: AdaptiveRetrieverConfig | None = None,
        graph_retriever: Retriever | None = None,
    ) -> None:
        missing = {"bm25", "dense", "hybrid"} - set(retrievers)
        if missing:
            raise ValueError(f"adaptive retriever missing strategies: {sorted(missing)}")
        self.retrievers = dict(retrievers)
        self.scorer = scorer
        self.analyzer = analyzer or QueryAnalyzer()
        self.config = config or AdaptiveRetrieverConfig()
        self.graph_retriever = graph_retriever
        self._last_decision: ContextVar[RetrievalDecision | None] = ContextVar(
            f"adaptive_retrieval_decision_{id(self)}", default=None
        )
        self._last_stage_trace: ContextVar[dict[str, object]] = ContextVar(
            f"adaptive_retrieval_stage_trace_{id(self)}", default={}
        )

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """执行策略选择、置信度判断和最多一次 CrossEncoder 重排。"""

        features = self.analyzer.analyze(query, source_types)
        evidence_requirement = self.analyzer.classify_evidence_need(features)
        strategy, strategy_reason = self.analyzer.choose_strategy(
            evidence_requirement, graph_enabled=self.graph_retriever is not None
        )
        if self.config.force_strategy is not None:
            strategy = self.config.force_strategy
            strategy_reason = f"forced {strategy} for controlled ablation"
        fallback_reason = (
            "graph capability unavailable"
            if evidence_requirement.graph_required and strategy == "hybrid"
            else None
        )
        if top_k <= 0 or not query.strip() or strategy == "none":
            decision = RetrievalDecision(
                strategy=strategy,
                confidence=1.0 if features.is_unanswerable_route else 0.0,
                rerank_invoked=False,
                rerank_applied=False,
                candidate_count=0,
                reason=strategy_reason,
                query_features=asdict(features),
                evidence_requirement=asdict(evidence_requirement),
                selection_rule=evidence_requirement.need_type,
                config_version=(
                    f"{self.analyzer.config.version}/"
                    f"{self.analyzer.strategy_config.version}"
                ),
                fallback_reason=fallback_reason,
            )
            self._last_decision.set(decision)
            self._last_stage_trace.set({})
            return []

        candidate_k = max(top_k, self.config.candidate_k)
        selected_retriever = (
            self.graph_retriever
            if strategy == "graph_hybrid"
            else self.retrievers[strategy]
        )
        if selected_retriever is None:
            selected_retriever = self.retrievers["hybrid"]
            strategy = "hybrid"
            strategy_reason = "graph retriever unavailable; fallback to hybrid"
            fallback_reason = "selected graph retriever is unavailable"
        candidates = selected_retriever(
            query, chunks, top_k=candidate_k, source_types=source_types
        )
        stage_trace = _read_retriever_trace(selected_retriever)
        confidence = _retrieval_confidence(candidates, top_k, source_types, strategy)
        can_rerank = bool(candidates) and strategy != "graph_hybrid"
        should_rerank = can_rerank and (
            self.config.rerank_policy == "always"
            or (
                self.config.rerank_policy == "low_confidence"
                and confidence < self.config.confidence_threshold
            )
        )
        ranked = candidates
        rerank_applied = False
        if should_rerank:
            ranked = self._rerank(query, candidates)
            rerank_applied = [item.chunk_id for item in ranked] != [
                item.chunk_id for item in candidates
            ]

        decision = RetrievalDecision(
            strategy=strategy,
            confidence=confidence,
            rerank_invoked=should_rerank,
            rerank_applied=rerank_applied,
            candidate_count=len(candidates),
            reason=(
                f"{strategy_reason}; confidence={confidence:.3f}; "
                + (
                    f"rerank candidates policy={self.config.rerank_policy}"
                    if should_rerank
                    else f"skip rerank policy={self.config.rerank_policy}"
                )
            ),
            reranker_name=self.scorer.name if should_rerank else None,
            reranker_version=self.scorer.version if should_rerank else None,
            query_features=asdict(features),
            evidence_requirement=asdict(evidence_requirement),
            selection_rule=evidence_requirement.need_type,
            config_version=(
                f"{self.analyzer.config.version}/"
                f"{self.analyzer.strategy_config.version}"
            ),
            fallback_reason=fallback_reason,
            escalation_reason=(
                "retrieval confidence below rerank threshold"
                if should_rerank else None
            ),
        )
        self._last_decision.set(decision)
        self._last_stage_trace.set(stage_trace)
        return _attach_decision(ranked[:top_k], decision)

    def get_last_trace(self) -> dict[str, object]:
        """返回当前执行上下文中最近一次检索决策，供 Pipeline/Runner 记录。"""

        decision = self._last_decision.get()
        return (
            {**self._last_stage_trace.get(), **decision.to_trace()}
            if decision is not None
            else {}
        )

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """调用 scorer 一次，并用加权 RRF 保守融合原始与重排名次。"""

        scores = self.scorer.score(query, [item.chunk.text for item in candidates])
        if len(scores) != len(candidates):
            raise ValueError("reranker score count does not match candidates")
        rerank_order = sorted(
            zip(candidates, scores),
            key=lambda item: (-item[1], item[0].rank, item[0].chunk_id),
        )
        rerank_ranks = {
            item.chunk_id: rank
            for rank, (item, _) in enumerate(rerank_order, start=1)
        }
        rerank_scores = {item.chunk_id: score for item, score in rerank_order}
        fused = []
        for item in candidates:
            rerank_rank = rerank_ranks[item.chunk_id]
            score = (
                self.config.original_rank_weight
                / (self.config.rerank_rrf_k + item.rank)
                + self.config.rerank_rank_weight
                / (self.config.rerank_rrf_k + rerank_rank)
            )
            fused.append((score, item, rerank_rank, rerank_scores[item.chunk_id]))
        fused.sort(key=lambda value: (-value[0], value[1].rank, value[1].chunk_id))
        return [
            RetrievalResult(
                chunk_id=item.chunk_id,
                score=score,
                rank=rank,
                chunk=item.chunk,
                reason=(
                    f"adaptive_rerank original_rank={item.rank}, "
                    f"rerank_rank={rerank_rank}, fused_score={score:.6f}"
                ),
                details={
                    **item.details,
                    "original_rank": item.rank,
                    "rerank_rank": rerank_rank,
                    "rerank_score": rerank_score,
                    "adaptive_fused_score": score,
                },
            )
            for rank, (score, item, rerank_rank, rerank_score) in enumerate(
                fused, start=1
            )
        ]


def _retrieval_confidence(
    results: list[RetrievalResult],
    top_k: int,
    source_types: set[str] | None,
    strategy: RetrievalStrategy,
) -> float:
    """把结果数量、margin、来源覆盖和双路一致性归一化为 0 到 1。"""

    if not results:
        return 0.0
    quantity = min(1.0, len(results) / max(1, top_k))
    top_score = abs(results[0].score)
    second_score = abs(results[1].score) if len(results) > 1 else 0.0
    margin = min(1.0, max(0.0, top_score - second_score) / max(top_score, 1e-9))
    if source_types:
        covered = {item.chunk.source_type for item in results[:top_k]}
        source_coverage = len(covered & source_types) / len(source_types)
    else:
        source_coverage = 1.0
    agreement = 0.5
    if strategy == "hybrid":
        details = results[0].details
        lexical_rank = details.get("keyword_rank", details.get("bm25_rank"))
        agreement = 1.0 if lexical_rank is not None and details.get("dense_rank") is not None else 0.0
    elif strategy == "graph_hybrid":
        details = results[0].details
        agreement = (
            1.0
            if details.get("graph_rank") is not None
            and details.get("vector_rank") is not None
            else 0.4
        )
    confidence = 0.25 * quantity + 0.20 * margin + 0.25 * source_coverage + 0.30 * agreement
    return round(min(1.0, max(0.0, confidence)), 6)


def _attach_decision(
    results: list[RetrievalResult],
    decision: RetrievalDecision,
) -> list[RetrievalResult]:
    """把 Query 级决策复制到最终结果，便于逐 Case 审计。"""

    return [
        RetrievalResult(
            chunk_id=item.chunk_id,
            score=item.score,
            rank=rank,
            chunk=item.chunk,
            reason=f"{item.reason}; {decision.reason}",
            details={
                **item.details,
                "adaptive_strategy": decision.strategy,
                "adaptive_confidence": decision.confidence,
                "rerank_invoked": int(decision.rerank_invoked),
                "rerank_applied": int(decision.rerank_applied),
            },
        )
        for rank, item in enumerate(results, start=1)
    ]


def _read_retriever_trace(retriever: Retriever) -> dict[str, object]:
    """读取 Graph 等组合 Retriever 暴露的阶段 Trace。"""

    get_last_trace = getattr(retriever, "get_last_trace", None)
    if not callable(get_last_trace):
        return {}
    trace = get_last_trace()
    return dict(trace) if isinstance(trace, dict) else {}
