from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass
import re
from typing import Literal

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.base import RetrievalResult, Retriever
from intern_rag.retrieval.rerank import RerankScorer


RetrievalStrategy = Literal["bm25", "dense", "hybrid", "graph_hybrid"]
RerankPolicy = Literal["never", "always", "low_confidence"]


@dataclass(frozen=True)
class QueryFeatures:
    """Query Analyzer 输出的可解释特征，不依赖评测标签或 LLM。"""

    char_count: int
    source_count: int
    has_exact_term: bool
    has_semantic_rewrite: bool
    is_multi_source: bool
    is_cross_document: bool
    is_unanswerable_route: bool


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

    def to_trace(self) -> dict[str, object]:
        """转换为可写入 AgentTrace/评测工件的普通字典。"""

        return asdict(self)


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
    """用稳定规则识别精确术语、语义改写与多来源 Query。"""

    _semantic_markers = ("换句话", "同义", "通俗", "口语", "改写", "另一种说法")
    _multi_source_markers = ("结合", "对比", "匹配", "综合", "个人经历")
    _cross_document_markers = (
        "哪些项目", "哪个项目", "哪些经历", "哪段经历", "能否证明", "可以证明",
        "是否匹配", "对应起来", "结合岗位", "岗位要求与", "候选人是否",
        "简历经历", "项目记录", "共同体现",
        "跨来源", "知识关系", "跳联系", "关系路径", "互补证据",
    )
    _technical_terms = (
        "混合检索", "引用校验", "请求追踪", "意图路由", "上下文预算", "失败回归",
        "bm25", "dense", "rrf", "agent", "rag", "trace", "citation",
    )
    _latin_term_pattern = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]+")

    def analyze(
        self,
        query: str,
        source_types: set[str] | None,
    ) -> QueryFeatures:
        """从 Query 和 Router source filter 提取策略选择所需特征。"""

        normalized = query.strip().lower()
        sources = source_types or set()
        return QueryFeatures(
            char_count=len(normalized),
            source_count=len(sources),
            has_exact_term=(
                any(term in normalized for term in self._technical_terms)
                or bool(self._latin_term_pattern.search(query))
            ),
            has_semantic_rewrite=any(
                marker in normalized for marker in self._semantic_markers
            ),
            is_multi_source=(
                len(sources) >= 2
                or any(marker in normalized for marker in self._multi_source_markers)
            ),
            is_cross_document=(
                any(marker in normalized for marker in self._cross_document_markers)
                or (
                    "岗位" in normalized
                    and any(marker in normalized for marker in ("项目", "经历", "简历"))
                )
            ),
            is_unanswerable_route=source_types is not None and not source_types,
        )

    def choose_strategy(
        self,
        features: QueryFeatures,
        *,
        graph_enabled: bool = False,
    ) -> tuple[RetrievalStrategy, str]:
        """根据特征选择检索策略；图未配置时保持 P1-D2 行为。"""

        if features.is_unanswerable_route:
            return "bm25", "router returned no searchable sources"
        if graph_enabled and features.is_cross_document:
            return "graph_hybrid", "cross-document relation query uses graph and vector evidence"
        if features.is_multi_source:
            return "hybrid", "multi-source query needs lexical and semantic recall"
        if features.has_semantic_rewrite and not features.has_exact_term:
            return "dense", "semantic rewrite has weak exact-term overlap"
        if features.char_count <= 8 and not features.has_exact_term:
            return "bm25", "short literal query uses low-cost lexical retrieval"
        if features.has_exact_term:
            return "hybrid", "exact term benefits from lexical recall with dense backup"
        return "hybrid", "ambiguous query keeps lexical and semantic recall paths"


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
        strategy, strategy_reason = self.analyzer.choose_strategy(
            features, graph_enabled=self.graph_retriever is not None
        )
        if self.config.force_strategy is not None:
            strategy = self.config.force_strategy
            strategy_reason = f"forced {strategy} for controlled ablation"
        if top_k <= 0 or not query.strip() or features.is_unanswerable_route:
            decision = RetrievalDecision(
                strategy=strategy,
                confidence=1.0 if features.is_unanswerable_route else 0.0,
                rerank_invoked=False,
                rerank_applied=False,
                candidate_count=0,
                reason=strategy_reason,
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
