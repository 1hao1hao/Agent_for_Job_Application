from __future__ import annotations

import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Literal

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.adaptive import AdaptiveRetriever
from intern_rag.retrieval.base import RetrievalResult, Retriever
from intern_rag.retrieval.graph import GraphRetriever

EvidenceGap = Literal[
    "exact_anchor_missing",
    "lexical_channel_missing",
    "semantic_channel_missing",
    "required_sources_missing",
    "multi_source_diversity_missing",
    "graph_path_missing",
]
RescuePath = Literal["bm25", "dense", "graph_path"]


@dataclass(frozen=True)
class EvidenceGapConfig:
    """受控补检索的候选预算、融合参数和版本。"""

    version: str = "evidence-gap-rescue-v1"
    enabled: bool = True
    candidate_k: int = 20
    rrf_k: int = 60
    max_paths: int = 2

    def __post_init__(self) -> None:
        if self.candidate_k <= 0 or self.rrf_k <= 0:
            raise ValueError("evidence gap candidate_k and rrf_k must be positive")
        if not 1 <= self.max_paths <= 3:
            raise ValueError("evidence gap max_paths must be between 1 and 3")


@dataclass(frozen=True)
class EvidenceGapAssessment:
    """只根据 Query 与系统可见结果描述证据缺口，不读取评测标签。"""

    gaps: tuple[EvidenceGap, ...]
    quoted_anchor: str | None
    lexical_covered: bool
    semantic_covered: bool
    graph_path_covered: bool
    observed_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]

    def to_trace(self) -> dict[str, object]:
        """转换为可持久化 Trace。"""

        return asdict(self)


class EvidenceGapGuidedRetriever:
    """在首轮 Adaptive 之后按可观测证据缺口执行至多一次补检索。

    `__call__` 完整保留原 Adaptive 行为，只额外评估词面、语义、来源和图路径
    覆盖。Pipeline 在 Gate 判定 retryable 后调用 `rescue()`；后者按缺口选择
    BM25、Dense 或 linked-entity Graph path，融合去重并保留每路 rank、路径与
    provenance。该类不接收 relevant chunk id，因此线上选择不会泄漏 gold 标签。
    """

    def __init__(
        self,
        base: AdaptiveRetriever,
        routes: Mapping[str, Retriever],
        *,
        graph_retriever: GraphRetriever | None = None,
        config: EvidenceGapConfig | None = None,
    ) -> None:
        missing = {"bm25", "dense"} - set(routes)
        if missing:
            raise ValueError(f"evidence gap retriever missing routes: {sorted(missing)}")
        self.base = base
        self.routes = dict(routes)
        self.graph_retriever = graph_retriever
        self.config = config or EvidenceGapConfig()
        self._last_trace: ContextVar[dict[str, object] | None] = ContextVar(
            f"evidence_gap_trace_{id(self)}", default=None
        )
        self._candidates: ContextVar[list[RetrievalResult] | None] = ContextVar(
            f"gap_candidates_{id(self)}", default=None)

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """执行首轮 Adaptive，并记录系统可观察到的证据缺口。"""

        results = self.base(query, chunks, top_k=top_k, source_types=source_types)
        self._candidates.set(self.base.get_candidates())
        trace = self.base.get_last_trace()
        assessment = assess_evidence_gap(query, results, trace)
        self._last_trace.set({
            **trace,
            "evidence_gap_config_version": self.config.version,
            "evidence_gaps": list(assessment.gaps),
            "evidence_coverage": assessment.to_trace(),
            "rescue_invoked": False,
            "rescue_paths": [],
            "rescue_call_count": 0,
        })
        return results

    def rescue(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        initial_results: list[RetrievalResult],
        top_k: int,
        evidence_requirement: Mapping[str, object] | None = None,
        initial_trace: Mapping[str, object] | None = None,
    ) -> list[RetrievalResult]:
        """按首轮缺口调用互补检索并融合候选，整个请求最多由 Pipeline 调用一次。"""

        trace = dict(initial_trace or self.get_last_trace())
        requirement = dict(evidence_requirement or trace.get("evidence_requirement", {}))
        assessment = assess_evidence_gap(query, initial_results, {
            **trace, "evidence_requirement": requirement
        })
        paths = _choose_rescue_paths(assessment, requirement, self.graph_retriever)
        if not self.config.enabled or not paths:
            self._last_trace.set({
                **trace,
                "evidence_gap_config_version": self.config.version,
                "evidence_gaps": list(assessment.gaps),
                "evidence_coverage": assessment.to_trace(),
                "rescue_invoked": False,
                "rescue_paths": [],
                "rescue_call_count": 0,
                "rescue_reason": "no actionable evidence gap",
            })
            return initial_results[:top_k]

        started_at = perf_counter()
        route_results: dict[str, list[RetrievalResult]] = {"initial": initial_results}
        # 显式标题本身就是跨来源定位键，不能继续受错误 Router 来源限制；只有
        # 纯来源覆盖缺口才定向搜索缺失来源。
        missing_sources = (
            None
            if assessment.quoted_anchor
            else (set(assessment.missing_sources) or None)
        )
        route_traces: dict[str, object] = {}
        for path in paths[: self.config.max_paths]:
            if path == "graph_path" and self.graph_retriever is not None:
                results = self.graph_retriever.retrieve_linked_paths(
                    query,
                    chunks,
                    top_k=self.config.candidate_k,
                    source_types=None,
                    max_paths=self.config.max_paths,
                )
                route_traces[path] = self.graph_retriever.get_last_trace()
            else:
                retriever = self.routes[path]
                results = retriever(
                    query,
                    chunks,
                    top_k=self.config.candidate_k,
                    source_types=missing_sources,
                )
            route_results[path] = results

        fused = fuse_retrieval_routes(
            route_results,
            top_k=self.config.candidate_k,
            rrf_k=self.config.rrf_k,
            route_weights={"initial": 1.0, **{path: 2.0 for path in paths}},
        )
        # 显式标题是强词面约束：候选已经召回后直接优先保留命中项，避免融合权重
        # 又把原来的错误候选推回首位。关系补救同理优先保留完整 linked path。
        if assessment.quoted_anchor:
            ranked = sorted(
                fused,
                key=lambda item: (
                    0 if _result_contains_anchor(item, assessment.quoted_anchor or "") else 1,
                    item.rank,
                    item.chunk_id,
                ),
            )
            ranked = _rerank_positions(ranked)
        elif "graph_path" in paths and not assessment.graph_path_covered:
            ranked = sorted(
                fused,
                key=lambda item: (
                    0 if item.details.get("path_group_id") else 1,
                    item.rank,
                    item.chunk_id,
                ),
            )
            ranked = _rerank_positions(ranked)
        else:
            ranked = self.base._rerank(query, fused) if fused else []
        output = ranked[:top_k]
        self._candidates.set(ranked)
        final_assessment = assess_evidence_gap(query, output, {
            **trace, "evidence_requirement": requirement
        })
        self._last_trace.set({
            **trace,
            "selected_strategy": "evidence_gap_rescue",
            "base_strategy": trace.get("selected_strategy"),
            "evidence_requirement": requirement,
            "evidence_gap_config_version": self.config.version,
            "initial_evidence_gaps": list(assessment.gaps),
            "evidence_gaps": list(final_assessment.gaps),
            "evidence_coverage": final_assessment.to_trace(),
            "rescue_invoked": True,
            "rescue_paths": list(paths[: self.config.max_paths]),
            "rescue_call_count": min(len(paths), self.config.max_paths),
            "rescue_candidate_count": len(fused),
            "rescue_latency_ms": (perf_counter() - started_at) * 1000,
            "rescue_route_traces": route_traces,
            "candidate_chunk_ids": [item.chunk_id for item in fused],
            "candidate_ranks": {item.chunk_id: item.rank for item in fused},
            "rescue_reason": ",".join(assessment.gaps),
        })
        return output

    def get_candidates(self) -> list[RetrievalResult]:
        """返回补检索融合后、top-k 截止前的候选，不改变旧检索行为。"""
        return list(self._candidates.get() or [])

    def retrieve_fixed_multi_route(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """运行 BM25、Dense、linked-path Graph，作为 dev 质量上界和成本参考。"""

        route_results = {
            "bm25": self.routes["bm25"](query, chunks, self.config.candidate_k, None),
            "dense": self.routes["dense"](query, chunks, self.config.candidate_k, None),
        }
        if self.graph_retriever is not None:
            route_results["graph_path"] = self.graph_retriever.retrieve_linked_paths(
                query,
                chunks,
                self.config.candidate_k,
                None,
                self.config.max_paths,
            )
        fused = fuse_retrieval_routes(
            route_results,
            top_k=self.config.candidate_k,
            rrf_k=self.config.rrf_k,
        )
        ranked = self.base._rerank(query, fused) if fused else []
        return ranked[:top_k]

    def get_last_trace(self) -> dict[str, object]:
        """返回首轮或补检索的完整决策 Trace。"""

        return dict(self._last_trace.get() or {})


def assess_evidence_gap(
    query: str,
    results: list[RetrievalResult],
    retrieval_trace: Mapping[str, object],
) -> EvidenceGapAssessment:
    """根据 Query、结果 details 与 EvidenceRequirement 判断可在线观察的缺口。"""

    requirement_raw = retrieval_trace.get("evidence_requirement", {})
    requirement = dict(requirement_raw) if isinstance(requirement_raw, Mapping) else {}
    lexical = any(
        any(key in item.details for key in ("bm25_score", "bm25_rank", "keyword_rank"))
        for item in results
    )
    semantic = any(
        any(key in item.details for key in ("dense_score", "dense_rank", "vector_rank"))
        for item in results
    )
    graph_path = any(
        bool(item.details.get("path_valid")) and bool(item.details.get("graph_edge_ids"))
        for item in results
    )
    observed = tuple(sorted({item.chunk.source_type for item in results}))
    required = tuple(
        sorted(str(item) for item in requirement.get("required_source_types", ()) if str(item))
    )
    missing = tuple(sorted(set(required) - set(observed)))
    anchor = _quoted_anchor(query)
    anchor_covered = anchor is None or any(_result_contains_anchor(item, anchor) for item in results)
    gaps: list[EvidenceGap] = []
    if not anchor_covered:
        gaps.append("exact_anchor_missing")
    if bool(requirement.get("lexical_required")) and not lexical:
        gaps.append("lexical_channel_missing")
    if bool(requirement.get("semantic_required")) and not semantic:
        gaps.append("semantic_channel_missing")
    if missing:
        gaps.append("required_sources_missing")
    if bool(requirement.get("multi_source_required")) and not required and len(observed) < 2:
        gaps.append("multi_source_diversity_missing")
    if bool(requirement.get("graph_required")) and not graph_path:
        gaps.append("graph_path_missing")
    return EvidenceGapAssessment(
        gaps=tuple(dict.fromkeys(gaps)),
        quoted_anchor=anchor,
        lexical_covered=lexical,
        semantic_covered=semantic,
        graph_path_covered=graph_path,
        observed_sources=observed,
        missing_sources=missing,
    )


def fuse_retrieval_routes(
    route_results: Mapping[str, list[RetrievalResult]],
    *,
    top_k: int,
    rrf_k: int = 60,
    route_weights: Mapping[str, float] | None = None,
) -> list[RetrievalResult]:
    """按加权 RRF 合并多路候选，并保存各路 rank 与原始图路径 provenance。"""

    weights = dict(route_weights or {})
    by_id: dict[str, dict[str, object]] = {}
    for route, results in route_results.items():
        for result in results:
            item = by_id.setdefault(result.chunk_id, {
                "chunk": result.chunk,
                "ranks": {},
                "details": {},
                "reasons": [],
            })
            ranks = item["ranks"]
            assert isinstance(ranks, dict)
            ranks[route] = result.rank
            details = item["details"]
            assert isinstance(details, dict)
            details.update(result.details)
            reasons = item["reasons"]
            assert isinstance(reasons, list)
            reasons.append(f"{route}:{result.reason}")
    fused: list[tuple[float, str, dict[str, object]]] = []
    for chunk_id, item in by_id.items():
        ranks = item["ranks"]
        assert isinstance(ranks, dict)
        score = sum(
            float(weights.get(route, 1.0)) / (rrf_k + int(rank))
            for route, rank in ranks.items()
        )
        fused.append((score, chunk_id, item))
    fused.sort(key=lambda value: (-value[0], value[1]))
    output: list[RetrievalResult] = []
    for rank, (score, chunk_id, item) in enumerate(fused[:top_k], 1):
        ranks = item["ranks"]
        details = item["details"]
        reasons = item["reasons"]
        assert isinstance(ranks, dict) and isinstance(details, dict) and isinstance(reasons, list)
        output.append(RetrievalResult(
            chunk_id=chunk_id,
            score=score,
            rank=rank,
            chunk=item["chunk"],  # type: ignore[arg-type]
            reason=f"evidence_gap_rrf routes={sorted(ranks)} score={score:.6f}",
            details={
                **details,
                "rescue_route_ranks": dict(ranks),
                "rescue_provenance": tuple(reasons),
                "rescue_fused_score": score,
            },
        ))
    return output


def _choose_rescue_paths(
    assessment: EvidenceGapAssessment,
    requirement: Mapping[str, object],
    graph_retriever: GraphRetriever | None,
) -> tuple[RescuePath, ...]:
    paths: list[RescuePath] = []
    gaps = set(assessment.gaps)
    need = str(requirement.get("need_type", ""))
    if need == "relation_reasoning" and graph_retriever is not None:
        paths.append("graph_path")
    if gaps & {"exact_anchor_missing", "lexical_channel_missing", "required_sources_missing", "multi_source_diversity_missing"}:
        paths.append("bm25")
    if gaps & {"exact_anchor_missing", "semantic_channel_missing", "required_sources_missing", "multi_source_diversity_missing"}:
        paths.append("dense")
    if not paths:
        if need == "relation_reasoning" and graph_retriever is not None:
            paths.append("graph_path")
        elif need == "exact_fact":
            paths.append("dense")
        else:
            paths.append("bm25")
    return tuple(dict.fromkeys(paths))


def _quoted_anchor(query: str) -> str | None:
    match = re.search(r"[《\"“]([^》\"”]{3,100})[》\"”]", query)
    return match.group(1).strip() if match else None


def _result_contains_anchor(result: RetrievalResult, anchor: str) -> bool:
    needle = _normalize_anchor(anchor)
    haystack = _normalize_anchor(
        f"{result.chunk.title} {result.chunk.text} {result.chunk.source_path}"
    )
    return bool(needle) and needle in haystack


def _normalize_anchor(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def _rerank_positions(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=item.chunk_id,
            score=item.score,
            rank=rank,
            chunk=item.chunk,
            reason=item.reason,
            details=item.details,
        )
        for rank, item in enumerate(results, 1)
    ]
