"""有界的证据任务调度：规划 -> 多路检索 -> 组装 -> 验证 -> 定向补救一次。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from time import perf_counter
from typing import Callable
import re

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.adaptive import AdaptiveRetriever
from intern_rag.retrieval.base import RetrievalResult
from intern_rag.retrieval.evidence_plan import (
    EvidenceAssembler, EvidenceBundle, EvidencePlan, EvidencePlanner, EvidenceScope,
    EvidenceVerification, EvidenceVerifier,
)


@dataclass(frozen=True)
class OrchestratorConfig:
    """所有槽共享调用预算，不按槽无限叠加成本。"""

    version: str = "evidence-oriented-v1-candidate"
    max_calls: int = 6
    max_workers: int = 2
    candidate_k: int = 20
    max_paths: int = 2
    rrf_k: int = 60
    evidence_token_budget: int = 1200
    rerank: bool = True

    def __post_init__(self) -> None:
        if min(self.max_calls, self.max_workers, self.candidate_k, self.max_paths,
               self.rrf_k, self.evidence_token_budget) <= 0:
            raise ValueError("orchestrator budgets must be positive")


@dataclass
class _RequestState:
    plan: EvidencePlan
    cache: dict[tuple[str, str, tuple[str, ...] | None], list[RetrievalResult]] = field(default_factory=dict)
    history: list[dict[str, object]] = field(default_factory=list)
    candidates: list[RetrievalResult] = field(default_factory=list)
    bundle: EvidenceBundle | None = None
    verification: EvidenceVerification | None = None
    rescued: bool = False
    reranked: bool = False
    errors: list[str] = field(default_factory=list)
    initial_call_count: int = 0


class EvidenceOrientedRetriever:
    """复用现有索引/模型的 Retriever，同时提供 Pipeline 的证据能力接口。"""

    def __init__(self, base: AdaptiveRetriever, *, graph_retriever=None,
                 planner: EvidencePlanner | None = None,
                 config: OrchestratorConfig | None = None,
                 count_tokens: Callable[[str], int] | None = None) -> None:
        self.base = base
        self.graph_retriever = graph_retriever
        self.planner = planner or EvidencePlanner(base.analyzer)
        self.config = config or OrchestratorConfig()
        self.assembler = EvidenceAssembler()
        self.verifier = EvidenceVerifier()
        self.count_tokens = count_tokens or (
            lambda text: len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+|[^\s]", text))
        )
        self._state: ContextVar[_RequestState | None] = ContextVar(
            f"evidence_oriented_{id(self)}", default=None)

    def __call__(self, query: str, chunks: list[Chunk], top_k: int = 5,
                 source_types: set[str] | None = None) -> list[RetrievalResult]:
        """兼容原 Retriever 契约；soft filter 不属于用户授权。"""
        return self.retrieve_evidence(query, chunks, top_k, source_types, EvidenceScope())

    def retrieve_evidence(self, query: str, chunks: list[Chunk], top_k: int = 5,
                          source_types: set[str] | None = None,
                          scope: EvidenceScope | None = None) -> list[RetrievalResult]:
        """规划槽后有限并行检索，融合和整组装箱，再逐槽验证。

        所有底层检索先拿授权子库；输出再检查 ACL。调用异常记录且继续可用通道，
        模型/图不完备不会被包装成完整证据；调用记录随 Trace 返回。
        """
        state = _RequestState(self.planner.plan(query, source_types, scope))
        self._state.set(state)
        if top_k <= 0 or not query.strip():
            return []
        strategy, _ = self.base.analyzer.choose_strategy(
            state.plan.requirement, graph_enabled=self.graph_retriever is not None)
        tasks = []
        if strategy == "graph_hybrid":
            tasks.extend([("graph", query, None), ("hybrid", query, source_types)])
        elif strategy == "bm25":
            # 精确槽仍需要语义佐证，BM25 分数本身不能证明事实。
            tasks.extend([("bm25", query, source_types), ("dense", query, source_types)])
        else:
            tasks.append((strategy, query, source_types))
        for slot in state.plan.slots[1:]:
            if slot.kind == "source":
                tasks.append(("hybrid", slot.subquery, set(slot.required_sources)))
            elif slot.slot_id.startswith("decomposed:"):
                tasks.append(("hybrid", slot.subquery, None))
        self._execute(state, tasks, chunks)
        state.initial_call_count = len(state.cache)
        return self._finish(state, top_k, exhausted=False)

    def rescue(self, query: str, chunks: list[Chunk], *, initial_results,
               top_k: int = 5, evidence_requirement=None, initial_trace=None) -> list[RetrievalResult]:
        """只对未满足槽补一次；去除路由偏好，绝不扩大服务端授权。"""
        state = self._state.get()
        if state is None or state.plan.query != query:
            raise ValueError("rescue requires this request's evidence plan")
        if state.rescued:
            return list(state.bundle.results) if state.bundle else []
        state.rescued = True
        missing = {v.slot_id for v in state.verification.verdicts
                   if v.status in {"missing", "unknown"}} if state.verification else set()
        tasks = []
        for slot in state.plan.slots:
            if slot.slot_id not in missing:
                continue
            source = set(slot.required_sources) or None
            if slot.kind == "relation" and self.graph_retriever is not None:
                tasks.append(("graph", slot.subquery, None))
                tasks.append(("dense", slot.subquery, None))
            elif slot.anchors:
                tasks.extend(("bm25", anchor, source) for anchor in slot.anchors)
                tasks.append(("dense", slot.subquery, source))
            else:
                tasks.append(("hybrid", slot.subquery, source))
        self._execute(state, tasks, chunks)
        return self._finish(state, top_k, exhausted=True)

    def _execute(self, state: _RequestState, tasks, chunks: list[Chunk]) -> None:
        """按请求共享预算去重任务，有限并行执行并保存各通道原始召回历史。"""
        authorized = [c for c in chunks if state.plan.scope.permits(c)]
        jobs = []
        for strategy, query, sources in tasks:
            key = (strategy, query, tuple(sorted(sources)) if sources is not None else None)
            if key in state.cache or key in jobs:
                continue
            if len(state.cache) + len(jobs) >= self.config.max_calls:
                break
            jobs.append(key)
        def run(key):
            strategy, query, sources = key
            started = perf_counter()
            try:
                if strategy == "graph":
                    rows = self.graph_retriever.retrieve_linked_paths(
                        query, authorized, self.config.candidate_k,
                        set(sources) if sources is not None else None,
                        max_paths=self.config.max_paths)
                    # 路径中的边不能引用不在授权子库的出处。
                    allowed_ids = {c.id for c in authorized}
                    edges = {e.edge_id: e for e in self.graph_retriever.graph.edges}
                    rows = [r for r in rows if all(
                        edge_id in edges and set(edges[edge_id].chunk_ids) <= allowed_ids
                        for edge_id in str(r.details.get("graph_edge_ids", "")).split("|")
                        if edge_id)]
                else:
                    rows = self.base.retrievers[strategy](
                        query, authorized, self.config.candidate_k,
                        set(sources) if sources is not None else None)
                rows = [replace(r, details={**r.details, "retrieval_channel": strategy,
                                          "semantic_channel": int(strategy in {"dense", "hybrid"})})
                        for r in rows if state.plan.scope.permits(r.chunk)]
                error = None
            except Exception as exc:
                rows, error = [], type(exc).__name__
            return rows, {"strategy": strategy, "query": query, "source_filter": sources,
                          "chunk_ids": [r.chunk_id for r in rows], "error_type": error,
                          "latency_ms": (perf_counter() - started) * 1000}
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            for key, (rows, history) in zip(jobs, pool.map(run, jobs)):
                state.cache[key] = rows
                state.history.append(history)
                if history["error_type"]:
                    state.errors.append(str(history["error_type"]))
        scores: dict[str, float] = {}
        merged: dict[str, RetrievalResult] = {}
        for key, rows in state.cache.items():
            for r in rows:
                scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1 / (self.config.rrf_k + r.rank)
                previous = merged.get(r.chunk_id)
                details = {**(previous.details if previous else {}), **r.details}
                if previous and previous.details.get("path_group_id"):
                    details.update({k: v for k, v in previous.details.items() if k.startswith("path_") or k.startswith("graph_")})
                details["semantic_channel"] = max(int(details.get("semantic_channel", 0)),
                                                    int(previous.details.get("semantic_channel", 0)) if previous else 0)
                merged[r.chunk_id] = replace(r, details=details)
        ordered = sorted(merged, key=lambda cid: (-scores[cid], cid))
        state.candidates = [replace(merged[cid], score=scores[cid], rank=i)
                            for i, cid in enumerate(ordered, 1)]

    def _finish(self, state: _RequestState, top_k: int, *, exhausted: bool) -> list[RetrievalResult]:
        """最多一次统一重排，优先完整证据组；top-k 不截断路径组。"""
        candidates = state.candidates
        if self.config.rerank and not state.reranked and candidates and not state.plan.requirement.graph_required:
            state.reranked = True
            candidates = self.base._rerank(state.plan.query, candidates[:self.config.candidate_k])
            state.candidates = candidates
        # 显式标题与关系路径是组装约束，而不是 gold 优先排序。
        anchors = tuple(a for slot in state.plan.slots for a in slot.anchors)
        candidates = sorted(candidates, key=lambda r: (
            0 if (state.plan.requirement.graph_required and r.details.get("path_group_id"))
                 or any(a.lower() in (r.chunk.title or "").lower() for a in anchors) else 1,
            r.rank, r.chunk_id))
        candidates = [replace(r, rank=i) for i, r in enumerate(candidates, 1)]
        # 先选择 top-k 的整组闭包，多出来的组成员受统一 Token 预算约束。
        selected_groups = {str(r.details.get("path_group_id") or r.chunk_id) for r in candidates[:top_k]}
        selected = [r for r in candidates if str(r.details.get("path_group_id") or r.chunk_id) in selected_groups]
        state.bundle = self.assembler.assemble(state.plan, selected,
                                               self.config.evidence_token_budget, self.count_tokens)
        state.verification = self.verifier.verify(state.plan, state.bundle, exhausted)
        return list(state.bundle.results)

    def verify_context(self, used_chunk_ids: list[str]) -> EvidenceVerification:
        """ContextEngine 裁剪后重新检查绑定和完整组，禁止半路径进入 Generator。"""
        state = self._state.get()
        if state is None or state.bundle is None:
            raise ValueError("context verification requires assembled evidence")
        retained = set(used_chunk_ids)
        bundle = replace(state.bundle,
                         results=tuple(r for r in state.bundle.results if r.chunk_id in retained),
                         groups=tuple(replace(g, complete=g.complete and set(g.chunk_ids) <= retained)
                                      for g in state.bundle.groups))
        return self.verifier.verify(state.plan, bundle, exhausted=True)

    def get_last_trace(self) -> dict[str, object]:
        """返回完整计划、调度历史、证据组与逐槽判定，供 Trace/Replay 使用。"""
        state = self._state.get()
        if state is None:
            return {}
        requirement = state.plan.requirement
        return {"config_version": self.config.version, "selected_strategy": "evidence_oriented",
                "evidence_requirement": asdict(requirement), "evidence_plan": state.plan.to_trace(),
                "evidence_bundle": state.bundle.to_trace() if state.bundle else {},
                "slot_verification": asdict(state.verification) if state.verification else {},
                "retrieval_history": state.history, "retrieval_call_count": len(state.cache),
                "rescue_invoked": state.rescued,
                "rescue_call_count": len(state.cache) - state.initial_call_count if state.rescued else 0,
                "rerank_invoked": state.reranked, "planning_fallback": state.plan.fallback_reason,
                "retrieval_errors": state.errors,
                "budget": asdict(self.config)}
