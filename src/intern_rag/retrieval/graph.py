from __future__ import annotations

from collections import deque
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from time import perf_counter

from intern_rag.graph import KnowledgeGraph, QueryDecomposer
from intern_rag.graph.schemas import GraphEdge
from intern_rag.ingestion import Chunk
from intern_rag.retrieval.base import RetrievalResult, Retriever


@dataclass(frozen=True)
class GraphRetrievalTrace:
    """一次图检索的分解、遍历边界和结果摘要。"""

    decomposition: dict[str, object]
    linked_node_ids: tuple[str, ...]
    visited_node_count: int
    path_count: int
    fallback_reason: str | None
    graph_version: str

    def to_dict(self) -> dict[str, object]:
        """转换为普通字典供 Trace 与实验工件保存。"""

        return asdict(self)


class GraphRetriever:
    """从 Query 实体出发执行有边界遍历并返回原始 Chunk 证据。

    输入仍是 query、完整 Chunk 列表、top-k 和来源过滤。先做确定性问题分解与
    实体链接，再从命中节点进行最多 `max_hops` 跳 BFS；遍历受节点数和超时限制。
    节点及边上的 `chunk_ids` 被转换为统一 `RetrievalResult`，`graph_path` 保存
    可审计路径。没有实体或达到受控边界时返回空结果，由上层 Graph + Vector
    Retriever 回退到原始向量结果。
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        *,
        decomposer: QueryDecomposer | None = None,
        max_hops: int = 2,
        max_nodes: int = 80,
        timeout_ms: float = 50.0,
    ) -> None:
        if max_hops not in {1, 2, 3}:
            raise ValueError("max_hops must be 1, 2 or 3")
        if max_nodes <= 0 or timeout_ms <= 0:
            raise ValueError("max_nodes and timeout_ms must be positive")
        self.graph = graph
        self.decomposer = decomposer or QueryDecomposer()
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.timeout_ms = timeout_ms
        self._last_trace: ContextVar[GraphRetrievalTrace | None] = ContextVar(
            f"graph_retrieval_trace_{id(self)}", default=None
        )

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """执行问题分解、实体链接、有界遍历和 Chunk 排序。"""

        decomposition = self.decomposer.decompose(query, self.graph)
        if top_k <= 0 or not query.strip() or not decomposition.entity_node_ids:
            self._set_trace(
                decomposition.to_trace(),
                decomposition.entity_node_ids,
                0,
                0,
                "invalid query or no linked graph entity",
            )
            return []

        chunk_by_id = {
            chunk.id: chunk
            for chunk in chunks
            if source_types is None or chunk.source_type in source_types
        }
        node_by_id = self.graph.node_by_id()
        adjacency = _build_adjacency(self.graph.edges)
        queue = deque(
            (node_id, 0, node_by_id[node_id].name, ())
            for node_id in decomposition.entity_node_ids
            if node_id in node_by_id
        )
        visited: set[str] = set()
        candidates: dict[str, tuple[float, int, str, tuple[str, ...]]] = {}
        started_at = perf_counter()
        timed_out = False

        while queue and len(visited) < self.max_nodes:
            if (perf_counter() - started_at) * 1000 >= self.timeout_ms:
                timed_out = True
                break
            node_id, hops, path, edge_ids = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = node_by_id[node_id]
            _collect_chunk_candidates(
                candidates,
                node.chunk_ids,
                chunk_by_id,
                1.0 / (1 + hops),
                hops,
                path,
                edge_ids,
            )
            if hops >= self.max_hops:
                continue
            for edge, next_node_id in adjacency.get(node_id, []):
                next_node = node_by_id[next_node_id]
                next_path = f"{path} -[{edge.edge_type}]-> {next_node.name}"
                next_edge_ids = (*edge_ids, edge.edge_id)
                relation_bonus = (
                    0.2 if edge.edge_type in decomposition.relations else 0.0
                )
                _collect_chunk_candidates(
                    candidates,
                    edge.chunk_ids,
                    chunk_by_id,
                    0.8 / (1 + hops) + relation_bonus,
                    hops + 1,
                    next_path,
                    next_edge_ids,
                )
                if next_node_id not in visited:
                    queue.append(
                        (next_node_id, hops + 1, next_path, next_edge_ids)
                    )

        ordered = sorted(
            candidates.items(),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
        )
        results = [
            RetrievalResult(
                chunk_id=chunk_id,
                score=value[0],
                rank=rank,
                chunk=chunk_by_id[chunk_id],
                reason=(
                    f"graph hops={value[1]}, score={value[0]:.6f}, "
                    f"path={value[2]}"
                ),
                details={
                    "graph_score": value[0],
                    "graph_hops": value[1],
                    "graph_path": value[2],
                    "graph_edge_ids": "|".join(value[3]),
                    "path_valid": int(self.validate_edge_path(value[3], value[1])),
                },
            )
            for rank, (chunk_id, value) in enumerate(ordered[:top_k], start=1)
        ]
        fallback_reason = "graph traversal timeout" if timed_out else None
        self._set_trace(
            decomposition.to_trace(),
            decomposition.entity_node_ids,
            len(visited),
            len(results),
            fallback_reason,
        )
        return results

    def validate_edge_path(self, edge_ids: tuple[str, ...], hops: int) -> bool:
        """检查路径引用的边存在，且边数与遍历 hop 一致。"""

        known_edge_ids = {edge.edge_id for edge in self.graph.edges}
        return len(edge_ids) == hops and all(
            edge_id in known_edge_ids for edge_id in edge_ids
        )

    def get_last_trace(self) -> dict[str, object]:
        """返回最近一次图检索 Trace。"""

        trace = self._last_trace.get()
        return trace.to_dict() if trace is not None else {}

    def _set_trace(
        self,
        decomposition: dict[str, object],
        linked_node_ids: tuple[str, ...],
        visited_node_count: int,
        path_count: int,
        fallback_reason: str | None,
    ) -> None:
        self._last_trace.set(
            GraphRetrievalTrace(
                decomposition=decomposition,
                linked_node_ids=linked_node_ids,
                visited_node_count=visited_node_count,
                path_count=path_count,
                fallback_reason=fallback_reason,
                graph_version=self.graph.version,
            )
        )


class GraphVectorRetriever:
    """使用 RRF 融合 Graph 与 Vector 排名，图失败时保留 Vector 结果。"""

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        vector_retriever: Retriever,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if rrf_k <= 0 or candidate_multiplier <= 0:
            raise ValueError("rrf_k and candidate_multiplier must be positive")
        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self._last_trace: ContextVar[dict[str, object]] = ContextVar(
            f"graph_vector_trace_{id(self)}", default={}
        )

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """召回两路候选、按 rank 融合，并保留图路径与各路名次。"""

        if top_k <= 0:
            return []
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        vector_results = self.vector_retriever(
            query, chunks, candidate_k, source_types
        )
        graph_results = self.graph_retriever(
            query, chunks, candidate_k, source_types
        )
        graph_trace = self.graph_retriever.get_last_trace()
        if not graph_results:
            self._last_trace.set(
                {
                    **graph_trace,
                    "graph_invoked": True,
                    "vector_fallback": True,
                    "reason": "graph returned no candidates; keep original vector ranking",
                }
            )
            return _rerank_results(vector_results[:top_k], "graph vector fallback")

        by_id: dict[str, dict[str, object]] = {}
        for route_name, results in (
            ("vector", vector_results),
            ("graph", graph_results),
        ):
            for result in results:
                item = by_id.setdefault(
                    result.chunk_id,
                    {
                        "chunk": result.chunk,
                        "vector_rank": None,
                        "graph_rank": None,
                        "graph_path": None,
                        "graph_edge_ids": None,
                        "path_valid": None,
                    },
                )
                item[f"{route_name}_rank"] = result.rank
                if route_name == "graph":
                    item["graph_path"] = result.details.get("graph_path")
                    item["graph_edge_ids"] = result.details.get("graph_edge_ids")
                    item["path_valid"] = result.details.get("path_valid")

        fused = []
        for chunk_id, item in by_id.items():
            ranks = (item["vector_rank"], item["graph_rank"])
            score = sum(
                1.0 / (self.rrf_k + int(rank))
                for rank in ranks
                if rank is not None
            )
            fused.append((score, chunk_id, item))
        fused.sort(key=lambda item: (-item[0], item[1]))
        results = [
            RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                rank=rank,
                chunk=item["chunk"],  # type: ignore[arg-type]
                reason=(
                    f"graph_vector_rrf vector_rank={item['vector_rank']}, "
                    f"graph_rank={item['graph_rank']}, fused_score={score:.6f}"
                ),
                details={
                    "vector_rank": item["vector_rank"],  # type: ignore[dict-item]
                    "graph_rank": item["graph_rank"],  # type: ignore[dict-item]
                    "graph_path": item["graph_path"],  # type: ignore[dict-item]
                    "graph_edge_ids": item["graph_edge_ids"],  # type: ignore[dict-item]
                    "path_valid": item["path_valid"],  # type: ignore[dict-item]
                    "fused_score": score,
                },
            )
            for rank, (score, chunk_id, item) in enumerate(fused[:top_k], start=1)
        ]
        self._last_trace.set(
            {
                **graph_trace,
                "graph_invoked": True,
                "vector_fallback": False,
                "vector_candidate_count": len(vector_results),
                "graph_candidate_count": len(graph_results),
            }
        )
        return results

    def get_last_trace(self) -> dict[str, object]:
        """返回分解、图遍历和融合摘要。"""

        return dict(self._last_trace.get())


def _build_adjacency(
    edges: tuple[GraphEdge, ...],
) -> dict[str, list[tuple[GraphEdge, str]]]:
    adjacency: dict[str, list[tuple[GraphEdge, str]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_node_id, []).append(
            (edge, edge.target_node_id)
        )
        adjacency.setdefault(edge.target_node_id, []).append(
            (edge, edge.source_node_id)
        )
    for values in adjacency.values():
        values.sort(key=lambda item: (item[0].edge_type, item[1]))
    return adjacency


def _collect_chunk_candidates(
    candidates: dict[str, tuple[float, int, str, tuple[str, ...]]],
    chunk_ids: tuple[str, ...],
    chunk_by_id: dict[str, Chunk],
    score: float,
    hops: int,
    path: str,
    edge_ids: tuple[str, ...],
) -> None:
    for chunk_id in chunk_ids:
        if chunk_id not in chunk_by_id:
            continue
        current = candidates.get(chunk_id)
        value = (score, hops, path, edge_ids)
        if current is None or (-score, hops, path) < (-current[0], current[1], current[2]):
            candidates[chunk_id] = value


def _rerank_results(
    results: list[RetrievalResult], reason_suffix: str
) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=item.chunk_id,
            score=item.score,
            rank=rank,
            chunk=item.chunk,
            reason=f"{item.reason}; {reason_suffix}",
            details={**item.details, "graph_rank": None, "vector_rank": item.rank},
        )
        for rank, item in enumerate(results, start=1)
    ]
