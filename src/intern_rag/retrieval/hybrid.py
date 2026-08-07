from __future__ import annotations

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.base import RetrievalResult, Retriever


class HybridRetriever:
    """使用 Reciprocal Rank Fusion 融合 Keyword 与 Dense 排名。"""

    def __init__(
        self,
        keyword_retriever: Retriever,
        dense_retriever: Retriever,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if rrf_k <= 0 or candidate_multiplier <= 0:
            raise ValueError("rrf_k and candidate_multiplier must be positive")
        self.keyword_retriever = keyword_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """分别取两路候选，按 rank 融合并对重复 Chunk 去重。"""

        if top_k <= 0:
            return []
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        keyword_results = self.keyword_retriever(
            query, chunks, candidate_k, source_types
        )
        dense_results = self.dense_retriever(
            query, chunks, candidate_k, source_types
        )
        by_id: dict[str, dict[str, object]] = {}
        for route_name, results in (
            ("keyword", keyword_results),
            ("dense", dense_results),
        ):
            for result in results:
                item = by_id.setdefault(
                    result.chunk_id,
                    {"chunk": result.chunk, "keyword_rank": None, "dense_rank": None},
                )
                item[f"{route_name}_rank"] = result.rank

        fused: list[tuple[float, str, dict[str, object]]] = []
        for chunk_id, item in by_id.items():
            keyword_rank = item["keyword_rank"]
            dense_rank = item["dense_rank"]
            score = sum(
                1.0 / (self.rrf_k + int(rank))
                for rank in (keyword_rank, dense_rank)
                if rank is not None
            )
            fused.append((score, chunk_id, item))

        fused.sort(key=lambda item: (-item[0], item[1]))
        results: list[RetrievalResult] = []
        for rank, (score, chunk_id, item) in enumerate(fused[:top_k], start=1):
            keyword_rank = item["keyword_rank"]
            dense_rank = item["dense_rank"]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    rank=rank,
                    chunk=item["chunk"],  # type: ignore[arg-type]
                    reason=(
                        f"rrf keyword_rank={keyword_rank}, "
                        f"dense_rank={dense_rank}, fused_score={score:.6f}"
                    ),
                    details={
                        "keyword_rank": keyword_rank,  # type: ignore[dict-item]
                        "dense_rank": dense_rank,  # type: ignore[dict-item]
                        "fused_score": score,
                    },
                )
            )
        return results
