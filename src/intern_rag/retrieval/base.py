from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from intern_rag.ingestion import Chunk


RetrievalDetail = str | int | float | None


@dataclass(frozen=True)
class RetrievalResult:
    """统一的检索结果契约。

    `score` 只在同一种 Retriever 内可比较；Hybrid 会把各路 rank 放进
    `details`，避免把 Keyword 与 Dense 的原始分数直接相加。
    """

    chunk_id: str
    score: float
    rank: int
    chunk: Chunk
    reason: str | None = None
    details: dict[str, RetrievalDetail] = field(default_factory=dict)


class Retriever(Protocol):
    """Keyword、Dense 与 Hybrid 共同遵守的最小检索接口。"""

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """返回经过来源过滤且按相关性排序的结果。"""
