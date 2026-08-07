from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from intern_rag.agent.answer import Citation


RagStatus = Literal["answered", "insufficient_evidence", "error"]
RetrieverName = Literal["keyword", "dense", "hybrid"]
RouterName = Literal["rule", "semantic", "hybrid"]


@dataclass(frozen=True)
class RagRequest:
    """一次 EvalRAG 查询请求。

    query 是用户问题；
    request_id 用于关联后续 response 和 trace;
    top_k 与retriever 保存本次请求的检索配置。
    当前任务只定义契约，不执行检索。
    """

    query: str
    request_id: str = field(default_factory=lambda: str(uuid4()))
    top_k: int = 5
    retriever: RetrieverName = "keyword"

    def __post_init__(self) -> None:
        """拒绝无法进入后续流程的空 query 和非正数 top_k。"""

        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")


@dataclass(frozen=True)
class RagResponse:
    """一次 EvalRAG 查询的统一响应契约。

    citations 直接复用已有 Citation，不重复定义引用结构。
    Pipeline 后续会根据 answered、insufficient_evidence 或 error 状态构造该对象。
    """

    request_id: str
    trace_id: str
    answer: str
    citations: list[Citation]
    routed_sources: list[str]
    status: RagStatus
    latency_ms: float
    error_type: str | None = None


@dataclass(frozen=True)
class ContextItem:
    """模型上下文中的一条完整证据。

    该结构保留 citation 校验需要的 chunk id，也保留来源、标题、原文、
    检索排名和分数。Context Builder 不修改这些证据信息。
    """

    chunk_id: str
    source_type: str
    source_path: str
    title: str
    text: str
    rank: int
    score: float


@dataclass(frozen=True)
class BuiltContext:
    """Context Builder 的输出。

    text 可直接交给后续 Generator；
    items 保留结构化证据；
    used/skipped
    chunk ids 说明预算选择结果，方便后续 Trace 与 Citation Validator 使用。
    """

    query: str
    text: str
    items: list[ContextItem]
    used_chunk_ids: list[str]
    skipped_chunk_ids: list[str]
    char_count: int
    max_chars: int

    @property
    def is_truncated(self) -> bool:
        """只要有 chunk 因预算未使用，就表示上下文发生了预算截断。"""

        return bool(self.skipped_chunk_ids)
