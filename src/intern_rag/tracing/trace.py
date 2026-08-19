from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from intern_rag.retrieval import RetrievalResult
from intern_rag.routing import RouteDecision

# 枚举错误类型
ErrorType = Literal[
    "none",
    "ingestion_error",
    "retrieval_miss",
    "router_error",
    "retriever_error",
    "rerank_error",
    "tool_error",
    "llm_error",
    "llm_format_error",
    "llm_timeout",
    "citation_error",
    "citation_invalid",
    "hallucination",
    "unknown_error",
]
TraceDict = dict[str, object]


@dataclass(frozen=True)
class AgentTrace:
    """一次单轮请求的可观测记录。

    Trace 用来回答“系统为什么这么做”：用户问了什么、router 判断了什么
    intent、检索到了哪些 chunk、各阶段耗时多少、是否出现错误。
    """

    request_id: str
    query: str
    intent: str
    routed_sources: list[str]
    retrieved_chunks: list[TraceDict]
    latency_ms: dict[str, float]
    error_type: ErrorType = "none"
    created_at: str = ""
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    rerank_results: list[TraceDict] = field(default_factory=list)
    tool_calls: list[TraceDict] = field(default_factory=list)
    citations: list[TraceDict] = field(default_factory=list)
    answer: str = ""
    routing: TraceDict = field(default_factory=dict)
    retrieval: TraceDict = field(default_factory=dict)
    context: TraceDict = field(default_factory=dict)
    evidence: TraceDict = field(default_factory=dict)
    generation: TraceDict = field(default_factory=dict)
    validation: TraceDict = field(default_factory=dict)
    model_config: TraceDict = field(default_factory=dict)
    prompt_version: str = ""
    response_status: str = ""
    error_message: str = ""
    attempts: list[TraceDict] = field(default_factory=list)
    token_usage: TraceDict = field(default_factory=dict)

    def to_dict(self) -> TraceDict:
        """转换成可写入 JSONL 的普通字典。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, trace_data: TraceDict) -> "AgentTrace":
        """从 JSONL 读取出的字典恢复 AgentTrace。"""

        return cls(
            request_id=str(trace_data["request_id"]),
            query=str(trace_data["query"]),
            intent=str(trace_data["intent"]),
            routed_sources=list(trace_data["routed_sources"]),  # type: ignore[arg-type]
            retrieved_chunks=list(trace_data["retrieved_chunks"]),  # type: ignore[arg-type]
            latency_ms=dict(trace_data["latency_ms"]),  # type: ignore[arg-type]
            error_type=_normalize_error_type(str(trace_data.get("error_type", "none"))),
            created_at=str(trace_data.get("created_at", "")),
            trace_id=str(trace_data.get("trace_id", "")),
            rerank_results=list(trace_data.get("rerank_results", [])),  # type: ignore[arg-type]
            tool_calls=list(trace_data.get("tool_calls", [])),  # type: ignore[arg-type]
            citations=list(trace_data.get("citations", [])),  # type: ignore[arg-type]
            answer=str(trace_data.get("answer", "")),
            routing=dict(trace_data.get("routing", {})),  # type: ignore[arg-type]
            retrieval=dict(trace_data.get("retrieval", {})),  # type: ignore[arg-type]
            context=dict(trace_data.get("context", {})),  # type: ignore[arg-type]
            evidence=dict(trace_data.get("evidence", {})),  # type: ignore[arg-type]
            generation=dict(trace_data.get("generation", {})),  # type: ignore[arg-type]
            validation=dict(trace_data.get("validation", {})),  # type: ignore[arg-type]
            model_config=dict(trace_data.get("model_config", {})),  # type: ignore[arg-type]
            prompt_version=str(trace_data.get("prompt_version", "")),
            response_status=str(trace_data.get("response_status", "")),
            error_message=str(trace_data.get("error_message", "")),
            attempts=list(trace_data.get("attempts", [])),  # type: ignore[arg-type]
            token_usage=dict(trace_data.get("token_usage", {})),  # type: ignore[arg-type]
        )


def build_agent_trace(
    query: str,
    route_decision: RouteDecision,
    retrieved_results: list[RetrievalResult],
    latency_ms: dict[str, float],
    error_type: ErrorType = "none",
    request_id: str | None = None,
    created_at: str | None = None,
    trace_id: str | None = None,
    citations: list[TraceDict] | None = None,
    answer: str = "",
    routing: TraceDict | None = None,
    retrieval: TraceDict | None = None,
    context: TraceDict | None = None,
    evidence: TraceDict | None = None,
    generation: TraceDict | None = None,
    validation: TraceDict | None = None,
    model_config: TraceDict | None = None,
    prompt_version: str = "",
    response_status: str = "",
    error_message: str = "",
    attempts: list[TraceDict] | None = None,
    token_usage: TraceDict | None = None,
) -> AgentTrace:
    """根据单轮 routing 和 retrieval 结果构造 AgentTrace。

    这个函数只负责组装 trace，不负责真正执行 router 或 retrieval。
    """

    return AgentTrace(
        request_id=request_id or str(uuid4()),
        query=query,
        intent=route_decision.intent,
        routed_sources=route_decision.routed_sources,
        retrieved_chunks=[
            retrieval_result_to_trace(result) for result in retrieved_results
        ],
        latency_ms=latency_ms,
        error_type=error_type,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        trace_id=trace_id or str(uuid4()),
        citations=citations or [],
        answer=answer,
        routing=routing or {},
        retrieval=retrieval or {},
        context=context or {},
        evidence=evidence or {},
        generation=generation or {},
        validation=validation or {},
        model_config=model_config or {},
        prompt_version=prompt_version,
        response_status=response_status,
        error_message=error_message,
        attempts=attempts or [],
        token_usage=token_usage or {},
    )


def retrieval_result_to_trace(result: RetrievalResult) -> TraceDict:
    """把 RetrievalResult 转成 trace 中的 retrieved chunk 记录。"""

    return {
        "chunk_id": result.chunk_id,
        "rank": result.rank,
        "score": result.score,
        "reason": result.reason,
        "source_type": result.chunk.source_type,
        "source_path": result.chunk.source_path,
        "title": result.chunk.title,
        "text": result.chunk.text,
        "metadata": result.chunk.metadata,
        "details": result.details,
    }


def write_trace_jsonl(trace: AgentTrace, path: Path) -> None:
    """把一条 AgentTrace 追加写入 JSONL 文件。

    JSONL 是“一行一个 JSON”的格式，适合持续追加请求记录，也方便后续
    evaluation 模块逐行读取分析。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def read_traces_jsonl(path: Path) -> list[AgentTrace]:
    """从 JSONL 文件读取 AgentTrace 列表。"""

    if not path.exists():
        return []

    traces: list[AgentTrace] = []
    with path.open("r", encoding="utf-8") as trace_file:
        for line in trace_file:
            if not line.strip():
                continue
            traces.append(AgentTrace.from_dict(json.loads(line)))
    return traces


def _normalize_error_type(error_type: str) -> ErrorType:
    """把外部读取的 error_type 归一化到支持列表。"""

    supported_errors = set(ErrorType.__args__)
    if error_type in supported_errors:
        return error_type  # type: ignore[return-value]
    return "unknown_error"
