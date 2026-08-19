from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from intern_rag.agent import RagRequest, RagResponse
from intern_rag.runtime.schemas import (
    RunContext,
    RuntimeExecution,
    SpanEvent,
    StageCheckpoint,
)
from intern_rag.runtime.store import CheckpointStore, SpanSink
from intern_rag.tracing import AgentTrace


class RuntimeExecutor(Protocol):
    """HTTP、CLI 与 Worker 可复用的业务执行接口。"""

    def execute(self, request: RagRequest) -> tuple[RagResponse, AgentTrace]: ...


@dataclass(frozen=True)
class RuntimeStage:
    """checkpoint workflow 的一个纯阶段或带幂等键的外部副作用。"""

    name: str
    handler: Callable[[dict[str, object]], dict[str, object]]
    side_effect_key: str | None = None


class PipelineRuntimeExecutor:
    """把现有 RagPipeline 适配到 RuntimeExecutor，不重复创建 AgentTrace。"""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def execute(self, request: RagRequest) -> tuple[RagResponse, AgentTrace]:
        response = self.pipeline.run(request)
        trace = self.pipeline.last_trace
        if trace is None or trace.trace_id != response.trace_id:
            raise RuntimeError("pipeline did not expose the request AgentTrace")
        return response, trace


class AgentRuntime:
    """统一执行、span、checkpoint/resume 和 sink 故障隔离。

    `execute` 调用已有 Pipeline 后，从同一 AgentTrace 派生阶段与 attempt 子 span；
    `run_stages` 用于需要 checkpoint 的有界工作流。Span sink 失败只记入
    observability_errors，不覆盖 Pipeline 原有 error_type。
    """

    def __init__(
        self,
        executor: RuntimeExecutor | None = None,
        *,
        span_sinks: Sequence[SpanSink] = (),
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.executor = executor
        self.span_sinks = tuple(span_sinks)
        self.checkpoint_store = checkpoint_store

    def execute(self, request: RagRequest, context: RunContext) -> RuntimeExecution:
        if self.executor is None:
            raise RuntimeError("runtime executor is not configured")
        started = perf_counter()
        response, trace = self.executor.execute(request)
        spans = _spans_from_trace(context, trace, (perf_counter() - started) * 1000)
        errors: list[str] = []
        for span in spans:
            for sink in self.span_sinks:
                try:
                    sink.write(span)
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
        return RuntimeExecution(context, response, trace, tuple(spans), tuple(errors))

    def execute_operation(
        self,
        context: RunContext,
        name: str,
        handler: Callable[[], object],
    ) -> object:
        """让 Worker 等非 RagRequest 入口复用相同 root span 与 sink 隔离。"""

        started = perf_counter()
        error: Exception | None = None
        try:
            return handler()
        except Exception as exc:
            error = exc
            raise
        finally:
            now = datetime.now(timezone.utc).isoformat()
            span = SpanEvent(
                str(uuid4()), context.run_id, name,
                "failed" if error is not None else "succeeded",
                context.created_at, now, (perf_counter() - started) * 1000,
                input_refs={"request_id": context.request_id},
                attributes={
                    "entrypoint": context.entrypoint,
                    "config_fingerprint": context.fingerprint,
                },
                error_type=type(error).__name__ if error is not None else None,
                error_message=str(error) if error is not None else "",
            )
            for sink in self.span_sinks:
                try:
                    sink.write(span)
                except Exception:
                    pass

    def run_stages(
        self,
        context: RunContext,
        stages: Sequence[RuntimeStage],
        initial_state: dict[str, object],
        *,
        resume: bool = False,
    ) -> dict[str, object]:
        """顺序运行阶段并在每步落 checkpoint；配置漂移或工件缺失时安全重跑。

        Resume 只跳过 fingerprint 一致、工件仍存在的已完成阶段；带 side_effect_key
        的阶段还会检查已完成键，避免重复模型调用或外部写入。无法信任 checkpoint
        时从 initial_state 重跑，不继续使用旧状态。
        """

        state = dict(initial_state)
        start_index = 0
        completed: set[str] = set()
        if resume and self.checkpoint_store is not None:
            checkpoint = self.checkpoint_store.latest(context.run_id)
            if checkpoint is not None and _checkpoint_is_usable(checkpoint, context):
                state = dict(checkpoint.state)
                start_index = checkpoint.stage_index + 1
                completed.update(checkpoint.completed_side_effect_keys)
        for index, stage in enumerate(stages):
            if index < start_index:
                continue
            if stage.side_effect_key and stage.side_effect_key in completed:
                continue
            state = stage.handler(dict(state))
            if stage.side_effect_key:
                completed.add(stage.side_effect_key)
            if self.checkpoint_store is not None:
                self.checkpoint_store.save(
                    StageCheckpoint(
                        run_id=context.run_id,
                        stage=stage.name,
                        stage_index=index,
                        fingerprint=context.fingerprint,
                        state=dict(state),
                        completed_side_effect_keys=tuple(sorted(completed)),
                        artifact_refs=dict(context.artifact_refs),
                    )
                )
        return state


def _checkpoint_is_usable(checkpoint: StageCheckpoint, context: RunContext) -> bool:
    if checkpoint.fingerprint != context.fingerprint:
        return False
    return all(PathLike.exists(path) for path in checkpoint.artifact_refs.values())


class PathLike:
    """隔离文件引用检查，测试可直接使用普通路径。"""

    @staticmethod
    def exists(value: str) -> bool:
        from pathlib import Path

        return Path(value).exists()


def _spans_from_trace(
    context: RunContext, trace: AgentTrace, runtime_latency_ms: float
) -> list[SpanEvent]:
    now = datetime.now(timezone.utc).isoformat()
    root_id = str(uuid4())
    status = "failed" if trace.error_type != "none" else "succeeded"
    root = SpanEvent(
        root_id,
        context.run_id,
        "agent.run",
        status,  # type: ignore[arg-type]
        trace.created_at or now,
        now,
        runtime_latency_ms,
        input_refs={"request_id": trace.request_id},
        output_refs={"trace_id": trace.trace_id},
        attributes={
            "entrypoint": context.entrypoint,
            "config_fingerprint": context.fingerprint,
            "dataset_version": context.dataset_version,
            "model_version": context.model_version,
            "prompt_version": context.prompt_version,
            "index_version": context.index_version,
            "response_status": trace.response_status,
        },
        error_type=None if trace.error_type == "none" else trace.error_type,
        error_message=trace.error_message,
    )
    spans = [root]
    for name in ("routing", "retrieval", "evidence", "context", "generation", "validation"):
        latency = float(trace.latency_ms.get(name, 0.0))
        attributes = dict(getattr(trace, name, {}) or {})
        # Trace 中可能含正文；Runtime span 只保留 ID、计数、状态和版本化元数据。
        safe = {
            key: value for key, value in attributes.items()
            if key not in {"text", "prompt", "answer", "content", "metadata"}
        }
        if name == "generation":
            safe["token_usage"] = trace.token_usage
        failed = (
            (name == "generation" and trace.error_type in {"llm_error", "llm_timeout", "llm_format_error"})
            or (name == "validation" and trace.error_type in {"citation_error", "citation_invalid"})
            or (name == "retrieval" and trace.error_type in {"retrieval_miss", "retriever_error"})
            or (name == "routing" and trace.error_type == "router_error")
        )
        output_refs = _stage_output_refs(name, safe)
        spans.append(
            SpanEvent(
                str(uuid4()), context.run_id, name,
                "failed" if failed else "succeeded", now, now, latency,
                parent_span_id=root_id,
                input_refs={"request_id": trace.request_id},
                output_refs=output_refs,
                attributes=safe,
                error_type=trace.error_type if failed else None,
                error_message=trace.error_message if failed else "",
            )
        )
    for attempt in trace.attempts:
        latency = sum(float(value) for value in dict(attempt.get("latency_ms", {})).values())
        spans.append(
            SpanEvent(
                str(uuid4()), context.run_id, f"attempt.{attempt.get('type', 'unknown')}",
                "failed" if attempt.get("status") in {"error", "timeout"} else "succeeded",
                now, now, latency, parent_span_id=root_id,
                attempt=int(attempt.get("attempt", 1)),
                attributes={
                    key: value for key, value in attempt.items()
                    if key not in {"token_usage", "prompt", "answer", "content"}
                },
                error_type=str(attempt.get("reason")) if attempt.get("status") in {"error", "timeout"} else None,
            )
        )
    return spans


def _stage_output_refs(name: str, attributes: Mapping[str, object]) -> dict[str, str]:
    """只提取可追踪 ID/状态引用，不把正文复制进 Span。"""

    keys_by_stage = {
        "routing": ("intent", "routed_sources"),
        "retrieval": ("chunk_ids", "result_count"),
        "evidence": ("decision", "reason"),
        "context": ("used_chunk_ids", "selection_strategy"),
        "generation": ("status", "cited_chunk_ids"),
        "validation": ("is_valid", "citation_ids"),
    }
    return {
        key: str(attributes[key])
        for key in keys_by_stage.get(name, ())
        if key in attributes
    }
