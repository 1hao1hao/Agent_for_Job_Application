from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Callable, Literal

from intern_rag.agent import Citation, RagRequest, RagResponse
from intern_rag.agent.generation import LlmClientError
from intern_rag.runtime.agent_runtime import AgentRuntime
from intern_rag.runtime.schemas import (
    ReplayDifference,
    ReplayResult,
    ReplayStageResult,
    RunContext,
)
from intern_rag.tracing import AgentTrace


ReplayStage = Literal[
    "full", "routing", "retrieval", "evidence", "controller", "context",
    "generation", "validation"
]
STAGE_ORDER = (
    "routing", "retrieval", "evidence", "controller", "context", "generation", "validation"
)
RuntimeFactory = Callable[["SavedRun", "ReplayLlmClient"], AgentRuntime]


@dataclass(frozen=True)
class SavedRun:
    """确定性 Replay 所需的历史请求、配置、Trace 和模型输出快照。"""

    context: RunContext
    request: RagRequest
    response_summary: dict[str, object]
    trace_summary: dict[str, object]
    external_model_reproducible: bool
    trace: AgentTrace | None = None
    model_outputs: tuple[str, ...] = ()
    artifact_hashes: dict[str, str] | None = None

    @classmethod
    def from_persisted(
        cls, request: RagRequest, response: RagResponse, trace: AgentTrace
    ) -> "SavedRun":
        """从数据库恢复对象，并验证 Trace 是否包含 Replay 快照。"""

        if not trace.run_context:
            raise ValueError("trace does not contain run_context snapshot")
        context = run_context_from_dict(trace.run_context)
        outputs = tuple(
            str(item["model_output"])
            for item in trace.attempts
            if isinstance(item.get("model_output"), str)
        )
        hashes = trace.run_context.get("artifact_hashes", {})
        return cls(
            context=context,
            request=request,
            response_summary=response_to_summary(response, trace),
            trace_summary={stage: _stage_projection(trace, stage) for stage in STAGE_ORDER},
            external_model_reproducible=bool(outputs) or not _generation_was_called(trace),
            trace=trace,
            model_outputs=outputs,
            artifact_hashes=dict(hashes) if isinstance(hashes, dict) else {},
        )


class ReplayLlmClient:
    """只返回历史模型输出的离线 Client，绝不访问外部网络。"""

    def __init__(self, responses: tuple[str, ...]) -> None:
        self._responses = responses
        self._position = 0
        self.prompts: list[str] = []
        self.last_token_usage: dict[str, int] | None = None

    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        """按历史调用顺序返回原始输出；缺失时受控失败。"""

        del model, temperature
        self.prompts.append(prompt)
        if self._position >= len(self._responses):
            raise LlmClientError("replay model output is unavailable")
        response = self._responses[self._position]
        self._position += 1
        return response


def replay_saved_run(
    saved: SavedRun, runtime_factory: RuntimeFactory, *, stage: ReplayStage = "full"
) -> ReplayResult:
    """离线重放历史 Pipeline，并逐阶段定位首个稳定字段分歧。

    routing 到 validation 的真实代码会重新执行；模型阶段只能消费 Trace 中保存的
    历史原始输出。stage replay 仍执行其必要前置链，但只返回指定阶段的比较结果。
    """

    unavailable = _validate_snapshot(saved, stage)
    if unavailable:
        return ReplayResult(
            saved.context.run_id, False, stage, None, unavailable,
            saved.response_summary, None,
        )
    client = ReplayLlmClient(saved.model_outputs)
    try:
        runtime = runtime_factory(saved, client)
        execution = runtime.execute(
            saved.request,
            RunContext(
                run_id=saved.context.run_id,
                request_id=saved.request.request_id,
                entrypoint="replay",
                config=dict(saved.context.config),
                artifact_refs=dict(saved.context.artifact_refs),
                dataset_version=saved.context.dataset_version,
                model_version=saved.context.model_version,
                prompt_version=saved.context.prompt_version,
                index_version=saved.context.index_version,
            ),
        )
    except Exception as error:
        return ReplayResult(
            saved.context.run_id, False, stage, None,
            f"replay unavailable: {type(error).__name__}: {error}",
            saved.response_summary, None,
        )

    stages = STAGE_ORDER if stage == "full" else (stage,)
    comparisons = tuple(
        _compare_stage(saved.trace, execution.trace, current)  # type: ignore[arg-type]
        for current in stages
    )
    first_divergent = next((item.stage for item in comparisons if not item.matched), None)
    replayed = response_to_summary(execution.response, execution.trace)
    matched = all(item.matched for item in comparisons)
    if stage == "full":
        matched = matched and replayed == saved.response_summary
        if first_divergent is None and replayed != saved.response_summary:
            first_divergent = "response"
    return ReplayResult(
        saved.context.run_id, True, stage, matched,
        "deterministic replay matched" if matched else "deterministic replay diverged",
        saved.response_summary, replayed, comparisons, first_divergent,
    )


def replay_run(saved: SavedRun, runtime: AgentRuntime, *, stage: str = "full", **_: object) -> ReplayResult:
    """兼容旧调用；新代码应使用强制注入离线模型的 replay_saved_run。"""

    if saved.trace is not None:
        return replay_saved_run(saved, lambda _saved, _client: runtime, stage=stage)  # type: ignore[arg-type]
    if stage != "full":
        return ReplayResult(saved.context.run_id, False, stage, None, "trace snapshot missing", {}, None)
    execution = runtime.execute(saved.request, saved.context)
    response = execution.response
    replayed = {
        "status": response.status,
        "answer": response.answer,
        "citation_ids": [item.chunk_id for item in response.citations],
        "error_type": response.error_type,
        "attempt_types": [item.get("type") for item in execution.trace.attempts],
    }
    return ReplayResult(
        saved.context.run_id, True, stage, replayed == saved.response_summary,
        "legacy full replay completed", saved.response_summary, replayed,
    )


def build_run_context_snapshot(context: RunContext) -> dict[str, object]:
    """序列化 RunContext，并记录每个本地文件工件的 SHA-256。"""

    payload = asdict(context)
    payload["artifact_hashes"] = {
        name: _artifact_hash(Path(path))
        for name, path in context.artifact_refs.items()
        if Path(path).exists()
    }
    return payload


def run_context_from_dict(payload: dict[str, object]) -> RunContext:
    """从 Trace 快照恢复版本、配置和工件引用。"""

    return RunContext(
        run_id=str(payload.get("run_id", "")),
        request_id=str(payload.get("request_id", "")),
        entrypoint=str(payload.get("entrypoint", "replay")),  # type: ignore[arg-type]
        config=dict(payload.get("config", {})),  # type: ignore[arg-type]
        artifact_refs=dict(payload.get("artifact_refs", {})),  # type: ignore[arg-type]
        dataset_version=str(payload.get("dataset_version", "")),
        model_version=str(payload.get("model_version", "")),
        prompt_version=str(payload.get("prompt_version", "")),
        index_version=str(payload.get("index_version", "")),
        created_at=str(payload.get("created_at", "")),
    )


def response_to_summary(response: object, trace: AgentTrace) -> dict[str, object]:
    """提取最终响应中适合确定性比较的稳定字段。"""

    return {
        "status": getattr(response, "status", ""),
        "answer": getattr(response, "answer", ""),
        "citation_ids": [item.chunk_id for item in getattr(response, "citations", [])],
        "routed_sources": list(getattr(response, "routed_sources", [])),
        "error_type": getattr(response, "error_type", None),
        "attempt_types": [item.get("type") for item in trace.attempts],
    }


def request_response_from_dict(
    request_payload: dict[str, object], response_payload: dict[str, object]
) -> tuple[RagRequest, RagResponse]:
    """恢复持久化的 RagRequest/RagResponse 契约。"""

    request = RagRequest(**request_payload)  # type: ignore[arg-type]
    citations = [Citation(**item) for item in response_payload.get("citations", [])]  # type: ignore[arg-type]
    response = RagResponse(
        request_id=str(response_payload["request_id"]),
        trace_id=str(response_payload["trace_id"]),
        answer=str(response_payload["answer"]),
        citations=citations,
        routed_sources=list(response_payload.get("routed_sources", [])),  # type: ignore[arg-type]
        status=str(response_payload["status"]),  # type: ignore[arg-type]
        latency_ms=float(response_payload.get("latency_ms", 0.0)),
        error_type=response_payload.get("error_type"),  # type: ignore[arg-type]
    )
    return request, response


def _validate_snapshot(saved: SavedRun, stage: str) -> str | None:
    if stage not in {"full", *STAGE_ORDER}:
        return f"unknown replay stage: {stage}"
    if saved.trace is None:
        return "trace snapshot missing"
    required_versions = {
        "dataset_version": saved.context.dataset_version,
        "index_version": saved.context.index_version,
        "prompt_version": saved.context.prompt_version,
        "model_version": saved.context.model_version,
    }
    missing_versions = [name for name, value in required_versions.items() if not value]
    if missing_versions:
        return f"snapshot version missing: {', '.join(missing_versions)}"
    missing_config = [
        name for name in ("router", "retriever", "top_k")
        if name not in saved.context.config
    ]
    if missing_config:
        return f"snapshot config missing: {', '.join(missing_config)}"
    for name, path_text in saved.context.artifact_refs.items():
        path = Path(path_text)
        if not path.exists():
            return f"artifact missing: {name}={path}"
        expected = (saved.artifact_hashes or {}).get(name)
        if expected and _artifact_hash(path) != expected:
            return f"artifact changed: {name}={path}"
    needs_model = stage in {"full", "generation", "validation"}
    if needs_model and _generation_was_called(saved.trace) and not saved.model_outputs:
        return "historical model output missing; deterministic generation replay unavailable"
    return None


def _compare_stage(original: AgentTrace, replayed: AgentTrace, stage: str) -> ReplayStageResult:
    differences = tuple(
        _structured_diff(_stage_projection(original, stage), _stage_projection(replayed, stage))
    )
    return ReplayStageResult(stage, not differences, differences)


def _stage_projection(trace: AgentTrace, stage: str) -> dict[str, object]:
    if stage == "controller":
        return _remove_volatile({"actions": trace.actions})  # type: ignore[return-value]
    value = dict(getattr(trace, stage, {}) or {})
    if stage == "retrieval":
        value.setdefault("chunk_ids", [item.get("chunk_id") for item in trace.retrieved_chunks])
        value["ranked_chunks"] = [
            {"chunk_id": item.get("chunk_id"), "rank": item.get("rank")}
            for item in trace.retrieved_chunks
        ]
    return _remove_volatile(value)  # type: ignore[return-value]


def _remove_volatile(value: object) -> object:
    volatile = {
        "trace_id", "request_id", "run_id", "span_id", "parent_span_id", "created_at",
        "started_at", "ended_at", "latency_ms", "token_usage", "provider_request_id",
    }
    if isinstance(value, dict):
        return {
            key: _remove_volatile(item)
            for key, item in sorted(value.items())
            if key not in volatile
        }
    if isinstance(value, list):
        return [_remove_volatile(item) for item in value]
    return value


def _structured_diff(original: object, replayed: object, path: str = "$") -> list[ReplayDifference]:
    if isinstance(original, dict) and isinstance(replayed, dict):
        differences: list[ReplayDifference] = []
        for key in sorted(set(original) | set(replayed)):
            differences.extend(_structured_diff(original.get(key), replayed.get(key), f"{path}.{key}"))
        return differences
    if isinstance(original, list) and isinstance(replayed, list):
        differences = []
        for index in range(max(len(original), len(replayed))):
            left = original[index] if index < len(original) else None
            right = replayed[index] if index < len(replayed) else None
            differences.extend(_structured_diff(left, right, f"{path}[{index}]"))
        return differences
    if original != replayed:
        return [ReplayDifference(path, original, replayed)]
    return []


def _generation_was_called(trace: AgentTrace) -> bool:
    return any(
        str(item.get("type", "")) in {"initial_generation", "format_repair", "generation"}
        for item in trace.attempts
    ) or bool(trace.generation)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hash(path: Path) -> str:
    """文件直接散列；目录按相对路径和文件内容生成稳定散列。"""

    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(_sha256(child).encode("ascii"))
    return digest.hexdigest()
