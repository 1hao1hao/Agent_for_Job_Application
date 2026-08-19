from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from intern_rag.agent import RagRequest
from intern_rag.runtime.agent_runtime import AgentRuntime
from intern_rag.runtime.schemas import ReplayResult, RunContext, RuntimeExecution


@dataclass(frozen=True)
class SavedRun:
    """Replay 所需的请求、配置引用和原始结果摘要。"""

    context: RunContext
    request: RagRequest
    response_summary: dict[str, object]
    trace_summary: dict[str, object]
    external_model_reproducible: bool


def replay_run(
    saved: SavedRun,
    runtime: AgentRuntime,
    *,
    stage: str = "full",
    stage_replayers: Mapping[str, Callable[[SavedRun], dict[str, object]]] | None = None,
) -> ReplayResult:
    """重放 Fake 全链或指定阶段；外部模型不可复现时明确返回 unavailable。"""

    if stage == "generation" and not saved.external_model_reproducible:
        return ReplayResult(
            saved.context.run_id, False, stage, None,
            "external model output is not reproducible; use saved output or Fake client",
            saved.response_summary, None,
        )
    if stage != "full":
        replayer = (stage_replayers or {}).get(stage)
        if replayer is None:
            return ReplayResult(saved.context.run_id, False, stage, None, "stage replayer missing", {}, None)
        replayed = replayer(saved)
        original = saved.trace_summary.get(stage, {})
        matched = replayed == original
        return ReplayResult(saved.context.run_id, True, stage, matched, "stage replay completed", dict(original), replayed)  # type: ignore[arg-type]
    execution = runtime.execute(
        saved.request,
        RunContext(
            run_id=saved.context.run_id,
            request_id=saved.request.request_id,
            entrypoint="replay",
            config=saved.context.config,
            artifact_refs=saved.context.artifact_refs,
            dataset_version=saved.context.dataset_version,
            model_version=saved.context.model_version,
            prompt_version=saved.context.prompt_version,
            index_version=saved.context.index_version,
        ),
    )
    replayed = _response_summary(execution)
    matched = replayed == saved.response_summary
    return ReplayResult(saved.context.run_id, True, stage, matched, "full replay completed", saved.response_summary, replayed)


def _response_summary(execution: RuntimeExecution) -> dict[str, object]:
    response = execution.response
    return {
        "status": response.status,
        "answer": response.answer,
        "citation_ids": [item.chunk_id for item in response.citations],
        "error_type": response.error_type,
        "attempt_types": [item.get("type") for item in execution.trace.attempts],
    }
