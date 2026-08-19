from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Literal
from uuid import uuid4


SpanStatus = Literal["running", "succeeded", "failed", "skipped"]
EntryPoint = Literal["http", "cli", "worker", "replay", "resume"]


@dataclass(frozen=True)
class RunContext:
    """一次 Runtime 执行的版本化上下文，不保存 API key 或原始 Prompt。"""

    run_id: str = field(default_factory=lambda: str(uuid4()))
    request_id: str = ""
    entrypoint: EntryPoint = "cli"
    config: dict[str, object] = field(default_factory=dict)
    artifact_refs: dict[str, str] = field(default_factory=dict)
    dataset_version: str = ""
    model_version: str = ""
    prompt_version: str = ""
    index_version: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def fingerprint(self) -> str:
        """对配置与工件引用生成稳定指纹，用于拒绝错误 checkpoint。"""

        payload = {
            "config": self.config,
            "artifact_refs": self.artifact_refs,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "index_version": self.index_version,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SpanEvent:
    """一个 Runtime 阶段或 attempt 的结构化 span。"""

    span_id: str
    run_id: str
    name: str
    status: SpanStatus
    started_at: str
    ended_at: str
    latency_ms: float
    parent_span_id: str | None = None
    attempt: int = 1
    input_refs: dict[str, str] = field(default_factory=dict)
    output_refs: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, object] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StageCheckpoint:
    """可恢复阶段的状态引用；只保存可序列化状态，不保存外部连接。"""

    run_id: str
    stage: str
    stage_index: int
    fingerprint: str
    state: dict[str, object]
    completed_side_effect_keys: tuple[str, ...]
    artifact_refs: dict[str, str]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True)
class RuntimeExecution:
    """Runtime 输出：业务结果引用、AgentTrace 与 parent/child spans。"""

    run_context: RunContext
    response: object
    trace: object
    spans: tuple[SpanEvent, ...]
    observability_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayResult:
    """Replay 对原始与重放结果的一致性判断。"""

    run_id: str
    replayable: bool
    stage: str
    matched: bool | None
    reason: str
    original: dict[str, object]
    replayed: dict[str, object] | None
