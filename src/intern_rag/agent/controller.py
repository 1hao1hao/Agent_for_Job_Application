from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping


AgentActionName = Literal["retrieve", "expand_sources", "generate", "abstain"]


@dataclass(frozen=True)
class AgentControllerConfig:
    """有界 Agent 的动作预算和版本。"""

    version: str = "bounded-agent-controller-v1"
    max_action_steps: int = 4
    max_source_retries: int = 1
    max_format_retries: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.max_action_steps <= 4:
            raise ValueError("max_action_steps must be between 1 and 4")
        if self.max_source_retries not in {0, 1}:
            raise ValueError("max_source_retries must be 0 or 1")
        if self.max_format_retries not in {0, 1}:
            raise ValueError("max_format_retries must be 0 or 1")


@dataclass(frozen=True)
class AgentState:
    """Controller 做下一步决策所需的最小状态，不保存正文或 Prompt。"""

    requested_retriever: str
    route_intent: str
    routed_sources: tuple[str, ...]
    explicit_out_of_domain: bool = False
    evidence_status: str | None = None
    evidence_reason: str | None = None
    evidence_requirement: Mapping[str, object] = field(default_factory=dict)
    retrieval_decision: Mapping[str, object] = field(default_factory=dict)
    source_retry_count: int = 0
    format_retry_count: int = 0
    generation_started: bool = False
    generation_format_error: bool = False
    step: int = 0

    def summary(self) -> dict[str, object]:
        """返回不含用户正文的状态摘要，供 Trace 解释动作原因。"""

        return {
            "route_intent": self.route_intent,
            "routed_sources": list(self.routed_sources),
            "evidence_need": self.evidence_requirement.get("need_type"),
            "effective_retriever": self.retrieval_decision.get(
                "selected_strategy", self.requested_retriever
            ),
            "evidence_status": self.evidence_status,
            "evidence_reason": self.evidence_reason,
            "source_retry_count": self.source_retry_count,
            "format_retry_count": self.format_retry_count,
            "generation_started": self.generation_started,
            "generation_format_error": self.generation_format_error,
        }


@dataclass(frozen=True)
class AgentAction:
    """一次由 Controller 选择、随后由 Pipeline 执行的有限动作。"""

    action: AgentActionName
    reason: str
    step: int
    config_version: str
    strategy: str | None = None
    state_summary: Mapping[str, object] = field(default_factory=dict)

    def to_trace(self) -> dict[str, object]:
        """转换成可持久化的动作 Trace。"""

        return asdict(self)


class AgentController:
    """根据观察到的状态选择下一步，并用硬上限阻止无限循环。

    Controller 不执行 Retriever 或 LLM，也不允许模型生成任意工具名。它只在
    retrieve、expand_sources、generate、abstain 四个动作中选择一个，Pipeline
    负责执行对应能力并把新状态交回来。
    """

    def __init__(self, config: AgentControllerConfig | None = None) -> None:
        self.config = config or AgentControllerConfig()

    def choose(self, state: AgentState) -> AgentAction:
        """根据门控与重试状态选择唯一下一动作。"""

        if state.step >= self.config.max_action_steps:
            return self._action(
                "abstain", "max_action_steps_exhausted", state.step, state
            )
        step = state.step + 1
        if state.evidence_status is None:
            if state.explicit_out_of_domain and not state.routed_sources:
                return self._action(
                    "abstain", "explicit_out_of_domain_without_searchable_source", step, state
                )
            return self._action(
                "retrieve", "initial_evidence_collection", step, state,
                strategy=state.requested_retriever,
            )
        if state.evidence_status == "retryable":
            if state.source_retry_count < self.config.max_source_retries:
                return self._action(
                    "expand_sources", "evidence_gate_requested_source_expansion", step, state,
                    strategy=str(
                        state.retrieval_decision.get(
                            "selected_strategy", state.requested_retriever
                        )
                    ),
                )
            return self._action("abstain", "source_retry_exhausted", step, state)
        if state.evidence_status != "sufficient":
            return self._action("abstain", "evidence_gate_rejected", step, state)
        if state.generation_format_error:
            if state.format_retry_count < self.config.max_format_retries:
                return self._action("generate", "repair_invalid_model_format", step, state)
            return self._action("abstain", "format_retry_exhausted", step, state)
        if not state.generation_started:
            return self._action("generate", "evidence_gate_accepted", step, state)
        return self._action("abstain", "generation_already_completed", step, state)

    def _action(
        self,
        name: AgentActionName,
        reason: str,
        step: int,
        state: AgentState,
        *,
        strategy: str | None = None,
    ) -> AgentAction:
        return AgentAction(
            action=name,
            reason=reason,
            step=step,
            config_version=self.config.version,
            strategy=strategy,
            state_summary=state.summary(),
        )
