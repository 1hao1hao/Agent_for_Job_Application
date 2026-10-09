from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Literal

from intern_rag.agent import RagResponse
from intern_rag.evaluation.knowledge_dataset import KnowledgeEvaluationCase
from intern_rag.tracing import AgentTrace


TerminalStage = Literal[
    "router_abstain",
    "retrieval_miss",
    "gate_reject",
    "context_evidence_dropped",
    "generator_abstain",
    "citation_invalid",
    "model_error",
    "answered",
]


@dataclass(frozen=True)
class FailureFunnelCase:
    """一条 E2E Case 的终止阶段及其证据流审计结果。"""

    case_id: str
    category: str
    answerable: bool
    terminal_stage: TerminalStage
    response_status: str
    reason: str
    relevant_chunk_ids: tuple[str, ...]
    ever_retrieved_relevant_ids: tuple[str, ...]
    final_retrieved_relevant_ids: tuple[str, ...]
    context_relevant_ids: tuple[str, ...]
    cited_chunk_ids: tuple[str, ...]
    citation_valid: bool
    evidence_status: str
    evidence_reason: str
    error_type: str | None
    latency_ms: float
    trace_id: str

    def to_dict(self) -> dict[str, object]:
        """转换为 JSONL 可写入字典。"""

        return asdict(self)


def classify_failure_stage(
    case: KnowledgeEvaluationCase,
    response: RagResponse,
    trace: AgentTrace,
) -> FailureFunnelCase:
    """根据真实 Trace 判断请求最终停止在哪个阶段。

    函数分别比较全部检索尝试、最终检索结果和 Context 中的 gold Chunk，因而能区分
    “证据从未召回”与“证据已召回但被 Gate、Context 或 Generator 挡住”。它不修改
    prediction，也不把 benchmark 标签传给在线 Pipeline。
    """

    relevant = set(case.relevant_chunk_ids)
    attempt_ids = {
        str(chunk_id)
        for attempt in trace.attempts
        if str(attempt.get("type", "")) in {
            "initial_retrieval", "source_expansion", "evidence_gap_rescue"
        }
        for chunk_id in attempt.get("retrieved_chunk_ids", [])
    }
    final_ids = {
        str(chunk_id) for chunk_id in trace.retrieval.get("chunk_ids", [])
    }
    context_ids = {
        str(chunk_id) for chunk_id in trace.context.get("used_chunk_ids", [])
    }
    cited_ids = tuple(citation.chunk_id for citation in response.citations)
    citation_valid = bool(cited_ids) and all(
        chunk_id in context_ids for chunk_id in cited_ids
    )
    evidence_status = str(trace.evidence.get("status", ""))
    evidence_reason = str(trace.evidence.get("reason", ""))

    if response.status == "answered":
        stage: TerminalStage = "answered"
        reason = "pipeline returned an answered response"
    elif response.error_type in {"citation_error", "citation_invalid"}:
        stage = "citation_invalid"
        reason = trace.error_message or "citation validator rejected model output"
    elif response.status == "error":
        stage = "model_error"
        reason = trace.error_message or response.error_type or "pipeline error"
    elif not trace.attempts or evidence_reason == "unanswerable_route":
        stage = "router_abstain"
        reason = evidence_reason or "router stopped before retrieval"
    elif relevant and not (relevant & attempt_ids):
        stage = "retrieval_miss"
        reason = "no retrieval attempt returned a labeled relevant chunk"
    elif relevant and not (relevant & final_ids):
        stage = "retrieval_miss"
        reason = "a retry displaced previously retrieved relevant evidence"
    elif evidence_status != "sufficient":
        stage = "gate_reject"
        reason = evidence_reason or "Evidence Gate rejected retrieved evidence"
    elif relevant and not (relevant & context_ids):
        stage = "context_evidence_dropped"
        reason = "relevant evidence was retrieved but omitted from model context"
    elif trace.generation and not bool(trace.generation.get("sufficient", False)):
        stage = "generator_abstain"
        reason = str(trace.generation.get("reason", "model marked evidence insufficient"))
    else:
        stage = "model_error"
        reason = trace.error_message or "response could not be mapped to a normal terminal stage"

    return FailureFunnelCase(
        case_id=case.case_id,
        category=case.category,
        answerable=case.answerable,
        terminal_stage=stage,
        response_status=response.status,
        reason=reason,
        relevant_chunk_ids=tuple(sorted(relevant)),
        ever_retrieved_relevant_ids=tuple(sorted(relevant & attempt_ids)),
        final_retrieved_relevant_ids=tuple(sorted(relevant & final_ids)),
        context_relevant_ids=tuple(sorted(relevant & context_ids)),
        cited_chunk_ids=cited_ids,
        citation_valid=citation_valid if response.status == "answered" else True,
        evidence_status=evidence_status,
        evidence_reason=evidence_reason,
        error_type=response.error_type,
        latency_ms=response.latency_ms,
        trace_id=response.trace_id,
    )


def summarize_failure_funnel(
    rows: list[FailureFunnelCase],
) -> dict[str, object]:
    """汇总拒答漏斗和可靠性指标，并明确每个指标的分母。"""

    answerable = [row for row in rows if row.answerable]
    unanswerable = [row for row in rows if not row.answerable]
    answered = [row for row in rows if row.response_status == "answered"]
    answerable_successes = [
        row
        for row in answerable
        if row.terminal_stage == "answered"
        and row.citation_valid
        and bool(row.context_relevant_ids)
    ]
    return {
        "case_count": len(rows),
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(unanswerable),
        "terminal_stage_counts": dict(Counter(row.terminal_stage for row in rows)),
        "answerable_terminal_stage_counts": dict(
            Counter(row.terminal_stage for row in answerable)
        ),
        "answerable_e2e_success": _ratio(len(answerable_successes), len(answerable)),
        "unexpected_abstention_rate": _ratio(
            sum(row.response_status == "insufficient_evidence" for row in answerable),
            len(answerable),
        ),
        "unanswerable_abstention_accuracy": _ratio(
            sum(row.response_status == "insufficient_evidence" for row in unanswerable),
            len(unanswerable),
        ),
        "citation_validity_answered": _ratio(
            sum(row.citation_valid for row in answered), len(answered)
        ),
        "error_count": sum(row.response_status == "error" for row in rows),
        "answered_count": len(answered),
        "mean_latency_ms": mean(row.latency_ms for row in rows) if rows else 0.0,
        "metric_definitions": {
            "answerable_e2e_success": "answerable 且 answered、引用合法、Context 含 gold evidence / answerable cases",
            "unexpected_abstention_rate": "answerable 但 insufficient_evidence / answerable cases",
            "unanswerable_abstention_accuracy": "unanswerable 且 insufficient_evidence / unanswerable cases",
            "citation_validity_answered": "answered 且所有 citation id 均存在于 Context / answered cases",
        },
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
