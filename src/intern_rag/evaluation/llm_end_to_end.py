from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable

from intern_rag.agent import (
    EvidenceConfig,
    LlmClient,
    PipelineConfig,
    RagPipeline,
    RagRequest,
)
from intern_rag.evaluation.dataset import EvaluationCase
from intern_rag.evaluation.end_to_end import (
    EndToEndRunResult,
    build_end_to_end_failures,
)
from intern_rag.evaluation.metrics import (
    calculate_citation_validity,
    calculate_key_point_coverage,
    calculate_recall_at_k,
    summarize_end_to_end_results,
)
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import Retriever
from intern_rag.routing import Router
from intern_rag.tracing import AgentTrace, read_traces_jsonl


@dataclass(frozen=True)
class LiveLlmRunConfig:
    """真实 LLM 端到端 Run 的模型、Pipeline、价格与数据版本配置。"""

    run_id: str
    dataset_version: str
    split: str
    router_name: str
    retriever_name: str
    top_k: int
    context_max_chars: int
    key_point_threshold: float
    model: str
    temperature: float
    prompt_version: str
    max_source_retries: int
    max_format_retries: int
    input_cache_hit_usd_per_million: float
    input_cache_miss_usd_per_million: float
    output_usd_per_million: float
    pricing_source: str
    pricing_checked_at: str
    git_revision: str
    command: str
    generator_name: str = "deepseek_chat_json_mode"
    unsupported_grader: str = "manual_support_review"


def run_live_llm_end_to_end_evaluation(
    cases: list[EvaluationCase],
    chunks: list[Chunk],
    config: LiveLlmRunConfig,
    router: Router,
    retriever: Retriever,
    llm_client: LlmClient,
    trace_path: Path,
    *,
    evidence_config: EvidenceConfig = EvidenceConfig(),
    support_labels: dict[str, bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> EndToEndRunResult:
    """通过完整 RagPipeline 运行真实模型并汇总响应、Trace、token 与成本。"""

    selected = [case for case in cases if case.split == config.split]
    if not selected:
        raise ValueError(f"dataset has no cases for split={config.split}")
    if trace_path.exists():
        raise FileExistsError(f"trace already exists: {trace_path}")

    pipeline = RagPipeline(
        chunks=chunks,
        llm_client=llm_client,
        config=PipelineConfig(
            model=config.model,
            temperature=config.temperature,
            prompt_version=config.prompt_version,
            context_max_chars=config.context_max_chars,
            router_name=config.router_name,
            evidence=evidence_config,
            max_source_retries=config.max_source_retries,
            max_format_retries=config.max_format_retries,
        ),
        trace_path=trace_path,
        router=router,
        routers={config.router_name: router},
        retriever=retriever,
        retrievers={config.retriever_name: retriever},
    )
    responses = {}
    for index, case in enumerate(selected, start=1):
        response = pipeline.run(
            RagRequest(
                query=case.query,
                request_id=case.case_id,
                top_k=config.top_k,
                retriever=config.retriever_name,  # type: ignore[arg-type]
            )
        )
        responses[case.case_id] = response
        if index == 1 and response.error_type == "llm_error":
            raise RuntimeError(
                "first live LLM request failed; stop repeated provider calls"
            )
        if progress_callback is not None:
            progress_callback(index, len(selected), case.case_id)
    traces = read_traces_jsonl(trace_path)
    if len(traces) != len(selected):
        raise RuntimeError("trace count does not match selected case count")
    traces_by_case = {trace.request_id: trace for trace in traces}
    case_results = [
        _build_case_result(
            case,
            responses[case.case_id],
            traces_by_case[case.case_id],
            support_labels or {},
        )
        for case in selected
    ]
    metric_summary = summarize_end_to_end_results(
        case_results,
        key_point_threshold=config.key_point_threshold,
    )
    token_summary = _summarize_tokens(traces)
    estimated_cost = _estimate_cost(token_summary, config)
    reviewed_answered = sum(
        result["status"] == "answered"
        and result["unsupported_answer"] is not None
        for result in case_results
    )
    answered = sum(result["status"] == "answered" for result in case_results)
    review_complete = reviewed_answered == answered
    summary = {
        "run_id": config.run_id,
        "report_status": (
            "formal_live_llm"
            if review_complete
            else "live_llm_pending_support_review"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config.dataset_version,
        "split": config.split,
        "case_count": len(case_results),
        **{
            key: value
            for key, value in metric_summary.items()
            if key != "case_metrics"
        },
        "latency_ms": _latency_summary(case_results),
        "tokens": token_summary,
        "estimated_cost_usd": estimated_cost,
        "cost_status": "estimated_from_client_reported_usage",
        "support_review": {
            "grader": config.unsupported_grader,
            "answered_count": answered,
            "reviewed_answered_count": reviewed_answered,
            "complete": review_complete,
        },
        "limitations": [
            "Key-point coverage uses normalized substring matching.",
            "Citation validity does not prove semantic citation support.",
            "Estimated cost uses the pricing snapshot stored in run config.",
        ],
    }
    failures = build_end_to_end_failures(metric_summary["case_metrics"])  # type: ignore[arg-type]
    return EndToEndRunResult(
        config=config,  # type: ignore[arg-type]
        summary=summary,
        case_results=case_results,
        failures=failures,
        traces=[trace.to_dict() for trace in traces],
    )


def _build_case_result(
    case: EvaluationCase,
    response: object,
    trace: AgentTrace,
    support_labels: dict[str, bool],
) -> dict[str, object]:
    citation_ids = [citation.chunk_id for citation in response.citations]
    context_ids = list(trace.context.get("used_chunk_ids", []))
    retrieved_ids = list(trace.retrieval.get("chunk_ids", []))
    coverage, covered_points = calculate_key_point_coverage(
        response.answer, case.expected_points
    )
    return {
        "case_id": case.case_id,
        "query": case.query,
        "category": case.category,
        "split": case.split,
        "answerable": case.answerable,
        "expected_points": case.expected_points,
        "status": response.status,
        "answer": response.answer,
        "citation_ids": citation_ids,
        "context_ids": context_ids,
        "citation_validity": calculate_citation_validity(
            citation_ids, context_ids, status=response.status
        ),
        "key_point_coverage": coverage,
        "covered_points": covered_points,
        "unsupported_answer": (
            support_labels.get(case.case_id)
            if response.status == "answered"
            else None
        ),
        "unsupported_grader": "manual_support_review",
        "router_correct": (
            trace.intent == case.expected_intent
            and set(trace.routed_sources) == set(case.expected_sources)
        ),
        "recall_at_5": (
            calculate_recall_at_k(retrieved_ids, case.relevant_chunk_ids, 5)
            if case.answerable
            else None
        ),
        "latency_ms": trace.latency_ms,
        "error_type": response.error_type,
        "trace_id": trace.trace_id,
    }


def _summarize_tokens(traces: list[AgentTrace]) -> dict[str, object]:
    totals = defaultdict(int)
    call_count = 0
    for trace in traces:
        attempts = trace.token_usage.get("attempts", [])
        for usage in attempts if isinstance(attempts, list) else []:
            if not isinstance(usage, dict):
                continue
            if isinstance(usage.get("input_tokens"), int):
                call_count += 1
            for field in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ):
                value = usage.get(field)
                if isinstance(value, int):
                    totals[field] += value
    return {
        **dict(totals),
        "llm_call_count": call_count,
        "status": "client_reported",
    }


def _estimate_cost(
    tokens: dict[str, object],
    config: LiveLlmRunConfig,
) -> float:
    input_tokens = int(tokens.get("input_tokens", 0))
    cache_hit = int(tokens.get("prompt_cache_hit_tokens", 0))
    cache_miss = int(tokens.get("prompt_cache_miss_tokens", input_tokens - cache_hit))
    output_tokens = int(tokens.get("output_tokens", 0))
    return (
        cache_hit * config.input_cache_hit_usd_per_million
        + cache_miss * config.input_cache_miss_usd_per_million
        + output_tokens * config.output_usd_per_million
    ) / 1_000_000


def _latency_summary(
    case_results: list[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    stages: dict[str, list[float]] = defaultdict(list)
    for result in case_results:
        for stage, value in dict(result["latency_ms"]).items():  # type: ignore[arg-type]
            stages[str(stage)].append(float(value))
    return {
        stage: {
            "count": len(values),
            "mean": mean(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
        }
        for stage, values in stages.items()
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * percentile + 0.999999))
    return ordered[rank - 1]
