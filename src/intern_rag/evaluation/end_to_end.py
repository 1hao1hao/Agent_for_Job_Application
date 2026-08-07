from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from intern_rag.agent.answer import compose_answer
from intern_rag.agent.context import build_context
from intern_rag.agent.evidence import EvidenceConfig, check_evidence
from intern_rag.agent.schemas import RagResponse
from intern_rag.evaluation.dataset import EvaluationCase
from intern_rag.evaluation.metrics import (
    calculate_citation_validity,
    calculate_key_point_coverage,
    calculate_recall_at_k,
    summarize_end_to_end_results,
)
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import Retriever
from intern_rag.routing import Router
from intern_rag.tracing import build_agent_trace


@dataclass(frozen=True)
class EndToEndRunConfig:
    """确定性 extractive 端到端 baseline 的可复现配置。"""

    run_id: str
    dataset_version: str
    split: str
    router_name: str
    retriever_name: str
    top_k: int
    context_max_chars: int
    key_point_threshold: float
    git_revision: str
    command: str
    generator_name: str = "deterministic_extractive_v1"
    unsupported_grader: str = "extractive_contract_not_human_semantic_review"


@dataclass(frozen=True)
class EndToEndRunResult:
    """端到端逐 Case、Trace、失败和汇总工件。"""

    config: EndToEndRunConfig
    summary: dict[str, object]
    case_results: list[dict[str, object]]
    failures: list[dict[str, object]]
    traces: list[dict[str, object]]


def run_extractive_end_to_end_evaluation(
    cases: list[EvaluationCase],
    chunks: list[Chunk],
    config: EndToEndRunConfig,
    router: Router,
    retriever: Retriever,
    *,
    evidence_config: EvidenceConfig = EvidenceConfig(),
) -> EndToEndRunResult:
    """运行不调用 LLM 的确定性证据摘录评测。

    输入是评测标签、版本化 Chunks、Run 配置以及可注入的 Router/Retriever；函数
    先筛选指定 split，再逐条执行路由、检索、Evidence Gate、Context 和摘录式回答，
    最后汇总端到端指标、延迟、失败记录与请求级 Trace。返回的
    ``EndToEndRunResult`` 是内存中的标准评测工件，之后可由
    ``save_end_to_end_artifacts`` 落盘。

    该函数被保留为可复现的 extractive baseline；真实 LLM 评测由
    ``run_live_llm_end_to_end_evaluation`` 执行，两者不能混称。
    """

    selected = [case for case in cases if case.split == config.split]
    if not selected:
        raise ValueError(f"dataset has no cases for split={config.split}")

    case_results: list[dict[str, object]] = []

    traces: list[dict[str, object]] = []

    for case in selected:
        case_result, trace = _run_extractive_case(
            case, chunks, config, router, retriever, evidence_config
        )
        case_results.append(case_result)
        traces.append(trace)

    metric_summary = summarize_end_to_end_results(
        case_results,
        key_point_threshold=config.key_point_threshold,
    )
    summary = {
        "run_id": config.run_id,
        "report_status": "formal_deterministic_extractive_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config.dataset_version,
        "split": config.split,
        "case_count": len(case_results),
        **{key: value for key, value in metric_summary.items() if key != "case_metrics"},
        "latency_ms": _latency_summary(case_results),
        "tokens": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "status": "unavailable_no_llm_call",
        },
        "estimated_cost_usd": None,
        "cost_status": "unavailable_no_llm_call",
        "limitations": [
            "Generator is deterministic extractive baseline, not an LLM.",
            "Unsupported rate is contract-level because answers only quote cited snippets; it is not independent human semantic review.",
            "Citation validity does not prove citation support or answer correctness.",
        ],
    }
    failures = build_end_to_end_failures(metric_summary["case_metrics"])  # type: ignore[arg-type]
    return EndToEndRunResult(config, summary, case_results, failures, traces)


def build_end_to_end_failures(
    evaluated: list[dict[str, object]],
) -> list[dict[str, object]]:
    """把未满足端到端成功条件的样例整理为可排查失败记录。"""

    return [
        {
            "case_id": item["case_id"],
            "category": item["category"],
            "failure_type": _e2e_failure_type(item),
            "metrics": item,
        }
        for item in evaluated  # type: ignore[union-attr]
        if item["end_to_end_success"] is not True
    ]


def save_end_to_end_artifacts(
    result: EndToEndRunResult,
    run_dir: Path,
) -> None:
    """保存 config、summary、case、failure、trace 和 latency 标准工件。"""

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_config.json", asdict(result.config))
    _write_json(run_dir / "summary.json", result.summary)
    _write_json(run_dir / "latency.json", dict(result.summary["latency_ms"]))
    _write_jsonl(run_dir / "case_results.jsonl", result.case_results)
    _write_jsonl(run_dir / "failures.jsonl", result.failures)
    _write_jsonl(run_dir / "traces.jsonl", result.traces)


def _run_extractive_case(
    case: EvaluationCase,
    chunks: list[Chunk],
    config: EndToEndRunConfig,
    router: Router,
    retriever: Retriever,
    evidence_config: EvidenceConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    """执行一条 extractive Case，返回 prediction/指标输入和 Agent Trace。

    处理顺序为 Router -> Retriever -> Evidence Gate（必要时扩源一次）-> Context
    Builder -> 摘录式 Generator。随后把系统输出与 ``EvaluationCase`` 标签对照，
    计算该 Case 的路由正确性、Recall@5、Citation Validity 和 Key-Point Coverage，
    并记录各阶段耗时与 attempt。第一个返回值供指标汇总和失败分析使用，第二个
    返回值用于请求级 Trace 落盘。

    Evidence 不足时不会强行生成答案，而是返回 ``insufficient_evidence``；这个
    受控拒答仍会形成完整的 Case Result 和 Trace。
    """

    total_started = perf_counter()
    route_started = perf_counter()
    route = router(case.query)
    routing_ms = _elapsed_ms(route_started)
    attempts: list[dict[str, object]] = []
    results = []
    decision = None
    retrieval_ms = 0.0
    evidence_ms = 0.0
    for retry_count in range(2):
        retrieval_started = perf_counter()
        source_types = set(route.routed_sources) if retry_count == 0 else None
        results = retriever(
            case.query, chunks, top_k=config.top_k, source_types=source_types
        )
        retrieval_attempt_ms = _elapsed_ms(retrieval_started)
        retrieval_ms += retrieval_attempt_ms
        evidence_started = perf_counter()
        decision = check_evidence(
            route,
            results,
            retriever_name=(
                "hybrid" if config.retriever_name == "hybrid_rerank"
                else config.retriever_name
            ),
            retry_count=retry_count,
            max_retries=1,
            config=evidence_config,
        )
        evidence_attempt_ms = _elapsed_ms(evidence_started)
        evidence_ms += evidence_attempt_ms
        attempts.append({
            "attempt": retry_count + 1,
            "type": "initial_retrieval" if retry_count == 0 else "source_expansion",
            "source_filter": sorted(source_types) if source_types else None,
            "retrieved_chunk_ids": [result.chunk_id for result in results],
            "evidence": asdict(decision),
            "latency_ms": {
                "retrieval": retrieval_attempt_ms,
                "evidence": evidence_attempt_ms,
            },
        })
        if decision.status != "retryable":
            break

    context_started = perf_counter()
    built_context = build_context(
        case.query,
        results if decision and decision.status == "sufficient" else [],
        max_chars=config.context_max_chars,
    )
    context_ms = _elapsed_ms(context_started)
    generation_started = perf_counter()
    if decision and decision.status == "sufficient":
        usable = [
            result for result in results
            if result.chunk_id in set(built_context.used_chunk_ids)
        ]
        answer_result = compose_answer(case.query, usable, max_chunks=3)
        response = RagResponse(
            request_id=case.case_id,
            trace_id=f"trace-{config.run_id}-{case.case_id}",
            answer=answer_result.answer,
            citations=answer_result.citations,
            routed_sources=route.routed_sources,
            status="answered",
            latency_ms=0.0,
        )
    else:
        response = RagResponse(
            request_id=case.case_id,
            trace_id=f"trace-{config.run_id}-{case.case_id}",
            answer="当前证据不足，无法基于已提供的资料可靠回答该问题。",
            citations=[],
            routed_sources=route.routed_sources,
            status="insufficient_evidence",
            latency_ms=0.0,
            error_type=(
                None if decision and decision.reason == "unanswerable_route"
                else "retrieval_miss"
            ),
        )
    generation_ms = _elapsed_ms(generation_started)
    total_ms = _elapsed_ms(total_started)
    citation_ids = [citation.chunk_id for citation in response.citations]
    citation_validity = calculate_citation_validity(
        citation_ids, built_context.used_chunk_ids, status=response.status
    )
    coverage, covered_points = calculate_key_point_coverage(
        response.answer, case.expected_points
    )
    retrieved_ids = [result.chunk_id for result in results]
    recall_at_5 = (
        calculate_recall_at_k(retrieved_ids, case.relevant_chunk_ids, 5)
        if case.answerable else None
    )
    router_correct = (
        route.intent == case.expected_intent
        and set(route.routed_sources) == set(case.expected_sources)
    )
    result = {
        "case_id": case.case_id,
        "query": case.query,
        "category": case.category,
        "split": case.split,
        "answerable": case.answerable,
        "expected_points": case.expected_points,
        "status": response.status,
        "answer": response.answer,
        "citation_ids": citation_ids,
        "context_ids": built_context.used_chunk_ids,
        "citation_validity": citation_validity,
        "key_point_coverage": coverage,
        "covered_points": covered_points,
        "unsupported_answer": False if response.status == "answered" else None,
        "unsupported_grader": config.unsupported_grader,
        "router_correct": router_correct,
        "recall_at_5": recall_at_5,
        "latency_ms": {
            "routing": routing_ms,
            "retrieval": retrieval_ms,
            "evidence": evidence_ms,
            "context": context_ms,
            "generation": generation_ms,
            "validation": 0.0,
            "total": total_ms,
        },
    }
    trace = build_agent_trace(
        query=case.query,
        route_decision=route,
        retrieved_results=results,
        latency_ms=dict(result["latency_ms"]),  # type: ignore[arg-type]
        request_id=case.case_id,
        trace_id=response.trace_id,
        citations=[citation.to_dict() for citation in response.citations],
        answer=response.answer,
        context={"used_chunk_ids": built_context.used_chunk_ids},
        evidence=asdict(decision) if decision else {},
        generation={"generator": config.generator_name},
        response_status=response.status,
        attempts=attempts,
        token_usage={"status": "unavailable_no_llm_call"},
    ).to_dict()
    return result, trace


def _latency_summary(case_results: list[dict[str, object]]) -> dict[str, object]:
    stages: dict[str, list[float]] = defaultdict(list)
    for result in case_results:
        for stage, value in dict(result["latency_ms"]).items():  # type: ignore[arg-type]
            stages[str(stage)].append(float(value))
    return {
        stage: {
            "count": len(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
        }
        for stage, values in stages.items()
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * percentile + 0.999999))
    return ordered[rank - 1]


def _e2e_failure_type(item: dict[str, object]) -> str:
    if bool(item["unexpected_abstention"]):
        return "unexpected_abstention"
    if bool(item["should_abstain_but_answered"]):
        return "should_abstain_but_answered"
    if item["citation_validity"] != 1.0 and bool(item["answerable"]):
        return "citation_invalid"
    if not bool(item["router_correct"]):
        return "router_wrong"
    if float(item["recall_at_5"] or 0.0) <= 0 and bool(item["answerable"]):
        return "retrieval_miss"
    return "key_point_incomplete"


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
