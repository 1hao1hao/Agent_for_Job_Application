from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from statistics import mean
from time import perf_counter

from intern_rag.evaluation.dataset import EvaluationCase
from intern_rag.evaluation.metrics import calculate_recall_at_k
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import Retriever, retrieve_top_k
from intern_rag.routing import Router, route_query


@dataclass(frozen=True)
class EvaluationRunConfig:
    """一次可复现 Router/Retriever 实验的完整配置。"""

    run_id: str
    dataset_version: str
    split: str
    retriever_name: str
    top_k: int
    chunk_max_chars: int
    git_commit: str
    command: str
    candidate_run: bool = False
    retriever_config: dict[str, object] = field(default_factory=dict)
    router_name: str = "rule"
    router_config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """拒绝未知检索器、非法 split 和不足以计算 Recall@5 的配置。"""

        if self.retriever_name not in {
            "keyword", "dense", "hybrid", "hybrid_rerank"
        }:
            raise ValueError(
                "retriever_name must be keyword, dense, hybrid or hybrid_rerank"
            )
        if self.split not in {"dev", "test"}:
            raise ValueError("split must be dev or test")
        if self.top_k < 5:
            raise ValueError("top_k must be at least 5")

    def to_dict(self) -> dict[str, object]:
        """转换为 run_config.json 内容。"""

        return asdict(self)


@dataclass(frozen=True)
class EvaluationRunResult:
    """一次真实 Router/Retriever 运行产生的全部报告数据。"""

    config: EvaluationRunConfig
    summary: dict[str, object]
    case_results: list[dict[str, object]]
    failures: list[dict[str, object]]


def run_retrieval_evaluation(
    cases: list[EvaluationCase],
    chunks: list[Chunk],
    config: EvaluationRunConfig,
    retriever: Retriever,
    router: Router = route_query,
) -> EvaluationRunResult:
    """执行指定 split，prediction 全部由注入的真实 Router/Retriever 产生。"""

    selected_cases = [case for case in cases if case.split == config.split]
    if not selected_cases:
        raise ValueError(f"dataset has no cases for split={config.split}")
    if not config.candidate_run and any(
        not case.human_reviewed for case in selected_cases
    ):
        raise ValueError(
            "formal evaluation requires all selected cases to be human reviewed"
        )

    case_results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for case in selected_cases:
        result = _run_case(case, chunks, config.top_k, retriever, router)
        case_results.append(result)
        failures.extend(_case_failures(result))

    summary = _build_summary(case_results, config)
    return EvaluationRunResult(
        config=config,
        summary=summary,
        case_results=case_results,
        failures=failures,
    )


def run_keyword_evaluation(
    cases: list[EvaluationCase],
    chunks: list[Chunk],
    config: EvaluationRunConfig,
) -> EvaluationRunResult:
    """保留旧调用方式的 Keyword baseline 兼容入口。"""

    if config.retriever_name != "keyword":
        raise ValueError("run_keyword_evaluation requires keyword config")
    return run_retrieval_evaluation(cases, chunks, config, retrieve_top_k)


def save_run_artifacts(result: EvaluationRunResult, run_dir: Path) -> None:
    """保存 run config、summary、逐 case 结果和失败列表。"""

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_config.json", result.config.to_dict())
    _write_json(run_dir / "summary.json", result.summary)
    _write_jsonl(run_dir / "case_results.jsonl", result.case_results)
    _write_jsonl(run_dir / "failures.jsonl", result.failures)


def _run_case(
    case: EvaluationCase,
    chunks: list[Chunk],
    top_k: int,
    retriever: Retriever,
    router: Router,
) -> dict[str, object]:
    """运行一条 case 并保存 expected 与系统 prediction。"""

    total_started_at = perf_counter()
    routing_started_at = perf_counter()
    route = router(case.query)
    routing_latency_ms = _elapsed_ms(routing_started_at)

    retrieval_started_at = perf_counter()
    retrieved_results = retriever(
        case.query,
        chunks,
        top_k=top_k,
        source_types=set(route.routed_sources),
    )
    retrieval_latency_ms = _elapsed_ms(retrieval_started_at)
    retrieved_ids = [result.chunk_id for result in retrieved_results]

    recall_at_3 = (
        calculate_recall_at_k(retrieved_ids, case.relevant_chunk_ids, 3)
        if case.answerable
        else None
    )
    recall_at_5 = (
        calculate_recall_at_k(retrieved_ids, case.relevant_chunk_ids, 5)
        if case.answerable
        else None
    )
    reciprocal_rank = (
        _reciprocal_rank(retrieved_ids, case.relevant_chunk_ids)
        if case.answerable
        else None
    )
    router_correct = (
        route.intent == case.expected_intent
        and set(route.routed_sources) == set(case.expected_sources)
    )
    return {
        "case_id": case.case_id,
        "query": case.query,
        "category": case.category,
        "split": case.split,
        "answerable": case.answerable,
        "human_reviewed": case.human_reviewed,
        "expected": {
            "intent": case.expected_intent,
            "sources": case.expected_sources,
            "relevant_chunk_ids": case.relevant_chunk_ids,
            "expected_points": case.expected_points,
        },
        "predicted": {
            "intent": route.intent,
            "sources": route.routed_sources,
            "matched_keywords": route.matched_keywords,
            "strategy": route.strategy,
            "confidence": route.confidence,
            "reason": route.reason,
            "details": route.details,
            "retrieved": [
                {
                    "chunk_id": result.chunk_id,
                    "rank": result.rank,
                    "score": result.score,
                    "reason": result.reason,
                    "details": result.details,
                }
                for result in retrieved_results
            ],
        },
        "metrics": {
            "router_correct": router_correct,
            "recall_at_3": recall_at_3,
            "recall_at_5": recall_at_5,
            "reciprocal_rank": reciprocal_rank,
        },
        "latency_ms": {
            "routing": routing_latency_ms,
            "retrieval": retrieval_latency_ms,
            "total": _elapsed_ms(total_started_at),
        },
    }


def _case_failures(case_result: dict[str, object]) -> list[dict[str, object]]:
    """根据逐 case 指标生成不可丢弃的失败记录。"""

    failures: list[dict[str, object]] = []
    metrics = dict(case_result["metrics"])  # type: ignore[arg-type]
    if not metrics["router_correct"]:
        failures.append(
            {
                "case_id": case_result["case_id"],
                "category": case_result["category"],
                "failure_type": "router_wrong",
                "expected": case_result["expected"],
                "predicted": case_result["predicted"],
            }
        )
    if (
        case_result["answerable"]
        and metrics["recall_at_5"] is not None
        and float(metrics["recall_at_5"]) < 1.0
    ):
        failures.append(
            {
                "case_id": case_result["case_id"],
                "category": case_result["category"],
                "failure_type": "retrieval_incomplete",
                "expected": case_result["expected"],
                "predicted": case_result["predicted"],
                "recall_at_5": metrics["recall_at_5"],
            }
        )
    return failures


def _build_summary(
    case_results: list[dict[str, object]],
    config: EvaluationRunConfig,
) -> dict[str, object]:
    """按协议汇总 Router、Retrieval 和延迟指标。"""

    answerable_results = [
        result for result in case_results if bool(result["answerable"])
    ]
    router_values = [
        float(bool(dict(result["metrics"])["router_correct"]))  # type: ignore[arg-type]
        for result in case_results
    ]
    recall_3_values = [
        float(dict(result["metrics"])["recall_at_3"])  # type: ignore[arg-type]
        for result in answerable_results
    ]
    recall_5_values = [
        float(dict(result["metrics"])["recall_at_5"])  # type: ignore[arg-type]
        for result in answerable_results
    ]
    reciprocal_ranks = [
        float(dict(result["metrics"])["reciprocal_rank"])  # type: ignore[arg-type]
        for result in answerable_results
    ]
    latencies: dict[str, list[float]] = defaultdict(list)
    for result in case_results:
        for stage, value in dict(result["latency_ms"]).items():  # type: ignore[arg-type]
            latencies[str(stage)].append(float(value))

    category_counts = dict(Counter(str(result["category"]) for result in case_results))
    failure_counts = {
        "router_wrong": sum(
            not bool(dict(result["metrics"])["router_correct"])  # type: ignore[arg-type]
            for result in case_results
        ),
        "retrieval_incomplete": sum(
            bool(result["answerable"])
            and float(dict(result["metrics"])["recall_at_5"]) < 1.0  # type: ignore[arg-type]
            for result in case_results
        ),
    }
    limitations = ["Retrieval metrics do not measure final answer correctness."]
    if config.candidate_run:
        limitations.append(
            "Candidate runs use labels pending human review and cannot be "
            "reported as formal evaluation."
        )

    return {
        "run_id": config.run_id,
        "report_status": (
            "candidate_not_human_verified"
            if config.candidate_run
            else "formal"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config.dataset_version,
        "split": config.split,
        "case_count": len(case_results),
        "answerable_case_count": len(answerable_results),
        "category_counts": category_counts,
        "category_metrics": _build_category_metrics(case_results),
        "failure_counts": failure_counts,
        "metrics": {
            "router_accuracy": mean(router_values) if router_values else 0.0,
            "recall_at_3": mean(recall_3_values) if recall_3_values else 0.0,
            "recall_at_5": mean(recall_5_values) if recall_5_values else 0.0,
            "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        },
        "latency_ms": {
            stage: {
                "count": len(values),
                "p50": _nearest_rank_percentile(values, 0.50),
                "p95": _nearest_rank_percentile(values, 0.95),
            }
            for stage, values in sorted(latencies.items())
        },
        "metric_scope": {
            "router_accuracy": "all selected cases",
            "recall_at_3": "answerable cases with relevant ids",
            "recall_at_5": "answerable cases with relevant ids",
            "mrr": "answerable cases with relevant ids",
        },
        "metric_formulas": {
            "router_accuracy": (
                "strict intent and source-set matches / all selected cases"
            ),
            "recall_at_k": (
                "|top-k retrieved ids intersect relevant ids| / "
                "|relevant ids|, macro average"
            ),
            "mrr": "macro average of 1 / first relevant rank",
            "latency_percentile": "nearest-rank percentile",
        },
        "runtime_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "latency_mode": "single process; exported chunks preloaded",
        },
        "limitations": limitations,
    }


def _build_category_metrics(
    case_results: list[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    """按四类 Query 汇总检索指标，避免平均分掩盖局部退化。"""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in case_results:
        grouped[str(result["category"])].append(result)

    category_metrics: dict[str, dict[str, float | int]] = {}
    for category, results in sorted(grouped.items()):
        answerable = [result for result in results if bool(result["answerable"])]
        category_metrics[category] = {
            "case_count": len(results),
            "answerable_case_count": len(answerable),
            "recall_at_3": _mean_metric(answerable, "recall_at_3"),
            "recall_at_5": _mean_metric(answerable, "recall_at_5"),
            "mrr": _mean_metric(answerable, "reciprocal_rank"),
        }
    return category_metrics


def _mean_metric(results: list[dict[str, object]], metric_name: str) -> float:
    """读取逐 Case 指标并计算均值；空子集返回 0。"""

    values = [
        float(dict(result["metrics"])[metric_name])  # type: ignore[arg-type]
        for result in results
    ]
    return mean(values) if values else 0.0


def _reciprocal_rank(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
) -> float:
    """计算第一条 relevant 证据的倒数排名。"""

    relevant_ids = set(relevant_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    """使用最近秩方法计算延迟百分位。"""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(1, int(len(sorted_values) * percentile + 0.999999))
    return sorted_values[rank - 1]


def _elapsed_ms(started_at: float) -> float:
    """计算毫秒耗时。"""

    return (perf_counter() - started_at) * 1000


def _write_json(path: Path, data: dict[str, object]) -> None:
    """写入格式化 JSON。"""

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """写入 JSONL，即一行一个 case 或 failure。"""

    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
