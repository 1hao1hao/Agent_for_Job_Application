from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from intern_rag.evaluation.graph_dataset import GraphChallengeCase
from intern_rag.evaluation.metrics import calculate_ndcg_at_k, calculate_recall_at_k
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import Retriever


@dataclass(frozen=True)
class GraphRunConfig:
    """一次关系型检索实验的版本、策略和运行边界。"""

    run_id: str
    dataset_version: str
    graph_version: str
    split: str
    retriever_name: str
    top_k: int
    command: str
    retriever_config: dict[str, object]


@dataclass(frozen=True)
class GraphRunResult:
    """Graph challenge 的 summary、逐 Case 结果和失败记录。"""

    config: GraphRunConfig
    summary: dict[str, object]
    case_results: list[dict[str, object]]
    failures: list[dict[str, object]]


def run_graph_evaluation(
    cases: list[GraphChallengeCase],
    chunks: list[Chunk],
    config: GraphRunConfig,
    retriever: Retriever,
) -> GraphRunResult:
    """运行关系型 challenge，并计算检索、路径、选路和延迟指标。

    只执行配置指定的 split。每条 Case 将 expected 标签与 Retriever 真实输出分开
    保存，计算 Recall@5、MRR、NDCG@5；Graph 路径使用 Retriever 输出的结构校验
    标记，Adaptive 策略折叠为 vector/graph_hybrid 后与标签比较。不可回答 Case
    不进入检索质量分母，但仍参与策略选择统计。所有失败保留到 failures.jsonl。
    """

    selected = [case for case in cases if case.split == config.split]
    if not selected:
        raise ValueError(f"graph dataset has no split={config.split}")
    case_results = [
        _run_graph_case(case, chunks, config.top_k, retriever)
        for case in selected
    ]
    failures = [
        failure
        for result in case_results
        for failure in _graph_case_failures(result)
    ]
    return GraphRunResult(
        config=config,
        summary=_summarize_graph_results(case_results, config),
        case_results=case_results,
        failures=failures,
    )


def save_graph_run(result: GraphRunResult, run_dir: Path) -> None:
    """保存 config、summary、case results 和 failures 四类标准工件。"""

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_config.json", asdict(result.config))
    _write_json(run_dir / "summary.json", result.summary)
    _write_jsonl(run_dir / "case_results.jsonl", result.case_results)
    _write_jsonl(run_dir / "failures.jsonl", result.failures)


def _run_graph_case(
    case: GraphChallengeCase,
    chunks: list[Chunk],
    top_k: int,
    retriever: Retriever,
) -> dict[str, object]:
    started_at = perf_counter()
    results = retriever(
        case.query,
        chunks,
        top_k=top_k,
        source_types=set(case.expected_sources),
    )
    latency_ms = (perf_counter() - started_at) * 1000
    retrieved_ids = [item.chunk_id for item in results]
    trace = _retriever_trace(retriever)
    raw_strategy = str(trace.get("strategy", "unknown"))
    selected_strategy = (
        "graph_hybrid" if raw_strategy == "graph_hybrid" else "vector"
    )
    graph_path_values = [
        int(item.details.get("path_valid", 0))
        for item in results
        if item.details.get("graph_rank") is not None
        or item.details.get("graph_path") is not None
    ]
    metrics = {
        "recall_at_5": (
            calculate_recall_at_k(retrieved_ids, case.relevant_chunk_ids, 5)
            if case.answerable
            else None
        ),
        "reciprocal_rank": (
            _reciprocal_rank(retrieved_ids, case.relevant_chunk_ids)
            if case.answerable
            else None
        ),
        "ndcg_at_5": (
            calculate_ndcg_at_k(retrieved_ids, case.relevant_chunk_ids, 5)
            if case.answerable
            else None
        ),
        "path_validity": (
            mean(graph_path_values) if graph_path_values else None
        ),
        "selector_correct": (
            selected_strategy == case.expected_strategy
            if raw_strategy != "unknown"
            else None
        ),
    }
    return {
        "case_id": case.case_id,
        "query": case.query,
        "split": case.split,
        "category": case.category,
        "answerable": case.answerable,
        "expected": {
            "strategy": case.expected_strategy,
            "sources": case.expected_sources,
            "relevant_chunk_ids": case.relevant_chunk_ids,
            "entities": case.expected_entities,
            "relations": case.expected_relations,
        },
        "predicted": {
            "strategy": selected_strategy,
            "retrieval_trace": trace,
            "retrieved": [
                {
                    "chunk_id": item.chunk_id,
                    "rank": item.rank,
                    "score": item.score,
                    "reason": item.reason,
                    "details": item.details,
                }
                for item in results
            ],
        },
        "metrics": metrics,
        "latency_ms": latency_ms,
    }


def _summarize_graph_results(
    results: list[dict[str, object]], config: GraphRunConfig
) -> dict[str, object]:
    answerable = [item for item in results if bool(item["answerable"])]
    path_values = [
        float(dict(item["metrics"])["path_validity"])  # type: ignore[arg-type]
        for item in results
        if dict(item["metrics"])["path_validity"] is not None  # type: ignore[arg-type]
    ]
    selector_values = [
        float(bool(dict(item["metrics"])["selector_correct"]))  # type: ignore[arg-type]
        for item in results
        if dict(item["metrics"])["selector_correct"] is not None  # type: ignore[arg-type]
    ]
    category_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in results:
        category_groups[str(item["category"])].append(item)
    latencies = [float(item["latency_ms"]) for item in results]
    return {
        "run_id": config.run_id,
        "dataset_version": config.dataset_version,
        "graph_version": config.graph_version,
        "split": config.split,
        "case_count": len(results),
        "answerable_case_count": len(answerable),
        "metrics": {
            "recall_at_5": _mean_metric(answerable, "recall_at_5"),
            "mrr": _mean_metric(answerable, "reciprocal_rank"),
            "ndcg_at_5": _mean_metric(answerable, "ndcg_at_5"),
            "path_validity": mean(path_values) if path_values else None,
            "selector_accuracy": mean(selector_values) if selector_values else None,
        },
        "category_metrics": {
            category: {
                "case_count": len(items),
                "recall_at_5": _mean_metric(
                    [item for item in items if bool(item["answerable"])],
                    "recall_at_5",
                ),
                "mrr": _mean_metric(
                    [item for item in items if bool(item["answerable"])],
                    "reciprocal_rank",
                ),
                "ndcg_at_5": _mean_metric(
                    [item for item in items if bool(item["answerable"])],
                    "ndcg_at_5",
                ),
            }
            for category, items in sorted(category_groups.items())
        },
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "metric_scope": (
            "Recall/MRR/NDCG use answerable cases; selector uses all cases; "
            "path validity uses returned graph paths."
        ),
        "boundary": (
            f"{config.split} relationship retrieval evaluation; metrics do not "
            "measure final answer correctness"
        ),
    }


def _graph_case_failures(result: dict[str, object]) -> list[dict[str, object]]:
    metrics = dict(result["metrics"])  # type: ignore[arg-type]
    failures = []
    if result["answerable"] and float(metrics["recall_at_5"]) < 1.0:
        failures.append(
            {
                "case_id": result["case_id"],
                "category": result["category"],
                "failure_type": "graph_retrieval_incomplete",
                "expected": result["expected"],
                "predicted": result["predicted"],
            }
        )
    if metrics["selector_correct"] is not None and not bool(metrics["selector_correct"]):
        failures.append(
            {
                "case_id": result["case_id"],
                "category": result["category"],
                "failure_type": "graph_strategy_wrong",
                "expected": result["expected"],
                "predicted": result["predicted"],
            }
        )
    if metrics["path_validity"] is not None and float(metrics["path_validity"]) < 1.0:
        failures.append(
            {
                "case_id": result["case_id"],
                "category": result["category"],
                "failure_type": "graph_path_invalid",
                "expected": result["expected"],
                "predicted": result["predicted"],
            }
        )
    return failures


def _retriever_trace(retriever: Retriever) -> dict[str, object]:
    get_last_trace = getattr(retriever, "get_last_trace", None)
    if not callable(get_last_trace):
        return {}
    trace = get_last_trace()
    return dict(trace) if isinstance(trace, dict) else {}


def _reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def _mean_metric(results: list[dict[str, object]], metric: str) -> float:
    values = [
        float(dict(item["metrics"])[metric])  # type: ignore[arg-type]
        for item in results
        if dict(item["metrics"])[metric] is not None  # type: ignore[arg-type]
    ]
    return mean(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * percentile + 0.999999))
    return ordered[rank - 1]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
