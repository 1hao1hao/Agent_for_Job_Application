from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from intern_rag.evaluation.knowledge_dataset import KnowledgeEvaluationCase
from intern_rag.evaluation.metrics import calculate_ndcg_at_k, calculate_recall_at_k
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import Retriever


@dataclass(frozen=True)
class KnowledgeRunConfig:
    """v0.3 检索实验的版本、策略和命令快照。"""

    run_id: str
    dataset_version: str
    graph_version: str
    split: str
    strategy: str
    top_k: int
    command: str
    retriever_config: dict[str, object]


@dataclass(frozen=True)
class KnowledgeRunResult:
    """v0.3 逐 Case prediction、失败和汇总。"""

    config: KnowledgeRunConfig
    summary: dict[str, object]
    case_results: list[dict[str, object]]
    failures: list[dict[str, object]]


def run_knowledge_evaluation(
    cases: list[KnowledgeEvaluationCase],
    chunks: list[Chunk],
    config: KnowledgeRunConfig,
    retriever: Retriever,
) -> KnowledgeRunResult:
    """在指定 split 上运行真实 Retriever 并计算 Recall/MRR/NDCG/延迟。

    每条 Case 调用一次传入 Retriever，prediction 与人工/AI 辅助标签分开保存。
    Recall@3/5、MRR、NDCG@5 只使用 answerable Case；unanswerable Case 记录是否
    返回候选但不冒充拒答准确率。Retriever 异常会成为受控 failure，不会删除 Case。
    """

    selected = [case for case in cases if case.split == config.split]
    results = [_run_case(case, chunks, config.top_k, retriever) for case in selected]
    failures: list[dict[str, object]] = []
    for result in results:
        metrics = dict(result["metrics"])  # type: ignore[arg-type]
        if result["error"]:
            failures.append(
                {"case_id": result["case_id"], "failure_type": "retriever_error", "error": result["error"]}
            )
        elif result["answerable"] and float(metrics["recall_at_5"] or 0.0) < 1.0:
            failures.append(
                {
                    "case_id": result["case_id"],
                    "category": result["category"],
                    "failure_type": "retrieval_incomplete",
                    "expected": result["expected"],
                    "predicted": result["predicted"],
                }
            )
    return KnowledgeRunResult(
        config=config,
        summary=_summarize(results, config),
        case_results=results,
        failures=failures,
    )


def save_knowledge_run(result: KnowledgeRunResult, run_dir: Path) -> None:
    """保存 config、case results、failures 和 summary 标准工件。"""

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_config.json", asdict(result.config))
    _write_json(run_dir / "summary.json", result.summary)
    _write_jsonl(run_dir / "case_results.jsonl", result.case_results)
    _write_jsonl(run_dir / "failures.jsonl", result.failures)


def _run_case(case, chunks, top_k, retriever):
    started = perf_counter()
    error = None
    try:
        retrieved = retriever(case.query, chunks, top_k=top_k, source_types=None)
    except Exception as caught:  # Runner 必须保留单 Case 外部 adapter 失败。
        retrieved = []
        error = {"type": type(caught).__name__, "message": str(caught)}
    latency = (perf_counter() - started) * 1000
    ids = [item.chunk_id for item in retrieved]
    relevant = list(case.relevant_chunk_ids)
    metrics = {
        "recall_at_3": calculate_recall_at_k(ids, relevant, 3) if case.answerable else None,
        "recall_at_5": calculate_recall_at_k(ids, relevant, 5) if case.answerable else None,
        "reciprocal_rank": _reciprocal_rank(ids, relevant) if case.answerable else None,
        "ndcg_at_5": calculate_ndcg_at_k(ids, relevant, 5) if case.answerable else None,
        "returned_for_unanswerable": bool(ids) if not case.answerable else None,
        "path_validity": _path_validity(retrieved),
    }
    return {
        "case_id": case.case_id,
        "query": case.query,
        "split": case.split,
        "category": case.category,
        "answerable": case.answerable,
        "expected": {
            "sources": list(case.expected_sources),
            "relevant_chunk_ids": relevant,
            "entities": list(case.expected_entities),
            "relations": list(case.expected_relations),
            "graph_edge_ids": list(case.graph_edge_ids),
        },
        "predicted": {
            "retrieved": [
                {
                    "chunk_id": item.chunk_id,
                    "rank": item.rank,
                    "score": item.score,
                    "reason": item.reason,
                    "details": item.details,
                }
                for item in retrieved
            ],
            "trace": _retriever_trace(retriever),
        },
        "metrics": metrics,
        "latency_ms": latency,
        "error": error,
    }


def _summarize(results, config):
    answerable = [item for item in results if item["answerable"] and not item["error"]]
    groups = defaultdict(list)
    for result in results:
        groups[str(result["category"])].append(result)
    latencies = [float(item["latency_ms"]) for item in results]
    return {
        "run_id": config.run_id,
        "dataset_version": config.dataset_version,
        "graph_version": config.graph_version,
        "split": config.split,
        "strategy": config.strategy,
        "case_count": len(results),
        "answerable_case_count": len(answerable),
        "error_count": sum(bool(item["error"]) for item in results),
        "metrics": {
            "recall_at_3": _mean_metric(answerable, "recall_at_3"),
            "recall_at_5": _mean_metric(answerable, "recall_at_5"),
            "mrr": _mean_metric(answerable, "reciprocal_rank"),
            "ndcg_at_5": _mean_metric(answerable, "ndcg_at_5"),
            "path_validity": _mean_optional(results, "path_validity"),
        },
        "category_metrics": {
            category: {
                "case_count": len(items),
                "recall_at_5": _mean_metric(
                    [item for item in items if item["answerable"] and not item["error"]],
                    "recall_at_5",
                ),
                "mrr": _mean_metric(
                    [item for item in items if item["answerable"] and not item["error"]],
                    "reciprocal_rank",
                ),
            }
            for category, items in sorted(groups.items())
        },
        "latency_ms": {"p50": _percentile(latencies, 0.50), "p95": _percentile(latencies, 0.95)},
        "metric_scope": "Retrieval metrics use answerable cases only; they do not measure answer correctness.",
    }


def _retriever_trace(retriever):
    getter = getattr(retriever, "get_last_trace", None)
    value = getter() if callable(getter) else {}
    return value if isinstance(value, dict) else {}


def _path_validity(results):
    values = [
        float(item.details["path_valid"])
        for item in results
        if item.details.get("path_valid") is not None
    ]
    return mean(values) if values else None


def _reciprocal_rank(retrieved, relevant):
    relevant_set = set(relevant)
    for rank, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def _mean_metric(results, metric):
    values = [float(dict(item["metrics"])[metric]) for item in results]
    return mean(values) if values else 0.0


def _mean_optional(results, metric):
    values = [
        float(dict(item["metrics"])[metric])
        for item in results
        if dict(item["metrics"])[metric] is not None
    ]
    return mean(values) if values else None


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * percentile + 0.999999))
    return ordered[rank - 1]


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
