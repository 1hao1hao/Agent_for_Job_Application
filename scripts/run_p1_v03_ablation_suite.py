from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.evaluation import load_chunks_jsonl  # noqa: E402
from intern_rag.evaluation.knowledge_dataset import (  # noqa: E402
    KnowledgeEvaluationCase,
    load_knowledge_dataset,
)
from intern_rag.evaluation.knowledge_runner import (  # noqa: E402
    KnowledgeRunConfig,
    run_knowledge_evaluation,
    save_knowledge_run,
)
from intern_rag.retrieval import (  # noqa: E402
    build_retriever_from_config,
    load_dense_index,
)
from intern_rag.routing import (  # noqa: E402
    DEFAULT_INTENT_PROTOTYPES,
    HybridRouter,
    HybridRouterConfig,
    SemanticRouter,
    SemanticRouterConfig,
    route_query,
)
from intern_rag.routing.feedback import (  # noqa: E402
    FeedbackRouter,
    JsonlRouterFeedbackStore,
    prototypes_from_feedback,
)


DEFAULT_RUN_ID = "p1-d9-v03-dev-ablation-20260817"
RETRIEVAL_CONFIGS = {
    "keyword": "configs/retrieval/keyword_v0.3.json",
    "bm25": "configs/retrieval/bm25_v0.3.json",
    "dense": "configs/retrieval/dense_v0.3.json",
    "keyword_dense_rrf": "configs/retrieval/hybrid_v0.3.json",
    "bm25_dense_rrf": "configs/retrieval/bm25_hybrid_v0.3.json",
    "graph_only": "configs/retrieval/graph_only_v0.3.json",
    "graph_vector_rrf": "configs/retrieval/graph_vector_v0.3.json",
    "adaptive_graph": "configs/retrieval/graph_adaptive_final_v0.3.json",
}
RERANK_CONFIGS = {
    "never": "configs/retrieval/rerank_never_minilm_v0.3.json",
    "always": "configs/retrieval/rerank_always_minilm_v0.3.json",
    "on_demand": "configs/retrieval/rerank_on_demand_minilm_v0.3.json",
}


def main() -> int:
    """在 v0.3/dev 运行 Router、Retriever 与 Rerank 矩阵并汇总 Context 消融。

    Router 因 v0.3 没有 expected_intent，只计算来源集合与不可回答路由指标；
    Retriever 和 Rerank 使用相同 160 条 dev Case、top-k=5 与固定索引版本。
    Context 复用已经完成的 60 组/300 turns 正式工件，避免无代码变化的重复运行。
    所有 prediction 均由系统产生，80 条 frozen test 不参与本次实验。
    """

    run_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_ID
    chunks = load_chunks_jsonl(
        ROOT / "data/processed/chunks/evalrag_v0.3.jsonl"
    )
    cases = load_knowledge_dataset(
        ROOT / "data/evaluation/evalrag_v0.3.jsonl"
    )
    dev_cases = [case for case in cases if case.split == "dev"]
    output_dir = ROOT / "reports/ablations" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    router = _run_router_matrix(run_id, dev_cases, output_dir)
    retrieval, retrieval_rows = _run_retrieval_matrix(
        run_id, cases, chunks, output_dir
    )
    rerank, rerank_rows = _run_rerank_matrix(
        run_id, cases, chunks, output_dir
    )
    context = json.loads(
        (
            ROOT
            / "reports/ablations/p1-d5-context-memory-v01-dev-20260816-bge/summary.json"
        ).read_text(encoding="utf-8")
    )
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "evalrag_v0.3",
        "split": "dev",
        "case_count": len(dev_cases),
        "router": router,
        "retrieval": retrieval,
        "rerank": rerank,
        "context": {
            "artifact": (
                "reports/ablations/"
                "p1-d5-context-memory-v01-dev-20260816-bge/summary.json"
            ),
            "dataset_version": context["dataset_version"],
            "split": context["split"],
            "case_count": context["case_count"],
            "turn_count": context["turn_count"],
            "strategies": context["strategies"],
            "metric_scope": context["metric_scope"],
        },
        "difference_cases": {
            "retrieval": _strategy_differences(retrieval_rows)[:10],
            "rerank": _strategy_differences(rerank_rows)[:10],
        },
        "supplementary_ablations": {
            "chunking": "reports/ablations/p1-chunking-v02-dev-20260812/",
            "source_balanced_context": (
                "reports/ablations/p1-context-builder-v02-dev-20260811/"
            ),
            "model_gateway_faults": (
                "reports/fault_injection/p1-d7-model-gateway-v01/"
            ),
        },
        "boundary": (
            "dev-only ablation; v0.3 labels are corpus-grounded AI-assisted; "
            "Router reports source routing rather than intent accuracy; "
            "retrieval metrics do not equal answer correctness; frozen test was not rerun"
        ),
    }
    _write_json(output_dir / "summary.json", payload)
    (output_dir / "report.md").write_text(
        _build_report(payload), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "router": {
                    name: round(float(value["source_exact_match"]), 4)
                    for name, value in router["strategies"].items()
                },
                "retrieval": {
                    name: {
                        "recall_at_5": round(
                            float(value["metrics"]["recall_at_5"]), 4
                        ),
                        "mrr": round(float(value["metrics"]["mrr"]), 4),
                        "p95_ms": round(float(value["latency_ms"]["p95"]), 3),
                    }
                    for name, value in retrieval["strategies"].items()
                },
                "rerank": {
                    name: {
                        "recall_at_5": round(
                            float(value["summary"]["metrics"]["recall_at_5"]), 4
                        ),
                        "mrr": round(
                            float(value["summary"]["metrics"]["mrr"]), 4
                        ),
                        "invocation_rate": round(
                            float(value["invocation_rate"]), 4
                        ),
                    }
                    for name, value in rerank["strategies"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_router_matrix(
    run_id: str,
    cases: list[KnowledgeEvaluationCase],
    output_dir: Path,
) -> dict[str, object]:
    """构建四个 Router，并按 v0.3 expected_sources 运行同集来源路由评测。"""

    config = _read_json(ROOT / "configs/routing/hybrid_feedback_v0.3.json")
    _, model = load_dense_index(
        ROOT / "data/processed/indexes/evalrag_v0.3/bge-small-zh-v1.5"
    )
    semantic_config = SemanticRouterConfig(
        float(config["semantic_min_score"]),
        float(config["semantic_min_margin"]),
    )
    hybrid_config = HybridRouterConfig(
        float(config["semantic_override_score"]),
        float(config["semantic_override_margin"]),
        int(config["max_weak_rule_keywords"]),
    )
    semantic = SemanticRouter(model, config=semantic_config)
    hybrid = HybridRouter(route_query, semantic, config=hybrid_config)
    feedback = JsonlRouterFeedbackStore(
        ROOT / str(config["feedback_dataset"])
    ).read_all()
    feedback_semantic = SemanticRouter(
        model,
        prototypes=prototypes_from_feedback(
            DEFAULT_INTENT_PROTOTYPES, feedback
        ),
        config=semantic_config,
    )
    feedback_hybrid = FeedbackRouter(
        HybridRouter(route_query, feedback_semantic, config=hybrid_config),
        feedback,
    )
    routers = {
        "rule": route_query,
        "semantic": semantic,
        "hybrid": hybrid,
        "feedback_hybrid": feedback_hybrid,
    }
    strategies: dict[str, object] = {}
    rows_by_strategy: dict[str, list[dict[str, object]]] = {}
    for name, router in routers.items():
        rows = _evaluate_source_router(router, cases)
        rows_by_strategy[name] = rows
        summary = _summarize_router(rows)
        strategies[name] = summary
        run_dir = ROOT / "reports/runs" / f"{run_id}-router-{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "run_config.json",
            {
                "run_id": f"{run_id}-router-{name}",
                "dataset_version": "evalrag_v0.3",
                "split": "dev",
                "strategy": name,
                "metric_scope": "source routing; expected_intent unavailable",
            },
        )
        _write_json(run_dir / "summary.json", summary)
        _write_jsonl(run_dir / "case_results.jsonl", rows)
        _write_jsonl(
            run_dir / "failures.jsonl",
            [row for row in rows if not row["source_exact"]],
        )
    return {
        "metric_scope": (
            "v0.3 has no expected_intent; source exact/precision/recall and "
            "unanswerable routing are reported instead of Router Intent Accuracy"
        ),
        "model": model.name,
        "model_revision": model.version,
        "feedback_count": len(feedback),
        "strategies": strategies,
        "difference_cases": _router_differences(rows_by_strategy)[:10],
    }


def _evaluate_source_router(router, cases) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        started = perf_counter()
        decision = router(case.query)
        latency = (perf_counter() - started) * 1000
        expected = set(case.expected_sources)
        predicted = set(decision.routed_sources)
        intersection = expected & predicted
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "category": case.category,
                "answerable": case.answerable,
                "expected_sources": sorted(expected),
                "predicted_intent": decision.intent,
                "predicted_sources": sorted(predicted),
                "source_exact": predicted == expected,
                "source_precision": (
                    len(intersection) / len(predicted)
                    if predicted
                    else float(not expected)
                ),
                "source_recall": (
                    len(intersection) / len(expected)
                    if expected
                    else float(not predicted)
                ),
                "unanswerable_correct": (
                    decision.intent == "unknown" and not predicted
                    if not case.answerable
                    else None
                ),
                "latency_ms": latency,
                "reason": decision.reason,
            }
        )
    return rows


def _summarize_router(rows: list[dict[str, object]]) -> dict[str, object]:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "case_count": len(rows),
        "source_exact_match": _mean_bool(rows, "source_exact"),
        "answerable_source_exact_match": _mean_bool(
            answerable, "source_exact"
        ),
        "source_precision": mean(
            float(row["source_precision"]) for row in rows
        ),
        "source_recall": mean(float(row["source_recall"]) for row in rows),
        "unanswerable_accuracy": _mean_bool(
            unanswerable, "unanswerable_correct"
        ),
        "predicted_intents": dict(
            Counter(str(row["predicted_intent"]) for row in rows)
        ),
        "category_source_exact": {
            name: _mean_bool(items, "source_exact")
            for name, items in sorted(groups.items())
        },
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }


def _run_retrieval_matrix(run_id, cases, chunks, output_dir):
    summaries = {}
    rows = {}
    for name, path in RETRIEVAL_CONFIGS.items():
        summary, case_rows = _run_retrieval(
            run_id, f"retrieval-{name}", path, cases, chunks
        )
        summaries[name] = summary
        rows[name] = case_rows
    payload = {
        "metric_scope": (
            "same evalrag_v0.3/dev, no Router source filter, top_k=5; "
            "metrics use 120 answerable cases"
        ),
        "strategies": summaries,
    }
    _write_json(output_dir / "retrieval_matrix.json", payload)
    return payload, rows


def _run_rerank_matrix(run_id, cases, chunks, output_dir):
    strategies = {}
    rows = {}
    for name, path in RERANK_CONFIGS.items():
        summary, case_rows = _run_retrieval(
            run_id, f"rerank-{name}", path, cases, chunks
        )
        traces = [
            dict(dict(row["predicted"]).get("trace", {}))
            for row in case_rows
        ]
        invoked = sum(bool(trace.get("rerank_invoked")) for trace in traces)
        applied = sum(bool(trace.get("rerank_applied")) for trace in traces)
        strategies[name] = {
            "summary": summary,
            "invocation_count": invoked,
            "applied_count": applied,
            "invocation_rate": invoked / len(case_rows) if case_rows else 0.0,
        }
        rows[name] = case_rows
    payload = {
        "controlled_variables": {
            "base_retriever": "BM25 + BGE Dense + RRF",
            "reranker": (
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
                "@1427fd652930e4ba29e8149678df786c240d8825"
            ),
            "candidate_k": 10,
            "top_k": 5,
            "policies": ["never", "always", "low_confidence"],
        },
        "strategies": strategies,
    }
    _write_json(output_dir / "rerank_matrix.json", payload)
    return payload, rows


def _run_retrieval(run_id, name, config_path, cases, chunks):
    raw = _read_json(ROOT / config_path)
    current_run_id = f"{run_id}-{name}"
    run_dir = ROOT / "reports/runs" / current_run_id
    if (run_dir / "summary.json").exists():
        summary = _read_json(run_dir / "summary.json")
        rows = _read_jsonl(run_dir / "case_results.jsonl")
        return summary, rows
    result = run_knowledge_evaluation(
        cases,
        chunks,
        KnowledgeRunConfig(
            run_id=current_run_id,
            dataset_version="evalrag_v0.3",
            graph_version="job-skill-experience-v0.2",
            split="dev",
            strategy=name,
            top_k=5,
            command=(
                "PYTHONPATH=src python "
                f"scripts/run_p1_v03_ablation_suite.py {run_id}"
            ),
            retriever_config=raw,
        ),
        build_retriever_from_config(raw),
    )
    save_knowledge_run(result, run_dir)
    return result.summary, result.case_results


def _strategy_differences(results):
    if len(results) < 2:
        return []
    indexed = {
        name: {str(row["case_id"]): row for row in rows}
        for name, rows in results.items()
    }
    baseline_name = next(iter(results))
    output = []
    for case_id, baseline in indexed[baseline_name].items():
        if not baseline["answerable"]:
            continue
        metrics = {
            name: dict(rows[case_id]["metrics"])
            for name, rows in indexed.items()
        }
        recall = [
            float(value["recall_at_5"] or 0.0) for value in metrics.values()
        ]
        mrr = [
            float(value["reciprocal_rank"] or 0.0)
            for value in metrics.values()
        ]
        output.append(
            {
                "case_id": case_id,
                "category": baseline["category"],
                "query": baseline["query"],
                "difference_score": (
                    max(recall) - min(recall) + max(mrr) - min(mrr)
                ),
                "metrics": metrics,
            }
        )
    return sorted(
        output,
        key=lambda item: (-item["difference_score"], item["case_id"]),
    )


def _router_differences(results):
    indexed = {
        name: {str(row["case_id"]): row for row in rows}
        for name, rows in results.items()
    }
    first = next(iter(indexed.values()))
    output = []
    for case_id, row in first.items():
        predictions = {
            name: {
                "intent": values[case_id]["predicted_intent"],
                "sources": values[case_id]["predicted_sources"],
                "correct": values[case_id]["source_exact"],
            }
            for name, values in indexed.items()
        }
        if len(
            {
                (item["intent"], tuple(item["sources"]))
                for item in predictions.values()
            }
        ) > 1:
            output.append(
                {
                    "case_id": case_id,
                    "category": row["category"],
                    "query": row["query"],
                    "expected_sources": row["expected_sources"],
                    "predictions": predictions,
                }
            )
    return output


def _build_report(payload: dict[str, object]) -> str:
    router = payload["router"]
    retrieval = payload["retrieval"]
    rerank = payload["rerank"]
    context = payload["context"]
    lines = [
        "# P1 v0.3 Dev Unified Ablation Report",
        "",
        f"- Run: `{payload['run_id']}`",
        "- Corpus/benchmark: `evalrag_v0.3/dev`, 160 cases; frozen test 未重跑。",
        "- Router 只报告来源路由指标，因为 v0.3 没有 expected_intent 标签。",
        "",
        "## 1. Router Source Routing",
        "",
        "| Router | Source Exact | Answerable Exact | Source Precision | Source Recall | Unanswerable Acc | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in router["strategies"].items():
        lines.append(
            f"| {name} | {value['source_exact_match']:.2%} | "
            f"{value['answerable_source_exact_match']:.2%} | "
            f"{value['source_precision']:.2%} | {value['source_recall']:.2%} | "
            f"{value['unanswerable_accuracy']:.2%} | "
            f"{value['latency_ms']['p95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 2. Retrieval Strategies",
            "",
            "| Retriever | Recall@3 | Recall@5 | MRR | NDCG@5 | P95 ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in retrieval["strategies"].items():
        metrics = value["metrics"]
        lines.append(
            f"| {name} | {metrics['recall_at_3']:.4f} | "
            f"{metrics['recall_at_5']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['ndcg_at_5']:.4f} | {value['latency_ms']['p95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 3. Rerank Policy",
            "",
            "三路固定 BM25 + Dense + RRF、同一 MiniLM revision、candidate_k=10，"
            "只改变 rerank policy。",
            "",
            "| Policy | Recall@3 | Recall@5 | MRR | NDCG@5 | Invocation | P95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in rerank["strategies"].items():
        summary = value["summary"]
        metrics = summary["metrics"]
        lines.append(
            f"| {name} | {metrics['recall_at_3']:.4f} | "
            f"{metrics['recall_at_5']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['ndcg_at_5']:.4f} | {value['invocation_rate']:.2%} | "
            f"{summary['latency_ms']['p95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Context Engine / Memory",
            "",
            f"- Dataset: `{context['dataset_version']}/dev`; "
            f"{context['case_count']} groups / {context['turn_count']} turns。",
            "- 该部分复用已完成的正式工件，没有因代码未变化重复运行。",
            "",
            "| Strategy | Follow-up | Key-point | Prompt Tokens | Token Reduction vs Raw | Repeat Reads | P95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in context["strategies"].items():
        lines.append(
            f"| {name} | {value['follow_up_success']:.2%} | "
            f"{value['semantic_key_point_coverage']:.2%} | "
            f"{value['mean_prompt_tokens']:.2f} | "
            f"{value['mean_compression_ratio']:.2%} | "
            f"{value['mean_repeated_history_reads']:.2f} | "
            f"{value['latency_ms']['p95']:.3f} |"
        )
    router_rule = router["strategies"]["rule"]
    router_feedback = router["strategies"]["feedback_hybrid"]
    fixed_graph = retrieval["strategies"]["graph_vector_rrf"]
    adaptive_graph = retrieval["strategies"]["adaptive_graph"]
    rerank_never = rerank["strategies"]["never"]
    rerank_always = rerank["strategies"]["always"]
    rerank_demand = rerank["strategies"]["on_demand"]
    summary_recent = context["strategies"]["summary_recent"]
    semantic_memory = context["strategies"]["semantic_memory"]
    lines.extend(
        [
            "",
            "## 5. Findings and Configuration Decision",
            "",
            f"- Router：Rule 与 Feedback Hybrid 的 Source Exact 均为 "
            f"{router_rule['source_exact_match']:.2%}；Feedback 在 v0.3 没有改善，"
            "因为旧反馈锚点没有覆盖新 Query 分布。v0.3 缺少 intent 标签，因此不能"
            "用该结果重新声称 Router Intent Accuracy。",
            f"- Retrieval：固定 Graph+Vector RRF 的 Recall@5/MRR 为 "
            f"{fixed_graph['metrics']['recall_at_5']:.2%}/"
            f"{fixed_graph['metrics']['mrr']:.2%}，高于 Adaptive Graph 的 "
            f"{adaptive_graph['metrics']['recall_at_5']:.2%}/"
            f"{adaptive_graph['metrics']['mrr']:.2%}。当前 selector 漏触发部分关系型"
            " Query，Graph+Vector 是新的 quality-first dev candidate；不重跑 frozen test。",
            f"- Rerank：always 将 MRR 从 "
            f"{rerank_never['summary']['metrics']['mrr']:.2%} 提高到 "
            f"{rerank_always['summary']['metrics']['mrr']:.2%}，但 P95 从 "
            f"{rerank_never['summary']['latency_ms']['p95']:.2f} ms 增至 "
            f"{rerank_always['summary']['latency_ms']['p95']:.2f} ms；on-demand "
            f"调用率 {rerank_demand['invocation_rate']:.2%}，MRR "
            f"{rerank_demand['summary']['metrics']['mrr']:.2%}。三者没有形成统一"
            " Pareto 最优，默认不把 Reranker 包装成质量提升。",
            f"- Context：Semantic Memory 与 Summary+Recent 均保持 "
            f"{semantic_memory['follow_up_success']:.2%} Follow-up Success，平均 "
            f"Prompt Token 从 {summary_recent['mean_prompt_tokens']:.2f} 降至 "
            f"{semantic_memory['mean_prompt_tokens']:.2f}、重复读取从 "
            f"{summary_recent['mean_repeated_history_reads']:.0f} 降至 "
            f"{semantic_memory['mean_repeated_history_reads']:.0f}，但 P95 增至 "
            f"{semantic_memory['latency_ms']['p95']:.2f} ms。",
            "",
            "## 6. Additional Existing Ablations",
            "",
            "- Chunking：段落固定切分 vs 句子边界切分。",
            "- Context Builder：rank-prefix vs source-balanced，不同预算。",
            "- Model Gateway：timeout、429/5xx、熔断、并发和 fallback 故障注入。",
            "",
            "## Boundary",
            "",
            str(payload["boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


def _mean_bool(rows, key):
    return (
        mean(float(bool(row[key])) for row in rows)
        if rows
        else 0.0
    )


def _percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * fraction + 0.999999))
    return ordered[rank - 1]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
