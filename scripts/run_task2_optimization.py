from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.evaluation import (  # noqa: E402
    calculate_ndcg_at_k,
    calculate_recall_at_k,
    load_chunks_jsonl,
)
from intern_rag.evaluation.knowledge_dataset import (  # noqa: E402
    KnowledgeEvaluationCase,
    load_knowledge_dataset,
)
from intern_rag.retrieval import (  # noqa: E402
    AdaptiveRetriever,
    build_retriever_from_config,
)


POLICIES = ("never", "always", "low_confidence")
ELIGIBLE_NEEDS = (
    "exact_fact",
    "semantic_explanation",
    "multi_source_synthesis",
    "balanced_retrieval",
)


def main() -> int:
    """在 v0.3/dev 上完成 Task 2 的 Rerank 归因、策略选择与 Rewrite 审计。

    输入固定的 160 条 dev Case、版本化 Chunk 和 Adaptive v2 配置；复用同一个
    Retriever/CrossEncoder 实例依次运行 Never、Always、旧 low-confidence，并按
    EvidenceRequirement 的真实收益选择第四组策略。输出四组逐 Case prediction、
    45 条历史 retrieval miss 的 ranking/recall 分类、Query Rewrite 决策及锁定配置。
    脚本拒绝 test split，也不会修改 benchmark 标签。
    """

    args = _parse_args()
    output = ROOT / "reports" / "ablations" / args.run_id
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = [
        case
        for case in load_knowledge_dataset(
            ROOT / "data/evaluation/evalrag_v0.3.jsonl"
        )
        if case.split == "dev"
    ]
    raw = _load_config(ROOT / args.retriever_config)
    raw["reranker_candidate_k"] = args.candidate_k
    retriever = build_retriever_from_config(raw)
    if not isinstance(retriever, AdaptiveRetriever):
        raise TypeError("Task 2 requires an AdaptiveRetriever")

    all_rows: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for policy in POLICIES:
        retriever.config = replace(
            retriever.config,
            candidate_k=args.candidate_k,
            rerank_policy=policy,
            rerank_need_types=(),
        )
        rows = _run_cases(cases, chunks, retriever)
        all_rows[policy] = rows
        summaries[policy] = _summarize(rows)
        _save_run(args.run_id, policy, raw, rows, summaries[policy])

    selected_needs, need_decisions = _select_need_types(
        all_rows["never"], all_rows["always"], summaries
    )
    retriever.config = replace(
        retriever.config,
        candidate_k=args.candidate_k,
        rerank_policy="evidence_need",
        rerank_need_types=tuple(selected_needs),
    )
    evidence_rows = _run_cases(cases, chunks, retriever)
    all_rows["evidence_need"] = evidence_rows
    summaries["evidence_need"] = _summarize(evidence_rows)

    locked = dict(raw)
    locked.update({
        "config_version": "adaptive-retrieval-v2.1-evidence-rerank-dev-locked",
        "adaptive_rerank_policy": "evidence_need",
        "adaptive_rerank_need_types": selected_needs,
        "reranker_candidate_k": args.candidate_k,
        "rerank_selection_basis": (
            "evalrag_v0.3/dev only; per-need MRR and NDCG@5 improve, "
            "Recall@5 stays within one-case tolerance, and recovered cases "
            "are not fewer than regressed cases"
        ),
    })
    _save_locked_config(ROOT / args.locked_config, locked)
    _save_run(args.run_id, "evidence_need", locked, evidence_rows, summaries["evidence_need"])

    misses = _load_historical_misses()
    miss_analysis = _classify_misses(misses, all_rows)
    rewrite_audit = _query_rewrite_audit(misses)
    differences = _strategy_differences(all_rows)[:15]
    payload = {
        "run_id": args.run_id,
        "dataset_version": "evalrag_v0.3",
        "split": "dev",
        "case_count": len(cases),
        "answerable_case_count": sum(case.answerable for case in cases),
        "candidate_k": args.candidate_k,
        "variants": summaries,
        "evidence_need_decisions": need_decisions,
        "selected_rerank_need_types": selected_needs,
        "historical_retrieval_miss_analysis": miss_analysis,
        "query_rewrite": rewrite_audit,
        "representative_differences": differences,
        "locked_config": args.locked_config,
        "boundary": "dev-only selection; evalrag_v0.3/test was not read or run",
    }
    _write_json(output / "summary.json", payload)
    _write_jsonl(output / "miss_analysis.jsonl", miss_analysis["cases"])
    _write_json(output / "query_rewrite_audit.json", rewrite_audit)
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvalRAG Task 2 dev optimization")
    parser.add_argument(
        "--run-id", default="p1-task2-rerank-context-v03-dev-20261006"
    )
    parser.add_argument(
        "--retriever-config", default="configs/retrieval/adaptive_v2_v0.3.json"
    )
    parser.add_argument(
        "--locked-config",
        default="configs/retrieval/adaptive_evidence_rerank_v0.3.json",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("bm25_index_path", "index_dir", "graph_index_path"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            raw[key] = str(ROOT / value)
    return raw


def _save_locked_config(path: Path, raw: dict[str, object]) -> None:
    output = dict(raw)
    for key in ("bm25_index_path", "index_dir", "graph_index_path"):
        value = output.get(key)
        if isinstance(value, str):
            try:
                output[key] = str(Path(value).relative_to(ROOT))
            except ValueError:
                pass
    _write_json(path, output)


def _run_cases(
    cases: list[KnowledgeEvaluationCase], chunks, retriever: AdaptiveRetriever
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        started = perf_counter()
        results = retriever(case.query, chunks, top_k=5, source_types=None)
        latency_ms = (perf_counter() - started) * 1000
        trace = retriever.get_last_trace()
        ids = [item.chunk_id for item in results]
        relevant = list(case.relevant_chunk_ids)
        rows.append({
            "case_id": case.case_id,
            "query": case.query,
            "category": case.category,
            "answerable": case.answerable,
            "relevant_chunk_ids": relevant,
            "retrieved_chunk_ids": ids,
            "candidate_chunk_ids": list(trace.get("candidate_chunk_ids", [])),
            "evidence_need": dict(trace.get("evidence_requirement", {})).get(
                "need_type", "unknown"
            ),
            "selected_strategy": trace.get("selected_strategy", "unknown"),
            "rerank_invoked": bool(trace.get("rerank_invoked", False)),
            "latency_ms": latency_ms,
            "metrics": {
                "recall_at_5": (
                    calculate_recall_at_k(ids, relevant, 5) if case.answerable else None
                ),
                "mrr": _reciprocal_rank(ids, relevant) if case.answerable else None,
                "ndcg_at_5": (
                    calculate_ndcg_at_k(ids, relevant, 5) if case.answerable else None
                ),
            },
            "trace": trace,
        })
    return rows


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    answerable = [row for row in rows if row["answerable"]]
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "metrics": {
            name: mean(float(dict(row["metrics"])[name]) for row in answerable)
            for name in ("recall_at_5", "mrr", "ndcg_at_5")
        },
        "rerank_invocation_count": sum(bool(row["rerank_invoked"]) for row in rows),
        "rerank_invocation_rate": _ratio(
            sum(bool(row["rerank_invoked"]) for row in rows), len(rows)
        ),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "strategy_distribution": dict(
            Counter(str(row["selected_strategy"]) for row in rows)
        ),
        "evidence_need_distribution": dict(
            Counter(str(row["evidence_need"]) for row in rows)
        ),
    }


def _select_need_types(never_rows, always_rows, summaries):
    never = {str(row["case_id"]): row for row in never_rows}
    always = {str(row["case_id"]): row for row in always_rows}
    decisions: dict[str, object] = {}
    selected: list[str] = []
    for need in ELIGIBLE_NEEDS:
        ids = [
            case_id for case_id, row in never.items()
            if row["answerable"] and row["evidence_need"] == need
        ]
        base = _mean_case_metrics([never[case_id] for case_id in ids])
        reranked = _mean_case_metrics([always[case_id] for case_id in ids])
        recovered = sum(
            _hit(always[case_id]) and not _hit(never[case_id]) for case_id in ids
        )
        regressed = sum(
            _hit(never[case_id]) and not _hit(always[case_id]) for case_id in ids
        )
        tolerance = 1.0 / len(ids) if ids else 0.0
        quality_pass = bool(ids) and (
            reranked["mrr"] > base["mrr"]
            and reranked["ndcg_at_5"] > base["ndcg_at_5"]
            and reranked["recall_at_5"] >= base["recall_at_5"] - tolerance
            and recovered >= regressed
        )
        if quality_pass:
            selected.append(need)
        decisions[need] = {
            "case_count": len(ids),
            "never": base,
            "always": reranked,
            "recovered": recovered,
            "regressed": regressed,
            "one_case_tolerance": tolerance,
            "selected": quality_pass,
        }
    decisions["relation_reasoning"] = {
        "selected": False,
        "reason": "graph_hybrid candidates are intentionally not CrossEncoder-reranked",
    }
    decisions["global_latency_context"] = {
        "never_p95_ms": summaries["never"]["latency_ms"]["p95"],
        "always_p95_ms": summaries["always"]["latency_ms"]["p95"],
        "note": "Final policy reports its own measured P95; no invocation-rate target is used.",
    }
    return selected, decisions


def _classify_misses(miss_rows, all_rows):
    never = {str(row["case_id"]): row for row in all_rows["never"]}
    indexed = {
        policy: {str(row["case_id"]): row for row in rows}
        for policy, rows in all_rows.items()
    }
    cases = []
    counts = Counter()
    recovered = Counter()
    for old in miss_rows:
        case_id = str(old["case_id"])
        row = never[case_id]
        relevant = set(str(item) for item in row["relevant_chunk_ids"])
        candidate_ids = [str(item) for item in row["candidate_chunk_ids"]]
        candidate_hits = [item for item in candidate_ids if item in relevant]
        miss_type = "ranking_miss" if candidate_hits else "recall_miss"
        counts[miss_type] += 1
        policy_hits = {}
        for policy, values in indexed.items():
            hit = bool(relevant & set(values[case_id]["retrieved_chunk_ids"]))
            policy_hits[policy] = hit
            if miss_type == "ranking_miss" and hit:
                recovered[policy] += 1
        cases.append({
            "case_id": case_id,
            "query": row["query"],
            "category": row["category"],
            "evidence_need": row["evidence_need"],
            "miss_type": miss_type,
            "relevant_chunk_ids": sorted(relevant),
            "candidate_hit_ids": candidate_hits,
            "candidate_ranks": {
                chunk_id: candidate_ids.index(chunk_id) + 1 for chunk_id in candidate_hits
            },
            "policy_top5_hit": policy_hits,
        })
    return {
        "source": "p1-e2e-abstention-v03-20261005 retrieval_miss cases",
        "case_count": len(cases),
        "ranking_miss_count": counts["ranking_miss"],
        "recall_miss_count": counts["recall_miss"],
        "ranking_miss_recovered": dict(recovered),
        "classification_rule": (
            "ranking_miss iff any labeled relevant chunk appears in the selected "
            "strategy top-20 pre-rerank candidate pool"
        ),
        "representative_cases": [
            *[item for item in cases if item["miss_type"] == "ranking_miss"][:3],
            *[item for item in cases if item["miss_type"] == "recall_miss"][:3],
        ],
        "cases": cases,
    }


def _query_rewrite_audit(miss_rows) -> dict[str, object]:
    markers = ("它", "这个", "那个", "上一个", "刚才", "之前", "继续", "再说")
    retrieval_followups = [
        {"case_id": row["case_id"], "query": row.get("query", "")}
        for row in miss_rows
        if any(marker in str(row.get("query", "")) for marker in markers)
    ]
    context_path = ROOT / "data/evaluation/evalrag_context_v0.1.jsonl"
    context_rows = [json.loads(line) for line in context_path.read_text(encoding="utf-8").splitlines()]
    contextual = [
        row for row in context_rows if row.get("category") in {"reference", "ellipsis"}
    ]
    return {
        "decision": "NOT_JUSTIFIED",
        "retrieval_miss_followup_count": len(retrieval_followups),
        "retrieval_miss_followup_cases": retrieval_followups,
        "context_reference_or_ellipsis_cases": len(contextual),
        "reason": (
            "The 45 real E2E retrieval misses contain no explicit pronoun/ellipsis "
            "follow-up. The context benchmark contains follow-ups but has no raw-query "
            "retrieval gold comparison, so it cannot prove rewrite recovery."
        ),
        "implementation": "not added",
    }


def _strategy_differences(all_rows):
    indexed = {
        name: {str(row["case_id"]): row for row in rows}
        for name, rows in all_rows.items()
    }
    first = indexed["never"]
    output = []
    for case_id, row in first.items():
        if not row["answerable"]:
            continue
        metrics = {
            name: dict(values[case_id]["metrics"]) for name, values in indexed.items()
        }
        values = [float(value["mrr"]) for value in metrics.values()]
        if max(values) != min(values):
            output.append({
                "case_id": case_id,
                "query": row["query"],
                "evidence_need": row["evidence_need"],
                "metrics": metrics,
                "spread": max(values) - min(values),
            })
    return sorted(output, key=lambda item: (-float(item["spread"]), str(item["case_id"])))


def _load_historical_misses() -> list[dict[str, object]]:
    path = ROOT / "reports/ablations/p1-e2e-abstention-v03-20261005/case_results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        row for row in rows
        if row.get("answerable") and row.get("terminal_stage") == "retrieval_miss"
    ]


def _save_run(run_id, policy, raw, rows, summary) -> None:
    run_dir = ROOT / "reports" / "runs" / f"{run_id}-{policy}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_config.json", {
        "dataset": "evalrag_v0.3",
        "split": "dev",
        "policy": policy,
        "retriever_config": _relative_artifact_paths(raw),
    })
    _write_json(run_dir / "summary.json", summary)
    _write_jsonl(run_dir / "case_results.jsonl", rows)


def _relative_artifact_paths(raw: dict[str, object]) -> dict[str, object]:
    """让公开 Run Config 不携带执行机器的绝对主目录。"""

    output = dict(raw)
    for key in ("bm25_index_path", "index_dir", "graph_index_path"):
        value = output.get(key)
        if not isinstance(value, str):
            continue
        try:
            output[key] = str(Path(value).relative_to(ROOT))
        except ValueError:
            pass
    return output


def _report(payload) -> str:
    lines = [
        "# Task 2: Evidence-Need Rerank and Rewrite Decision",
        "",
        "- Dataset: `evalrag_v0.3/dev` (160 cases; no test access).",
        f"- Candidate pool: top-{payload['candidate_k']}.",
        "",
        "| Policy | Recall@5 | MRR | NDCG@5 | Invoke | P95 ms | Ranking misses recovered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    miss = payload["historical_retrieval_miss_analysis"]
    recovered = dict(miss["ranking_miss_recovered"])
    for name, item in payload["variants"].items():
        metric = item["metrics"]
        lines.append(
            f"| {name} | {metric['recall_at_5']:.4f} | {metric['mrr']:.4f} | "
            f"{metric['ndcg_at_5']:.4f} | {item['rerank_invocation_rate']:.2%} | "
            f"{item['latency_ms']['p95']:.2f} | {recovered.get(name, 0)} |"
        )
    lines.extend([
        "",
        "## Failure Attribution",
        "",
        f"- Historical retrieval misses: {miss['case_count']}",
        f"- ranking_miss: {miss['ranking_miss_count']}",
        f"- recall_miss: {miss['recall_miss_count']}",
        "",
        "## Locked Decision",
        "",
        f"- Rerank need types: `{payload['selected_rerank_need_types']}`",
        f"- Query rewrite: `{payload['query_rewrite']['decision']}`",
        f"- Config: `{payload['locked_config']}`",
        "",
        "This report measures retrieval placement only; it does not claim answer accuracy.",
    ])
    return "\n".join(lines) + "\n"


def _mean_case_metrics(rows) -> dict[str, float]:
    if not rows:
        return {"recall_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0}
    return {
        name: mean(float(dict(row["metrics"])[name]) for row in rows)
        for name in ("recall_at_5", "mrr", "ndcg_at_5")
    }


def _hit(row) -> bool:
    return bool(set(row["relevant_chunk_ids"]) & set(row["retrieved_chunk_ids"]))


def _reciprocal_rank(retrieved, relevant) -> float:
    wanted = set(relevant)
    for rank, chunk_id in enumerate(retrieved, 1):
        if chunk_id in wanted:
            return 1.0 / rank
    return 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * fraction + 0.999999))
    return ordered[rank - 1]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
