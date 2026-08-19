from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from intern_rag.evaluation import (
    GraphRunConfig,
    load_chunks_jsonl,
    load_graph_challenge,
    run_graph_evaluation,
    save_graph_run,
    validate_graph_challenge,
)
from intern_rag.retrieval import build_retriever_from_config


CONFIGS = {
    "adaptive_vector": "configs/retrieval/adaptive_v0.2.json",
    "graph_only": "configs/retrieval/graph_only_v0.2.json",
    "graph_vector": "configs/retrieval/graph_adaptive_v0.2.json",
}


def main() -> int:
    """在同一 challenge dev 上比较 Vector、Graph 与 Graph + Vector。"""

    args = _parse_args()
    ablation_id = args.run_id or datetime.now(timezone.utc).strftime(
        "p1-d3-graph-dev-%Y%m%dT%H%M%SZ"
    )
    chunks = load_chunks_jsonl(
        Path("data/processed/chunks/evalrag_v0.2.jsonl")
    )
    cases = load_graph_challenge(
        Path("data/evaluation/evalrag_graph_v0.1.jsonl")
    )
    validation = validate_graph_challenge(
        cases, available_chunk_ids={chunk.id for chunk in chunks}
    )
    if not validation.is_valid:
        print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
        return 1

    run_dirs: dict[str, Path] = {}
    summaries: dict[str, dict[str, object]] = {}
    case_results: dict[str, list[dict[str, object]]] = {}
    for strategy, config_path in CONFIGS.items():
        raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        run_id = f"{ablation_id}-{strategy}"
        run_config = GraphRunConfig(
            run_id=run_id,
            dataset_version="evalrag_graph_v0.1",
            graph_version="job-skill-experience-v0.1",
            split="dev",
            retriever_name=str(raw_config["retriever_name"]),
            top_k=5,
            command=(
                "PYTHONPATH=src python scripts/run_graph_retrieval_ablation.py "
                f"--run-id {ablation_id}"
            ),
            retriever_config=raw_config,
        )
        result = run_graph_evaluation(
            cases,
            chunks,
            run_config,
            build_retriever_from_config(raw_config),
        )
        run_dir = Path("reports/runs") / run_id
        save_graph_run(result, run_dir)
        run_dirs[strategy] = run_dir
        summaries[strategy] = result.summary
        case_results[strategy] = result.case_results

    payload = {
        "ablation_id": ablation_id,
        "dataset_version": "evalrag_graph_v0.1",
        "graph_version": "job-skill-experience-v0.1",
        "split": "dev",
        "case_count": 30,
        "run_dirs": {name: str(path) for name, path in run_dirs.items()},
        "summaries": summaries,
        "strategy_differences": _build_differences(case_results),
        "boundary": (
            "dev-only relation retrieval ablation; frozen 10 cases were not run; "
            "retrieval metrics do not measure final answer correctness"
        ),
    }
    report_dir = Path("reports/ablations") / ablation_id
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text(
        _format_report(payload), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Ablation report written to {report_dir}")
    return 0


def _build_differences(
    results: dict[str, list[dict[str, object]]]
) -> list[dict[str, object]]:
    indexed = {
        name: {str(item["case_id"]): item for item in values}
        for name, values in results.items()
    }
    differences = []
    for case_id, baseline in indexed["adaptive_vector"].items():
        if not bool(baseline["answerable"]):
            continue
        metrics = {
            name: dict(indexed[name][case_id]["metrics"]) for name in CONFIGS
        }
        values = [
            float(metrics[name][metric])
            for name in CONFIGS
            for metric in ("recall_at_5", "reciprocal_rank", "ndcg_at_5")
        ]
        differences.append(
            {
                "case_id": case_id,
                "category": baseline["category"],
                "query": baseline["query"],
                "difference_score": max(values) - min(values),
                "metrics": metrics,
                "graph_trace": dict(
                    indexed["graph_vector"][case_id]["predicted"]  # type: ignore[arg-type]
                ).get("retrieval_trace", {}),
            }
        )
    differences.sort(
        key=lambda item: (-float(item["difference_score"]), str(item["case_id"]))
    )
    return differences[:10]


def _format_report(payload: dict[str, object]) -> str:
    summaries = dict(payload["summaries"])
    lines = [
        "# P1-D3 Graph + Vector Dev Ablation",
        "",
        "- Dataset: `evalrag_graph_v0.1`; split: `dev`; cases: 30。",
        "- 10 条 frozen test 未运行，也未用于配置选择。",
        "- 三种策略使用相同 Chunk、source filter、top-k 和标签。",
        "",
        "| Strategy | Recall@5 | MRR | NDCG@5 | Path Validity | Selector Accuracy | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONFIGS:
        summary = dict(summaries[name])
        metrics = dict(summary["metrics"])
        latency = dict(summary["latency_ms"])
        path_validity = metrics["path_validity"]
        path_text = (
            f"{float(path_validity):.4f}"
            if path_validity is not None
            else "N/A"
        )
        selector_accuracy = metrics["selector_accuracy"]
        selector_text = (
            f"{float(selector_accuracy):.4f}"
            if selector_accuracy is not None
            else "N/A"
        )
        lines.append(
            f"| {name} | {float(metrics['recall_at_5']):.4f} | "
            f"{float(metrics['mrr']):.4f} | {float(metrics['ndcg_at_5']):.4f} | "
            f"{path_text} | "
            f"{selector_text} | "
            f"{float(latency['p50']):.3f} | {float(latency['p95']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Largest Strategy Differences",
            "",
        ]
    )
    for item in payload["strategy_differences"]:  # type: ignore[union-attr]
        metrics = item["metrics"]
        lines.append(
            f"- `{item['case_id']}` ({item['category']}): Recall@5 vector/graph/graph+vector="
            f"{float(metrics['adaptive_vector']['recall_at_5']):.3f}/"
            f"{float(metrics['graph_only']['recall_at_5']):.3f}/"
            f"{float(metrics['graph_vector']['recall_at_5']):.3f}。"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "这些指标只衡量关系证据召回和路径结构，不代表最终答案准确率。",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
