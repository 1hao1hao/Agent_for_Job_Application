from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.evaluation import load_chunks_jsonl  # noqa: E402
from intern_rag.evaluation.knowledge_dataset import (  # noqa: E402
    load_knowledge_dataset,
)
from intern_rag.evaluation.knowledge_runner import (  # noqa: E402
    KnowledgeRunConfig,
    run_knowledge_evaluation,
    save_knowledge_run,
)
from intern_rag.retrieval import build_retriever_from_config  # noqa: E402


CONFIGS = {
    "bm25": "configs/retrieval/bm25_v0.3.json",
    "dense": "configs/retrieval/dense_v0.3.json",
    "adaptive_vector": "configs/retrieval/adaptive_v0.3.json",
    "graph_only": "configs/retrieval/graph_only_v0.3.json",
    "graph_vector": "configs/retrieval/graph_adaptive_v0.3.json",
}


def main() -> int:
    """在相同 v0.3/dev 标签上运行五种策略并保存分类差异报告。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--strategies", nargs="*", choices=tuple(CONFIGS))
    args = parser.parse_args()
    ablation_id = args.run_id or datetime.now(timezone.utc).strftime(
        "p1-d4-v03-dev-%Y%m%dT%H%M%SZ"
    )
    strategies = args.strategies or list(CONFIGS)
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl")
    summaries = {}
    results = {}
    for strategy in strategies:
        config_path = ROOT / CONFIGS[strategy]
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        run_id = f"{ablation_id}-{strategy}"
        run_dir = ROOT / "reports/runs" / run_id
        if (run_dir / "summary.json").exists() and (run_dir / "case_results.jsonl").exists():
            summaries[strategy] = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            results[strategy] = [
                json.loads(line)
                for line in (run_dir / "case_results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            continue
        result = run_knowledge_evaluation(
            cases,
            chunks,
            KnowledgeRunConfig(
                run_id=run_id,
                dataset_version="evalrag_v0.3",
                graph_version="job-skill-experience-v0.2",
                split="dev",
                strategy=strategy,
                top_k=5,
                command=(
                    "PYTHONPATH=src python scripts/run_v03_retrieval_ablation.py "
                    f"--run-id {ablation_id}"
                ),
                retriever_config=raw_config,
            ),
            build_retriever_from_config(raw_config),
        )
        save_knowledge_run(result, run_dir)
        summaries[strategy] = result.summary
        results[strategy] = result.case_results

    comparison = {
        "ablation_id": ablation_id,
        "dataset_version": "evalrag_v0.3",
        "split": "dev",
        "case_count": 160,
        "strategies": strategies,
        "summaries": summaries,
        "largest_differences": _differences(results)[:15],
        "storage_strategies": {
            "pgvector_exact": "requires Docker-backed PostgreSQL/pgvector run",
            "pgvector_hnsw": "requires Docker-backed PostgreSQL/pgvector run",
            "neo4j": "repository equivalence is validated separately",
        },
        "boundary": (
            "dev-only retrieval ablation; 80 frozen test cases were not run; "
            "metrics do not measure final answer correctness"
        ),
    }
    output = ROOT / "reports/ablations" / ablation_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(comparison), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


def _differences(results):
    if len(results) < 2:
        return []
    indexed = {
        strategy: {str(item["case_id"]): item for item in values}
        for strategy, values in results.items()
    }
    baseline_name = next(iter(results))
    output = []
    for case_id, baseline in indexed[baseline_name].items():
        if not baseline["answerable"]:
            continue
        metrics = {
            strategy: dict(items[case_id]["metrics"])
            for strategy, items in indexed.items()
        }
        recalls = [float(value["recall_at_5"] or 0.0) for value in metrics.values()]
        mrrs = [float(value["reciprocal_rank"] or 0.0) for value in metrics.values()]
        output.append(
            {
                "case_id": case_id,
                "category": baseline["category"],
                "query": baseline["query"],
                "difference_score": (max(recalls) - min(recalls)) + (max(mrrs) - min(mrrs)),
                "metrics": metrics,
            }
        )
    output.sort(key=lambda item: (-item["difference_score"], item["case_id"]))
    return output


def _report(payload):
    lines = [
        "# P1-D4 Corpus v0.3 Dev Retrieval Ablation",
        "",
        "- Dataset: `evalrag_v0.3`; split: `dev`; cases: 160。",
        "- 80 条 frozen test 未运行，也未用于配置选择。",
        "",
        "| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in payload["strategies"]:
        summary = payload["summaries"][strategy]
        metrics = summary["metrics"]
        latency = summary["latency_ms"]
        lines.append(
            f"| {strategy} | {metrics['recall_at_3']:.4f} | {metrics['recall_at_5']:.4f} | "
            f"{metrics['mrr']:.4f} | {metrics['ndcg_at_5']:.4f} | "
            f"{latency['p50']:.3f} | {latency['p95']:.3f} |"
        )
    lines.extend(["", "## Strategy Differences", ""])
    for item in payload["largest_differences"]:
        lines.append(f"- `{item['case_id']}` ({item['category']}): {item['query']}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "本报告只衡量检索和图路径，不代表最终回答准确率。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
