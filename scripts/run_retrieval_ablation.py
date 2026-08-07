from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


CONFIGS = {
    "keyword": "configs/retrieval/keyword_v0.2.json",
    "dense": "configs/retrieval/dense_v0.2.json",
    "hybrid": "configs/retrieval/hybrid_v0.2.json",
}


def main() -> int:
    """同集运行三种 Retriever，并生成包含策略差异 Case 的消融报告。"""

    args = _parse_args()
    ablation_id = args.run_id or datetime.now(timezone.utc).strftime(
        "retrieval-ablation-v02-%Y%m%dT%H%M%SZ"
    )
    run_dirs: dict[str, Path] = {}
    for name, config_path in CONFIGS.items():
        run_id = f"{ablation_id}-{name}"
        command = [
            sys.executable, "scripts/run_evaluation.py",
            "--dataset-version", args.dataset_version,
            "--split", "dev",
            "--config", config_path,
            "--run-id", run_id,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
        run_dirs[name] = Path("reports/runs") / run_id

    report_dir = Path("reports/ablations") / ablation_id
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        name: _read_json(path / "summary.json")
        for name, path in run_dirs.items()
    }
    cases = {
        name: _read_jsonl(path / "case_results.jsonl")
        for name, path in run_dirs.items()
    }
    differences = _build_differences(cases)
    payload = {
        "ablation_id": ablation_id,
        "dataset_version": args.dataset_version,
        "split": "dev",
        "run_dirs": {name: str(path) for name, path in run_dirs.items()},
        "summaries": summaries,
        "strategy_differences": differences,
    }
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
    cases: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """找出三种策略 Recall/MRR 差异最大的 Case，至少保留五条。"""

    by_strategy = {
        name: {str(item["case_id"]): item for item in items}
        for name, items in cases.items()
    }
    differences: list[dict[str, object]] = []
    for case_id, keyword_case in by_strategy["keyword"].items():
        if not bool(keyword_case["answerable"]):
            continue
        metrics = {
            name: dict(by_strategy[name][case_id]["metrics"])
            for name in CONFIGS
        }
        mrr_values = [float(value["reciprocal_rank"]) for value in metrics.values()]
        recall_values = [float(value["recall_at_5"]) for value in metrics.values()]
        difference = max(mrr_values) - min(mrr_values) + max(recall_values) - min(recall_values)
        differences.append({
            "case_id": case_id,
            "category": keyword_case["category"],
            "query": keyword_case["query"],
            "difference_score": difference,
            "metrics": metrics,
            "top_chunk_ids": {
                name: [
                    item["chunk_id"]
                    for item in dict(by_strategy[name][case_id]["predicted"])["retrieved"][:5]  # type: ignore[index]
                ]
                for name in CONFIGS
            },
        })
    differences.sort(key=lambda item: (-float(item["difference_score"]), str(item["case_id"])))
    return differences[:10]


def _format_report(payload: dict[str, object]) -> str:
    """把消融结果格式化成便于面试复盘的 Markdown。"""

    summaries = dict(payload["summaries"])
    lines = [
        "# Retrieval Dev Ablation",
        "",
        f"- Dataset: `{payload['dataset_version']}`",
        "- Split: `dev`",
        "- 三组运行使用相同 Query、Chunks、Router source filter 和 top-k。",
        "- 本报告只衡量检索，不代表最终答案准确率。",
        "",
        "| Retriever | Recall@3 | Recall@5 | MRR | Retrieval P50 ms | Retrieval P95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CONFIGS:
        summary = dict(summaries[name])
        metrics = dict(summary["metrics"])
        latency = dict(dict(summary["latency_ms"])["retrieval"])
        lines.append(
            f"| {name} | {float(metrics['recall_at_3']):.4f} | "
            f"{float(metrics['recall_at_5']):.4f} | {float(metrics['mrr']):.4f} | "
            f"{float(latency['p50']):.3f} | {float(latency['p95']):.3f} |"
        )
    lines.extend(["", "## Strategy Differences", ""])
    for item in payload["strategy_differences"]:  # type: ignore[union-attr]
        metrics = item["metrics"]
        keyword_mrr = float(metrics["keyword"]["reciprocal_rank"])
        dense_mrr = float(metrics["dense"]["reciprocal_rank"])
        hybrid_mrr = float(metrics["hybrid"]["reciprocal_rank"])
        lines.append(
            f"- `{item['case_id']}` ({item['category']}): {item['query']} "
            f"MRR keyword/dense/hybrid = {keyword_mrr:.3f}/"
            f"{dense_mrr:.3f}/{hybrid_mrr:.3f}。"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "Dense 使用固定 commit 的 BAAI/bge-small-zh-v1.5 预训练中文 embedding；文档向量离线构建。",
        "RRF 只融合排名，不直接相加不可比的 Keyword/Dense 原始分数。",
        "本次 Hybrid 提升 Recall@3/Recall@5，但 MRR 下降；说明候选覆盖改善不等于首位排序改善。",
        "CPU 语义编码显著增加 P95；后续可评估批处理、模型量化或缓存，而不能忽略延迟代价。",
    ])
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="evalrag_v0.2")
    parser.add_argument("--run-id")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
