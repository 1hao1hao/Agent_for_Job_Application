from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


CONFIGS = {
    "keyword": "configs/retrieval/keyword_v0.2.json",
    "bm25": "configs/retrieval/bm25_v0.2.json",
    "dense": "configs/retrieval/dense_v0.2.json",
    "bm25_hybrid": "configs/retrieval/bm25_hybrid_v0.2.json",
}


def main() -> int:
    """在相同 dev 数据、Router 与 top-k 下运行四种检索策略并保存差异 Case。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="evalrag_v0.2")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    ablation_id = args.run_id or datetime.now(timezone.utc).strftime(
        "p1-d1-bm25-dev-%Y%m%dT%H%M%SZ"
    )
    run_dirs: dict[str, Path] = {}
    for name, config_path in CONFIGS.items():
        run_id = f"{ablation_id}-{name}"
        command = [
            sys.executable,
            "scripts/run_evaluation.py",
            "--dataset-version",
            args.dataset_version,
            "--split",
            "dev",
            "--config",
            config_path,
            "--run-id",
            run_id,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
        run_dirs[name] = Path("reports/runs") / run_id

    summaries = {
        name: _read_json(path / "summary.json") for name, path in run_dirs.items()
    }
    cases = {
        name: _read_jsonl(path / "case_results.jsonl")
        for name, path in run_dirs.items()
    }
    differences = _build_differences(cases)
    report_dir = Path("reports/ablations") / ablation_id
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ablation_id": ablation_id,
        "dataset_version": args.dataset_version,
        "split": "dev",
        "case_count": int(summaries["keyword"]["case_count"]),
        "run_dirs": {name: str(path) for name, path in run_dirs.items()},
        "summaries": summaries,
        "strategy_differences": differences,
        "boundary": "dev-only; retrieval metrics do not measure answer accuracy",
    }
    (report_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text(_format_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Ablation report written to {report_dir}")
    return 0


def _build_differences(
    cases: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """按 Recall@5 与首个相关结果排名差异选出至少五条策略差异 Case。"""

    by_strategy = {
        name: {str(item["case_id"]): item for item in values}
        for name, values in cases.items()
    }
    differences: list[dict[str, object]] = []
    for case_id, baseline in by_strategy["keyword"].items():
        if not bool(baseline["answerable"]):
            continue
        metrics = {
            name: dict(by_strategy[name][case_id]["metrics"]) for name in CONFIGS
        }
        mrr_values = [float(value["reciprocal_rank"]) for value in metrics.values()]
        recall_values = [float(value["recall_at_5"]) for value in metrics.values()]
        difference_score = (
            max(mrr_values) - min(mrr_values)
            + max(recall_values) - min(recall_values)
        )
        differences.append(
            {
                "case_id": case_id,
                "category": baseline["category"],
                "query": baseline["query"],
                "difference_score": difference_score,
                "metrics": metrics,
                "top_chunk_ids": {
                    name: [
                        item["chunk_id"]
                        for item in dict(by_strategy[name][case_id]["predicted"])[
                            "retrieved"
                        ][:5]
                    ]
                    for name in CONFIGS
                },
            }
        )
    differences.sort(
        key=lambda item: (-float(item["difference_score"]), str(item["case_id"]))
    )
    return differences[:10]


def _format_report(payload: dict[str, object]) -> str:
    """生成包含指标口径和差异 Case 的 Markdown 报告。"""

    summaries = dict(payload["summaries"])
    lines = [
        "# P1-D1 BM25 Dev Ablation",
        "",
        f"- Dataset: `{payload['dataset_version']}`",
        "- Split: `dev`",
        f"- Cases: {payload['case_count']}",
        "- 相同标签、Router source filter 与 top-k；本报告不代表答案准确率。",
        "",
        "| Retriever | Recall@3 | Recall@5 | MRR | P50 ms | P95 ms |",
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
        lines.append(
            f"- `{item['case_id']}` ({item['category']}): {item['query']}；"
            f"MRR keyword/bm25/dense/bm25_hybrid="
            f"{float(metrics['keyword']['reciprocal_rank']):.3f}/"
            f"{float(metrics['bm25']['reciprocal_rank']):.3f}/"
            f"{float(metrics['dense']['reciprocal_rank']):.3f}/"
            f"{float(metrics['bm25_hybrid']['reciprocal_rank']):.3f}。"
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    raise SystemExit(main())
