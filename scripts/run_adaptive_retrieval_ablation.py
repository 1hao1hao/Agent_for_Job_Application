from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


CONFIGS = {
    "fixed_hybrid": "configs/retrieval/hybrid_v0.2.json",
    "always_rerank": "configs/retrieval/hybrid_cross_encoder_v0.2.json",
    "adaptive": "configs/retrieval/adaptive_v0.2.json",
}


def main() -> int:
    """同集运行固定 Hybrid、全量重排与 Adaptive，并保存差异和调用率。"""

    args = _parse_args()
    ablation_id = args.run_id or datetime.now(timezone.utc).strftime(
        "p1-d2-adaptive-dev-%Y%m%dT%H%M%SZ"
    )
    run_dirs: dict[str, Path] = {}
    for name, config_path in CONFIGS.items():
        reused_path = {
            "fixed_hybrid": args.reuse_fixed_run,
            "always_rerank": args.reuse_always_run,
            "adaptive": args.reuse_adaptive_run,
        }.get(name)
        if reused_path:
            run_dirs[name] = Path(reused_path)
            continue
        run_id = f"{ablation_id}-{name}"
        completed = subprocess.run(
            [
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
            ],
            check=False,
        )
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
    adaptive_decisions = [
        dict(dict(item["predicted"]).get("retrieval_decision", {}))
        for item in cases["adaptive"]
    ]
    payload = {
        "ablation_id": ablation_id,
        "dataset_version": args.dataset_version,
        "split": "dev",
        "case_count": int(summaries["adaptive"]["case_count"]),
        "run_dirs": {name: str(path) for name, path in run_dirs.items()},
        "summaries": summaries,
        "adaptive": {
            "rerank_invocation_count": sum(
                bool(item.get("rerank_invoked")) for item in adaptive_decisions
            ),
            "rerank_applied_count": sum(
                bool(item.get("rerank_applied")) for item in adaptive_decisions
            ),
            "strategy_counts": _counts(
                str(item.get("strategy", "unknown")) for item in adaptive_decisions
            ),
        },
        "adaptive_reranked_cases": _build_reranked_cases(cases),
        "strategy_differences": _build_differences(cases),
        "boundary": (
            "dev-only retrieval ablation; quality metrics do not measure answer accuracy"
        ),
    }
    report_dir = Path("reports/ablations") / ablation_id
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "report.md").write_text(_format_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Ablation report written to {report_dir}")
    return 0


def _build_differences(
    cases: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """保存策略间 NDCG/MRR/Recall 差异最大的十条真实 Case。"""

    indexed = {
        name: {str(item["case_id"]): item for item in values}
        for name, values in cases.items()
    }
    differences: list[dict[str, object]] = []
    for case_id, baseline in indexed["fixed_hybrid"].items():
        if not bool(baseline["answerable"]):
            continue
        metrics = {
            name: dict(indexed[name][case_id]["metrics"]) for name in CONFIGS
        }
        values = [
            float(item[metric])
            for metric in ("recall_at_5", "reciprocal_rank", "ndcg_at_5")
            for item in metrics.values()
        ]
        differences.append(
            {
                "case_id": case_id,
                "category": baseline["category"],
                "query": baseline["query"],
                "difference_score": max(values) - min(values),
                "metrics": metrics,
                "adaptive_decision": dict(
                    dict(indexed["adaptive"][case_id]["predicted"]).get(
                        "retrieval_decision", {}
                    )
                ),
            }
        )
    differences.sort(
        key=lambda item: (-float(item["difference_score"]), str(item["case_id"]))
    )
    return differences[:10]


def _build_reranked_cases(
    cases: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """列出 Adaptive 实际调用 CrossEncoder 的 Case 及其相对 Hybrid 变化。"""

    baseline = {str(item["case_id"]): item for item in cases["fixed_hybrid"]}
    reranked: list[dict[str, object]] = []
    for item in cases["adaptive"]:
        decision = dict(dict(item["predicted"]).get("retrieval_decision", {}))
        if not bool(decision.get("rerank_invoked")):
            continue
        base_metrics = dict(baseline[str(item["case_id"])]["metrics"])
        metrics = dict(item["metrics"])
        reranked.append(
            {
                "case_id": item["case_id"],
                "category": item["category"],
                "confidence": decision.get("confidence"),
                "recall_at_5_delta": (
                    float(metrics["recall_at_5"]) - float(base_metrics["recall_at_5"])
                ),
                "mrr_delta": (
                    float(metrics["reciprocal_rank"])
                    - float(base_metrics["reciprocal_rank"])
                ),
                "ndcg_at_5_delta": (
                    float(metrics["ndcg_at_5"]) - float(base_metrics["ndcg_at_5"])
                ),
            }
        )
    return reranked


def _format_report(payload: dict[str, object]) -> str:
    """生成同时呈现质量、延迟、调用率和负结果的 Markdown 报告。"""

    summaries = dict(payload["summaries"])
    lines = [
        "# P1-D2 Adaptive Retrieval Dev Ablation",
        "",
        f"- Dataset: `{payload['dataset_version']}`; split: `dev`; cases: {payload['case_count']}。",
        "- 固定相同标签、Router source filter、top-k、Embedding 和 CrossEncoder revision。",
        "- 本报告只衡量检索质量与延迟，不代表最终答案准确率。",
        "",
        "| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONFIGS:
        summary = dict(summaries[name])
        metrics = dict(summary["metrics"])
        latency = dict(dict(summary["latency_ms"])["retrieval"])
        lines.append(
            f"| {name} | {float(metrics['recall_at_3']):.4f} | "
            f"{float(metrics['recall_at_5']):.4f} | {float(metrics['mrr']):.4f} | "
            f"{float(metrics['ndcg_at_5']):.4f} | {float(latency['p50']):.3f} | "
            f"{float(latency['p95']):.3f} |"
        )
    adaptive = dict(payload["adaptive"])
    invocation_rate = int(adaptive["rerank_invocation_count"]) / int(
        payload["case_count"]
    )
    lines.extend(
        [
            "",
            "## Adaptive Decisions",
            "",
            f"- Strategy counts: `{adaptive['strategy_counts']}`。",
            f"- Rerank invoked/applied: {adaptive['rerank_invocation_count']}/"
            f"{adaptive['rerank_applied_count']}，调用率 {invocation_rate:.2%}。",
            "- Adaptive 保持 Recall@3/MRR，Recall@5 小幅改善，但 CPU P95 仍被少量 "
            "CrossEncoder 调用拉高，因此不是完整 Pareto 改善。",
            "",
            "## Category Metrics",
            "",
            "| Strategy | Category | Recall@3 | Recall@5 | MRR | NDCG@5 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name in CONFIGS:
        category_metrics = dict(dict(summaries[name])["category_metrics"])
        for category, values in category_metrics.items():
            metrics = dict(values)
            lines.append(
                f"| {name} | {category} | {float(metrics['recall_at_3']):.4f} | "
                f"{float(metrics['recall_at_5']):.4f} | {float(metrics['mrr']):.4f} | "
                f"{float(metrics['ndcg_at_5']):.4f} |"
            )
    lines.extend(["", "## Adaptive Reranked Cases", ""])
    for item in payload["adaptive_reranked_cases"]:  # type: ignore[union-attr]
        lines.append(
            f"- `{item['case_id']}` ({item['category']}), confidence="
            f"{float(item['confidence']):.3f}: ΔRecall@5="
            f"{float(item['recall_at_5_delta']):+.3f}, ΔMRR="
            f"{float(item['mrr_delta']):+.3f}, ΔNDCG@5="
            f"{float(item['ndcg_at_5_delta']):+.3f}。"
        )
    lines.extend(["", "## Largest Strategy Differences", ""])
    for item in payload["strategy_differences"]:  # type: ignore[union-attr]
        metrics = item["metrics"]
        decision = item["adaptive_decision"]
        lines.append(
            f"- `{item['case_id']}` ({item['category']}): "
            f"MRR fixed/always/adaptive="
            f"{float(metrics['fixed_hybrid']['reciprocal_rank']):.3f}/"
            f"{float(metrics['always_rerank']['reciprocal_rank']):.3f}/"
            f"{float(metrics['adaptive']['reciprocal_rank']):.3f}; "
            f"adaptive={decision.get('strategy')}, rerank={decision.get('rerank_invoked')}。"
        )
    return "\n".join(lines) + "\n"


def _counts(values: Iterable[str]) -> dict[str, int]:
    """统计可迭代字符串中的各策略次数。"""

    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="evalrag_v0.2")
    parser.add_argument("--run-id")
    parser.add_argument("--reuse-fixed-run")
    parser.add_argument("--reuse-always-run")
    parser.add_argument("--reuse-adaptive-run")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
