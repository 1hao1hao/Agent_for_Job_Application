from __future__ import annotations

import json
from pathlib import Path


BASELINE = Path("reports/runs/p0-d5-v02-dev-20260804-hybrid-baseline")
RERANK = Path("reports/runs/p0-d5-v02-dev-20260804-hybrid-rerank")
OUTPUT = Path("reports/ablations/p0-d5-reranker-dev-20260804")


def main() -> int:
    """生成 Hybrid 与 Hybrid+Reranker 的 dev 单变量对照报告。"""

    baseline_summary = _read_json(BASELINE / "summary.json")
    rerank_summary = _read_json(RERANK / "summary.json")
    baseline_cases = _index(_read_jsonl(BASELINE / "case_results.jsonl"))
    rerank_cases = _index(_read_jsonl(RERANK / "case_results.jsonl"))
    differences = []
    for case_id, baseline in baseline_cases.items():
        if not baseline["answerable"]:
            continue
        baseline_metrics = dict(baseline["metrics"])
        rerank_metrics = dict(rerank_cases[case_id]["metrics"])
        delta_mrr = float(rerank_metrics["reciprocal_rank"]) - float(
            baseline_metrics["reciprocal_rank"]
        )
        delta_recall = float(rerank_metrics["recall_at_5"]) - float(
            baseline_metrics["recall_at_5"]
        )
        if delta_mrr == 0 and delta_recall == 0:
            continue
        differences.append({
            "case_id": case_id,
            "category": baseline["category"],
            "query": baseline["query"],
            "delta_mrr": delta_mrr,
            "delta_recall_at_5": delta_recall,
            "baseline_top5": _top_ids(baseline),
            "rerank_top5": _top_ids(rerank_cases[case_id]),
            "outcome": "improved" if delta_mrr + delta_recall > 0 else "regressed",
        })
    differences.sort(
        key=lambda item: (
            -abs(float(item["delta_mrr"]) + float(item["delta_recall_at_5"])),
            str(item["case_id"]),
        )
    )
    payload = {
        "dataset_version": "evalrag_v0.2",
        "split": "dev",
        "case_count": 80,
        "baseline_run": str(BASELINE),
        "rerank_run": str(RERANK),
        "baseline": baseline_summary,
        "rerank": rerank_summary,
        "differences": differences[:10],
        "decision": {
            "reranker_enabled": False,
            "reason": "Recall@3, Recall@5 and MRR regressed while P95 increased.",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "report.md").write_text(_format_report(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))
    print(f"difference cases: {len(payload['differences'])}")
    return 0


def _format_report(payload: dict[str, object]) -> str:
    baseline = dict(dict(payload["baseline"])["metrics"])
    rerank = dict(dict(payload["rerank"])["metrics"])
    baseline_latency = dict(dict(dict(payload["baseline"])["latency_ms"])["retrieval"])
    rerank_latency = dict(dict(dict(payload["rerank"])["latency_ms"])["retrieval"])
    lines = [
        "# P0-D5 Reranker Dev Ablation",
        "",
        "- Dataset: `evalrag_v0.2`; split: `dev`; 80 cases。",
        "- 唯一主要变量：是否对 Hybrid top-20 使用中文 token-overlap 重排。",
        "- CrossEncoder adapter 已实现，但外部权重下载未完成，本报告不是神经 Reranker 结果。",
        "",
        "| Strategy | Recall@3 | Recall@5 | MRR | Retrieval P95 ms |",
        "|---|---:|---:|---:|---:|",
        f"| Hybrid | {baseline['recall_at_3']:.4f} | {baseline['recall_at_5']:.4f} | {baseline['mrr']:.4f} | {baseline_latency['p95']:.3f} |",
        f"| Hybrid + token rerank | {rerank['recall_at_3']:.4f} | {rerank['recall_at_5']:.4f} | {rerank['mrr']:.4f} | {rerank_latency['p95']:.3f} |",
        "",
        "## Difference Cases",
        "",
    ]
    for item in payload["differences"]:  # type: ignore[union-attr]
        lines.append(
            f"- `{item['case_id']}` {item['outcome']}: "
            f"ΔMRR={item['delta_mrr']:+.3f}, "
            f"ΔRecall@5={item['delta_recall_at_5']:+.3f}。"
        )
    lines.extend([
        "",
        "## Frozen Decision",
        "",
        "最终配置禁用该 Reranker。它只能重排已有候选，且本次 dev 同时损害召回、首位排序和延迟。",
    ])
    return "\n".join(lines) + "\n"


def _top_ids(case: dict[str, object]) -> list[str]:
    retrieved = dict(case["predicted"])["retrieved"]
    return [str(item["chunk_id"]) for item in retrieved[:5]]  # type: ignore[index]


def _index(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(record["case_id"]): record for record in records}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    raise SystemExit(main())
