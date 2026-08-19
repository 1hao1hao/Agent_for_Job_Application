from __future__ import annotations

import json
from pathlib import Path


BASELINE = Path("reports/runs/p0-d5-v02-dev-20260804-hybrid-baseline")
RERANK = Path("reports/runs/p1-cross-encoder-v02-dev-formal-20260811")
OUTPUT = Path("reports/ablations/p1-cross-encoder-v02-dev-20260811")


def main() -> int:
    """生成 Hybrid 与真实 BGE CrossEncoder 的 dev 单变量对照。"""

    baseline_summary = _read_json(BASELINE / "summary.json")
    rerank_summary = _read_json(RERANK / "summary.json")
    baseline_cases = _index(_read_jsonl(BASELINE / "case_results.jsonl"))
    rerank_cases = _index(_read_jsonl(RERANK / "case_results.jsonl"))
    differences = []
    for case_id, baseline in baseline_cases.items():
        if not baseline["answerable"]:
            continue
        before = dict(baseline["metrics"])
        after = dict(rerank_cases[case_id]["metrics"])
        delta_mrr = float(after["reciprocal_rank"]) - float(before["reciprocal_rank"])
        delta_recall = float(after["recall_at_5"]) - float(before["recall_at_5"])
        if delta_mrr == 0 and delta_recall == 0:
            continue
        differences.append(
            {
                "case_id": case_id,
                "category": baseline["category"],
                "query": baseline["query"],
                "delta_mrr": delta_mrr,
                "delta_recall_at_5": delta_recall,
                "baseline_top5": _top_ids(baseline),
                "rerank_top5": _top_ids(rerank_cases[case_id]),
                "outcome": "improved" if delta_mrr + delta_recall > 0 else "regressed",
            }
        )
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
        "single_variable": "BGE CrossEncoder reranking over Hybrid top-20",
        "model": "BAAI/bge-reranker-base",
        "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
        "baseline_run": str(BASELINE),
        "rerank_run": str(RERANK),
        "baseline": baseline_summary,
        "rerank": rerank_summary,
        "differences": differences[:10],
        "decision": {
            "reranker_enabled": False,
            "reason": "Recall@3, Recall@5 and MRR regressed while CPU P95 increased.",
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
    """把指标、差异 Case 和冻结决策渲染为 Markdown。"""

    baseline = dict(dict(payload["baseline"])["metrics"])
    rerank = dict(dict(payload["rerank"])["metrics"])
    baseline_latency = dict(dict(dict(payload["baseline"])["latency_ms"])["retrieval"])
    rerank_latency = dict(dict(dict(payload["rerank"])["latency_ms"])["retrieval"])
    lines = [
        "# BGE CrossEncoder Reranker Dev Ablation",
        "",
        "- Dataset: `evalrag_v0.2`; split: `dev`; 80 cases。",
        "- 唯一变量：对相同 Hybrid top-20 使用 BGE CrossEncoder 重排。",
        f"- Model: `{payload['model']}`; revision: `{payload['revision']}`。",
        "",
        "| Strategy | Recall@3 | Recall@5 | MRR | Retrieval P95 ms |",
        "|---|---:|---:|---:|---:|",
        f"| Hybrid | {baseline['recall_at_3']:.4f} | {baseline['recall_at_5']:.4f} | {baseline['mrr']:.4f} | {baseline_latency['p95']:.3f} |",
        f"| Hybrid + BGE CrossEncoder | {rerank['recall_at_3']:.4f} | {rerank['recall_at_5']:.4f} | {rerank['mrr']:.4f} | {rerank_latency['p95']:.3f} |",
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
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "当前默认仍关闭 Reranker。通用模型把同主题近似段落排在 benchmark 指定的精确证据前，",
            "同时 CPU P95 显著上升；本结果不修改 frozen test，也不包装成质量提升。",
            "",
        ]
    )
    return "\n".join(lines)


def _top_ids(case: dict[str, object]) -> list[str]:
    retrieved = dict(case["predicted"])["retrieved"]
    return [str(item["chunk_id"]) for item in retrieved[:5]]  # type: ignore[index]


def _index(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(record["case_id"]): record for record in records}


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
