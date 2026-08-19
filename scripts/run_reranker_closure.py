from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.evaluation import load_chunks_jsonl  # noqa: E402
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset  # noqa: E402
from intern_rag.evaluation.knowledge_runner import (  # noqa: E402
    KnowledgeRunConfig,
    run_knowledge_evaluation,
    save_knowledge_run,
)
from intern_rag.retrieval import build_retriever_from_config  # noqa: E402


VARIANTS = {
    "bge_base_k20": "configs/retrieval/adaptive_v0.3.json",
    "bge_base_k10": "configs/retrieval/adaptive_reranker_base_k10_v0.3.json",
    "minilm_k10": "configs/retrieval/adaptive_reranker_minilm_k10_v0.3.json",
}


def main() -> int:
    """在同一 v0.3/dev 上比较候选数与轻量多语 CrossEncoder。"""

    ablation_id = "p1-d7-reranker-closure-v03-dev-20260816"
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl")
    variants: dict[str, object] = {}
    case_rows: dict[str, list[dict[str, object]]] = {}
    for name, relative in VARIANTS.items():
        raw = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        run_id = f"{ablation_id}-{name}"
        run_dir = ROOT / "reports/runs" / run_id
        if (run_dir / "summary.json").exists():
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            rows = _read_jsonl(run_dir / "case_results.jsonl")
        else:
            result = run_knowledge_evaluation(
                cases,
                chunks,
                KnowledgeRunConfig(
                    run_id=run_id,
                    dataset_version="evalrag_v0.3",
                    graph_version="job-skill-experience-v0.2",
                    split="dev",
                    strategy=name,
                    top_k=5,
                    command="PYTHONPATH=src python scripts/run_reranker_closure.py",
                    retriever_config=raw,
                ),
                build_retriever_from_config(raw),
            )
            save_knowledge_run(result, run_dir)
            summary, rows = result.summary, result.case_results
        traces = [dict(dict(row["predicted"])["trace"]) for row in rows]
        reranked = [trace for trace in traces if trace.get("rerank_invoked")]
        variants[name] = {
            "config": relative,
            "model": raw["reranker_model"],
            "revision": raw["reranker_revision"],
            "candidate_k": raw["reranker_candidate_k"],
            "metrics": summary["metrics"],
            "latency_ms": summary["latency_ms"],
            "reranker_invocation_rate": len(reranked) / len(rows),
            "reranked_case_count": len(reranked),
        }
        case_rows[name] = rows
    payload = {
        "ablation_id": ablation_id,
        "dataset_version": "evalrag_v0.3",
        "split": "dev",
        "case_count": 160,
        "variants": variants,
        "differences": _differences(case_rows)[:10],
        "decision": _decision(variants),
        "boundary": "dev-only closure; frozen test was not read or run",
    }
    out = ROOT / "reports/ablations" / ablation_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _differences(all_rows: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    indexed = {name: {str(row["case_id"]): row for row in rows} for name, rows in all_rows.items()}
    first = next(iter(indexed.values()))
    output = []
    for case_id, row in first.items():
        if not row["answerable"]:
            continue
        metrics = {name: dict(values[case_id]["metrics"]) for name, values in indexed.items()}
        mrr = [float(value["reciprocal_rank"] or 0) for value in metrics.values()]
        ndcg = [float(value["ndcg_at_5"] or 0) for value in metrics.values()]
        if max(mrr) != min(mrr) or max(ndcg) != min(ndcg):
            output.append({"case_id": case_id, "query": row["query"], "metrics": metrics, "difference": max(mrr) - min(mrr) + max(ndcg) - min(ndcg)})
    return sorted(output, key=lambda item: (-float(item["difference"]), str(item["case_id"])))


def _decision(variants: dict[str, object]) -> dict[str, str]:
    base = dict(variants["bge_base_k20"])
    light = dict(variants["minilm_k10"])
    base_mrr = float(dict(base["metrics"])["mrr"])
    light_mrr = float(dict(light["metrics"])["mrr"])
    base_p95 = float(dict(base["latency_ms"])["p95"])
    light_p95 = float(dict(light["latency_ms"])["p95"])
    recall_delta = float(dict(light["metrics"])["recall_at_5"]) - float(dict(base["metrics"])["recall_at_5"])
    if light_p95 < base_p95 and light_mrr >= base_mrr - 0.001 and recall_delta >= -0.01:
        return {
            "mode": "on_demand_minilm_k10",
            "reason": "P95 明显下降，MRR 基本持平且 Recall@5 退化不超过 1 个百分点",
        }
    return {"mode": "default_off_on_demand_bge_k20", "reason": "轻量模型未满足预先声明的质量/延迟边界"}


def _report(payload: dict[str, object]) -> str:
    lines = ["# P1-D7 Reranker Closure", "", "| Variant | Recall@3 | Recall@5 | MRR | NDCG@5 | Invoke | P95 ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, raw in dict(payload["variants"]).items():
        item = dict(raw)
        metrics, latency = dict(item["metrics"]), dict(item["latency_ms"])
        lines.append(f"| {name} | {metrics['recall_at_3']:.4f} | {metrics['recall_at_5']:.4f} | {metrics['mrr']:.4f} | {metrics['ndcg_at_5']:.4f} | {float(item['reranker_invocation_rate']):.2%} | {latency['p95']:.3f} |")
    lines.extend(["", f"Decision: `{dict(payload['decision'])['mode']}`。{dict(payload['decision'])['reason']}。", "", "只使用 dev；未读取 frozen test。"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
