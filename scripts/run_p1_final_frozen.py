from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.evaluation import (  # noqa: E402
    GraphRunConfig,
    load_chunks_jsonl,
    load_graph_challenge,
    run_graph_evaluation,
    save_graph_run,
)
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset  # noqa: E402
from intern_rag.evaluation.knowledge_runner import (  # noqa: E402
    KnowledgeRunConfig,
    run_knowledge_evaluation,
    save_knowledge_run,
)
from intern_rag.retrieval import build_retriever_from_config  # noqa: E402


RELEASE_ID = "p1-d7-v03-frozen-20260816"


def main() -> int:
    """校验冻结清单后，一次性运行 general 与 graph frozen test。"""

    manifest_path = ROOT / "configs/final/p1_v0.3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_inputs(dict(manifest["frozen_inputs"]))
    release_dir = ROOT / "reports/releases" / RELEASE_ID
    if release_dir.exists():
        raise FileExistsError(f"frozen release already exists: {release_dir}")
    release_dir.mkdir(parents=True)

    chunks_v03 = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases_v03 = load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl")
    general_summaries = {}
    for name, path in {
        "bm25_baseline": "configs/retrieval/bm25_v0.3.json",
        "p1_final_graph_vector": "configs/retrieval/graph_adaptive_final_v0.3.json",
    }.items():
        raw = json.loads((ROOT / path).read_text(encoding="utf-8"))
        run_id = f"{RELEASE_ID}-{name}"
        result = run_knowledge_evaluation(
            cases_v03,
            chunks_v03,
            KnowledgeRunConfig(
                run_id=run_id,
                dataset_version="evalrag_v0.3",
                graph_version="job-skill-experience-v0.2",
                split="test",
                strategy=name,
                top_k=5,
                command="PYTHONPATH=src python scripts/run_p1_final_frozen.py",
                retriever_config=raw,
            ),
            build_retriever_from_config(raw),
        )
        run_dir = ROOT / "reports/runs" / run_id
        save_knowledge_run(result, run_dir)
        general_summaries[name] = result.summary

    graph_config_path = ROOT / "configs/retrieval/graph_adaptive_v0.2.json"
    graph_raw = json.loads(graph_config_path.read_text(encoding="utf-8"))
    graph_result = run_graph_evaluation(
        load_graph_challenge(ROOT / "data/evaluation/evalrag_graph_v0.1.jsonl"),
        load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.2.jsonl"),
        GraphRunConfig(
            run_id=f"{RELEASE_ID}-graph-challenge",
            dataset_version="evalrag_graph_v0.1",
            graph_version="job-skill-experience-v0.1",
            split="test",
            retriever_name="graph_adaptive",
            top_k=5,
            command="PYTHONPATH=src python scripts/run_p1_final_frozen.py",
            retriever_config=graph_raw,
        ),
        build_retriever_from_config(graph_raw),
    )
    save_graph_run(graph_result, ROOT / "reports/runs" / f"{RELEASE_ID}-graph-challenge")
    summary = {
        "release_id": RELEASE_ID,
        "frozen_at": manifest["frozen_at"],
        "general_retrieval": general_summaries,
        "graph_challenge": graph_result.summary,
        "multi_turn": {
            "status": "not_frozen",
            "reason": "evalrag_context_v0.1 contains only 60 dev cases; no untouched test split exists",
            "dev_reference": "reports/ablations/p1-d5-context-memory-v01-dev-20260816-bge/summary.json",
        },
        "persistent_backends": {
            "status": "configuration_and_fake_repository_verified",
            "real_docker_restart": "not_executed_on_campus_server_without Docker Engine",
        },
        "policy": "test predictions are immutable; failures are retained and this dataset version will not be retuned",
    }
    (release_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (release_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _verify_inputs(frozen_inputs: dict[str, object]) -> None:
    for relative, expected in frozen_inputs.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen input changed: {relative}")


def _report(summary: dict[str, object]) -> str:
    general = dict(summary["general_retrieval"])
    lines = ["# P1 Final Frozen Release", "", "| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P95 ms |", "|---|---:|---:|---:|---:|---:|"]
    for name, raw in general.items():
        item, metrics = dict(raw), dict(dict(raw)["metrics"])
        latency = dict(item["latency_ms"])
        lines.append(f"| {name} | {metrics['recall_at_3']:.4f} | {metrics['recall_at_5']:.4f} | {metrics['mrr']:.4f} | {metrics['ndcg_at_5']:.4f} | {latency['p95']:.3f} |")
    graph = dict(summary["graph_challenge"])
    lines.extend(["", "## Graph Challenge", "", f"10 frozen cases; metrics: `{json.dumps(graph['metrics'], ensure_ascii=False)}`。", "", "## Boundary", "", "Context benchmark has no untouched test split, so it remains a dev-only reference rather than a fabricated frozen result."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
