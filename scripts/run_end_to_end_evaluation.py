from __future__ import annotations

import json
from pathlib import Path
import subprocess

from intern_rag.agent import EvidenceConfig
from intern_rag.evaluation import (
    EndToEndRunConfig,
    load_chunks_jsonl,
    load_evaluation_dataset,
    run_extractive_end_to_end_evaluation,
    save_end_to_end_artifacts,
)
from intern_rag.retrieval import build_retriever_from_config
from intern_rag.routing import build_router_from_config


RUN_ID = "p0-d5-v02-frozen-test-20260804-extractive-e2e"


def main() -> int:
    """在冻结最终配置上运行 deterministic extractive 端到端 baseline。"""

    final = _read_json(Path("configs/final/p0_v0.2.json"))
    router_config = _read_json(Path(str(final["router_config"])))
    retriever_config = _read_json(Path(str(final["retriever_config"])))
    evidence_data = _read_json(Path(str(final["evidence_config"])))
    cases = load_evaluation_dataset(Path("data/evaluation/evalrag_v0.2.jsonl"))
    chunks = load_chunks_jsonl(Path("data/processed/chunks/evalrag_v0.2.jsonl"))
    config = EndToEndRunConfig(
        run_id=RUN_ID,
        dataset_version="evalrag_v0.2",
        split="test",
        router_name=str(router_config["router_name"]),
        retriever_name=str(retriever_config["retriever_name"]),
        top_k=int(retriever_config["top_k"]),
        context_max_chars=4000,
        key_point_threshold=0.5,
        git_revision=_git_revision(),
        command="PYTHONPATH=src python scripts/run_end_to_end_evaluation.py",
    )
    evidence = EvidenceConfig(
        min_results=int(evidence_data["min_results"]),
        min_scores=dict(evidence_data["min_scores"]),
        require_source_coverage=bool(evidence_data["require_source_coverage"]),
    )
    result = run_extractive_end_to_end_evaluation(
        cases,
        chunks,
        config,
        build_router_from_config(router_config),
        build_retriever_from_config(retriever_config),
        evidence_config=evidence,
    )
    run_dir = Path("reports/runs") / RUN_ID
    if run_dir.exists():
        print(f"ERROR: end-to-end frozen run already exists: {run_dir}")
        return 2
    save_end_to_end_artifacts(result, run_dir)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _git_revision() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return f"{commit}-dirty" if dirty else commit


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
