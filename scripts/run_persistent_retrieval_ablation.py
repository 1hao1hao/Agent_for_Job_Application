from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
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


def main() -> int:
    """在同一 v0.3/dev 上比较文件精确扫描、pgvector exact 与 HNSW。

    输入是已经由 ``load_v03_stores.py`` 写入 pgvector 的 BGE 向量、v0.3 Chunk 和
    dev EvaluationCase。脚本只切换存储/查询后端，依次运行真实 Retriever，保存每个
    Case 的 prediction、延迟与 Recall/MRR，并汇总索引体积。数据库不可用时直接失败，
    不使用内存结果伪装持久化实验。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is required; load pgvector before this ablation")

    ablation_id = args.run_id or datetime.now(timezone.utc).strftime(
        "p1-d4-persistent-v03-dev-%Y%m%dT%H%M%SZ"
    )
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl")
    base = {
        "dataset_version": "evalrag_v0.3",
        "top_k": 5,
        "index_dir": "data/processed/indexes/evalrag_v0.3/bge-small-zh-v1.5",
        "pgvector_table": "rag_chunk_embeddings",
    }
    configs = {
        "file_exact": {**base, "retriever_name": "dense"},
        "pgvector_exact": {**base, "retriever_name": "pgvector_exact"},
        "pgvector_hnsw": {**base, "retriever_name": "pgvector_dense"},
    }
    summaries: dict[str, object] = {}
    for strategy, config in configs.items():
        run_id = f"{ablation_id}-{strategy}"
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
                    "PYTHONPATH=src python scripts/run_persistent_retrieval_ablation.py "
                    f"--run-id {ablation_id}"
                ),
                retriever_config=config,
            ),
            build_retriever_from_config(config),
        )
        save_knowledge_run(result, ROOT / "reports/runs" / run_id)
        summaries[strategy] = result.summary

    index_dir = ROOT / str(base["index_dir"])
    payload = {
        "ablation_id": ablation_id,
        "dataset_version": "evalrag_v0.3",
        "split": "dev",
        "case_count": 160,
        "summaries": summaries,
        "file_index_bytes": sum(
            path.stat().st_size for path in index_dir.rglob("*") if path.is_file()
        ),
        "boundary": (
            "dev-only storage backend comparison; frozen test is not run; "
            "database table size must be collected from PostgreSQL separately"
        ),
    }
    output = ROOT / "reports/ablations" / ablation_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
