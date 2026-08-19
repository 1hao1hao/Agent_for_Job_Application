from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.evaluation import load_chunks_jsonl  # noqa: E402
from intern_rag.graph import load_knowledge_graph  # noqa: E402
from intern_rag.graph.neo4j import create_neo4j_repository  # noqa: E402
from intern_rag.retrieval import load_dense_index  # noqa: E402
from intern_rag.retrieval.pgvector import (  # noqa: E402
    PgVectorIndexConfig,
    PgVectorIndexRepository,
    psycopg_connection_factory,
)


def main() -> int:
    """把 v0.3 Dense Index 与 Graph 工件写入 pgvector/Neo4j 并核对数量。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pgvector", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")
    args = parser.parse_args()
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    index, _ = load_dense_index(
        ROOT / "data/processed/indexes/evalrag_v0.3/bge-small-zh-v1.5"
    )
    graph = load_knowledge_graph(
        ROOT / "data/processed/graphs/evalrag_v0.3/job-skill-experience-v0.2.json"
    )
    report: dict[str, object] = {"dataset_version": "evalrag_v0.3"}

    if not args.skip_pgvector:
        started = perf_counter()
        repository = PgVectorIndexRepository(
            psycopg_connection_factory(os.environ["DATABASE_URL"]),
            PgVectorIndexConfig(
                dataset_version="evalrag_v0.3",
                embedding_name=index.metadata.embedding_name,
                embedding_version=index.metadata.embedding_version,
                dimensions=index.metadata.dimensions,
            ),
        )
        repository.migrate()
        repository.replace(chunks, index.vectors)
        report["pgvector"] = {
            "row_count": repository.count(),
            "dimensions": index.metadata.dimensions,
            "elapsed_ms": (perf_counter() - started) * 1000,
        }

    if not args.skip_neo4j:
        started = perf_counter()
        repository = create_neo4j_repository(
            os.environ["NEO4J_URI"],
            os.environ["NEO4J_USER"],
            os.environ["NEO4J_PASSWORD"],
        )
        repository.migrate()
        repository.replace(graph)
        restored = repository.load("evalrag_v0.3")
        repository.close()
        report["neo4j"] = {
            "node_count": len(restored.nodes),
            "edge_count": len(restored.edges),
            "path_equivalent": restored.to_dict() == graph.to_dict(),
            "elapsed_ms": (perf_counter() - started) * 1000,
        }

    output = ROOT / "reports/infrastructure/p1-d4-v03/storage_load.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
