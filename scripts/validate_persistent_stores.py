from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.graph import GraphEdge, GraphNode, KnowledgeGraph  # noqa: E402
from intern_rag.graph.neo4j import create_neo4j_repository  # noqa: E402
from intern_rag.ingestion import Chunk  # noqa: E402
from intern_rag.retrieval.pgvector import (  # noqa: E402
    PgVectorIndexConfig,
    PgVectorIndexRepository,
    psycopg_connection_factory,
)


def main() -> int:
    """在真实 pgvector/Neo4j 中写入小工件并验证服务重启后仍可读取。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    chunks = [
        Chunk("store-c1", "jd", "fixture", "岗位", "需要 RAG", {}),
        Chunk("store-c2", "project_logs", "fixture", "项目", "实现 RAG", {}),
    ]
    vector_repository = PgVectorIndexRepository(
        psycopg_connection_factory(os.environ["DATABASE_URL"]),
        PgVectorIndexConfig(
            dataset_version="p1-d4-store-fixture",
            embedding_name="fixture",
            embedding_version="v1",
            dimensions=2,
            table_name="rag_chunk_embeddings_ci",
        ),
    )
    graph_repository = create_neo4j_repository(
        os.environ["NEO4J_URI"],
        os.environ["NEO4J_USER"],
        os.environ["NEO4J_PASSWORD"],
    )
    if not args.verify_only:
        vector_repository.migrate()
        vector_repository.replace(chunks, [[1.0, 0.0], [0.8, 0.2]])
        graph_repository.migrate()
        graph_repository.replace(_fixture_graph())

    vector_rows = vector_repository.query([1.0, 0.0], top_k=2)
    graph = graph_repository.load("p1-d4-store-fixture")
    graph_repository.close()
    report = {
        "mode": "verify_only" if args.verify_only else "load_and_verify",
        "pgvector": {
            "count": vector_repository.count(),
            "query_ids": [item[0] for item in vector_rows],
        },
        "neo4j": {"node_count": len(graph.nodes), "edge_count": len(graph.edges)},
    }
    assert report["pgvector"]["count"] == 2
    assert report["pgvector"]["query_ids"][0] == "store-c1"
    assert report["neo4j"] == {"node_count": 2, "edge_count": 1}
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _fixture_graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        version="fixture-v1",
        dataset_version="p1-d4-store-fixture",
        config_hash="fixture",
        nodes=(
            GraphNode("job-1", "job", "RAG 岗位", chunk_ids=("store-c1",)),
            GraphNode("skill-1", "skill", "RAG", chunk_ids=("store-c2",)),
        ),
        edges=(
            GraphEdge("edge-1", "requires", "job-1", "skill-1", ("store-c1",)),
        ),
        stats={"node_count": 2, "edge_count": 1},
    )


if __name__ == "__main__":
    raise SystemExit(main())
