import unittest

from intern_rag.graph import GraphEdge, GraphNode, KnowledgeGraph
from intern_rag.graph.neo4j import Neo4jGraphRepository
from intern_rag.ingestion import Chunk
from intern_rag.retrieval.pgvector import (
    PgVectorIndexConfig,
    PgVectorIndexRepository,
    PgVectorRetriever,
)


class FakeCursor:
    def __init__(self, rows=None) -> None:
        self.rows = rows or []
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def executemany(self, query, params):
        self.executed.append((query, list(params)))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0]


class FakeConnection:
    def __init__(self, cursor) -> None:
        self.value = cursor
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def cursor(self):
        return self.value

    def commit(self):
        self.committed = True


class FakeEmbedder:
    name = "fake"
    version = "v1"

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class PersistentRetrievalTests(unittest.TestCase):
    def test_pgvector_retriever_restores_chunks_and_filter(self) -> None:
        cursor = FakeCursor([("c1", 0.91)])
        repository = PgVectorIndexRepository(
            lambda: FakeConnection(cursor),
            PgVectorIndexConfig("v03", "fake", "v1", 2),
        )
        retriever = PgVectorRetriever(repository, FakeEmbedder())
        chunk = Chunk("c1", "jd", "p", "t", "text", {})

        results = retriever("query", [chunk], top_k=1, source_types={"jd"})

        self.assertEqual(results[0].chunk_id, "c1")
        self.assertEqual(results[0].rank, 1)
        query, params = cursor.executed[-1]
        self.assertIn("source_type = ANY", query)
        self.assertIn(["jd"], params)

    def test_pgvector_replace_checks_dimensions(self) -> None:
        repository = PgVectorIndexRepository(
            lambda: FakeConnection(FakeCursor()),
            PgVectorIndexConfig("v03", "fake", "v1", 2),
        )
        chunk = Chunk("c1", "jd", "p", "t", "text", {})
        with self.assertRaises(ValueError):
            repository.replace([chunk], [[1.0]])

    def test_neo4j_replace_sends_versioned_nodes_and_edges(self) -> None:
        calls = []

        class Tx:
            def run(self, query, **params):
                calls.append((query, params))

        graph = KnowledgeGraph(
            "g2",
            "v03",
            "hash",
            (GraphNode("n1", "skill", "RAG", chunk_ids=("c1",), provenance=("u",)),),
            (GraphEdge("e1", "related_to", "n1", "n1", ("c1",), ("u",)),),
            {"node_count": 1},
        )

        Neo4jGraphRepository._replace_tx(
            Tx(),
            graph,
            [{"node_id": "n1"}],
            [{"edge_id": "e1", "source_node_id": "n1", "target_node_id": "n1"}],
        )

        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][1]["dataset_version"], "v03")
        self.assertIn("EVIDENCE_RELATION", calls[2][0])


if __name__ == "__main__":
    unittest.main()
