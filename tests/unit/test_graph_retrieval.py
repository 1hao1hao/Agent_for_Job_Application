import tempfile
import unittest
from pathlib import Path

from intern_rag.graph import (
    GraphEdge,
    GraphExtraction,
    GraphNode,
    KnowledgeGraph,
    build_knowledge_graph,
    load_knowledge_graph,
    save_knowledge_graph,
)
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import (
    AdaptiveRetriever,
    FakeRerankScorer,
    GraphRetriever,
    GraphVectorRetriever,
    RetrievalResult,
)


def _chunk(chunk_id: str, source_type: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_type=source_type,
        source_path=f"data/raw/{source_type}/{chunk_id}.md",
        title=chunk_id,
        text=text,
        metadata={"source_type": source_type},
    )


def _graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        version="test-v1",
        dataset_version="test-data",
        config_hash="abc",
        nodes=(
            GraphNode("job", "job", "Python 岗位", (), ("jd-1",)),
            GraphNode("skill", "skill", "Python", ("python",), ("jd-1", "resume-1")),
            GraphNode("project", "project", "接口项目", (), ("project-1",)),
        ),
        edges=(
            GraphEdge("e1", "requires", "job", "skill", ("jd-1",)),
            GraphEdge("e2", "demonstrates", "project", "skill", ("project-1",)),
        ),
    )


class FakeExtractor:
    def extract(self, chunk: Chunk) -> GraphExtraction:
        return GraphExtraction(
            nodes=(GraphNode("skill", "skill", "Python", (), (chunk.id,)),),
            edges=(),
        )


class FixedRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls = 0

    def __call__(self, query, chunks, top_k=5, source_types=None):
        del query, chunks
        self.calls += 1
        return [
            item
            for item in self.results
            if source_types is None or item.chunk.source_type in source_types
        ][:top_k]


class GraphRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            _chunk("jd-1", "jd", "Python 岗位要求"),
            _chunk("resume-1", "resume", "Python 经历"),
            _chunk("project-1", "project_logs", "接口项目使用 Python"),
        ]

    def test_graph_build_merges_chunk_ids_and_round_trips(self) -> None:
        graph = build_knowledge_graph(
            self.chunks[:2],
            FakeExtractor(),
            version="v1",
            dataset_version="data-v1",
            config_hash="hash",
        )
        self.assertEqual(graph.nodes[0].chunk_ids, ("jd-1", "resume-1"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "graph.json"
            save_knowledge_graph(graph, path)
            loaded = load_knowledge_graph(path)
        self.assertEqual(loaded, graph)

    def test_graph_retriever_returns_valid_paths_and_applies_source_filter(self) -> None:
        retriever = GraphRetriever(_graph(), max_hops=2)

        results = retriever(
            "Python 岗位要求与哪个项目对应起来",
            self.chunks,
            top_k=5,
            source_types={"jd", "project_logs"},
        )

        self.assertEqual({item.chunk.source_type for item in results}, {"jd", "project_logs"})
        self.assertTrue(all(item.details["path_valid"] == 1 for item in results))
        self.assertEqual(retriever.get_last_trace()["graph_version"], "test-v1")

    def test_graph_vector_deduplicates_and_falls_back_without_entity(self) -> None:
        vector_results = [
            RetrievalResult("jd-1", 0.9, 1, self.chunks[0]),
            RetrievalResult("resume-1", 0.8, 2, self.chunks[1]),
        ]
        vector = FixedRetriever(vector_results)
        retriever = GraphVectorRetriever(GraphRetriever(_graph()), vector)

        fused = retriever("Python 岗位和经历是否匹配", self.chunks, top_k=5)
        fallback = retriever("完全没有实体的查询", self.chunks, top_k=2)

        self.assertEqual(len({item.chunk_id for item in fused}), len(fused))
        self.assertEqual([item.chunk_id for item in fallback], ["jd-1", "resume-1"])
        self.assertTrue(retriever.get_last_trace()["vector_fallback"])

    def test_adaptive_selects_graph_only_for_cross_document_query(self) -> None:
        base_results = [RetrievalResult("jd-1", 0.9, 1, self.chunks[0])]
        base = FixedRetriever(base_results)
        graph_vector = GraphVectorRetriever(GraphRetriever(_graph()), base)
        adaptive = AdaptiveRetriever(
            {"bm25": base, "dense": base, "hybrid": base},
            FakeRerankScorer({"Python 岗位要求": 1.0}),
            graph_retriever=graph_vector,
        )

        adaptive(
            "Python 岗位要求与候选人的哪个项目对应起来",
            self.chunks,
            source_types={"jd", "project_logs"},
        )
        graph_trace = adaptive.get_last_trace()
        adaptive("Python 岗位要求", self.chunks, source_types={"jd"})
        vector_trace = adaptive.get_last_trace()

        self.assertEqual(graph_trace["strategy"], "graph_hybrid")
        self.assertIn("decomposition", graph_trace)
        self.assertTrue(graph_trace["decomposition"]["is_cross_document"])
        self.assertNotEqual(vector_trace["strategy"], "graph_hybrid")

    def test_confirmed_skill_project_phrase_stays_graph_routed(self) -> None:
        """回归 P1-D3 首轮 dev 中“共同体现”未触发图检索的问题。"""

        base = FixedRetriever(
            [RetrievalResult("resume-1", 0.9, 1, self.chunks[1])]
        )
        adaptive = AdaptiveRetriever(
            {"bm25": base, "dense": base, "hybrid": base},
            FakeRerankScorer({"Python 经历": 1.0}),
            graph_retriever=GraphVectorRetriever(GraphRetriever(_graph()), base),
        )

        adaptive(
            "候选人的简历经历和项目记录如何共同体现处理错误的能力？",
            self.chunks,
            source_types={"resume", "project_logs"},
        )

        self.assertEqual(adaptive.get_last_trace()["strategy"], "graph_hybrid")


if __name__ == "__main__":
    unittest.main()
