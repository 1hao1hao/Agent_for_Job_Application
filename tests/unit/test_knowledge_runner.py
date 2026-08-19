import unittest

from intern_rag.evaluation.knowledge_dataset import KnowledgeEvaluationCase
from intern_rag.evaluation.knowledge_runner import (
    KnowledgeRunConfig,
    run_knowledge_evaluation,
)
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult


class KnowledgeRunnerTests(unittest.TestCase):
    def test_vector_candidate_with_none_path_does_not_break_run(self) -> None:
        chunk = Chunk("c1", "jd", "p", "t", "证据", {})

        def retriever(query, chunks, top_k=5, source_types=None):
            return [
                RetrievalResult(
                    "c1", 0.9, 1, chunk, "vector", {"path_valid": None}
                )
            ]

        case = KnowledgeEvaluationCase(
            "case-1",
            "问题",
            "single_source",
            "dev",
            ("jd",),
            ("c1",),
            ("证据",),
            True,
        )
        config = KnowledgeRunConfig(
            "run", "v03", "g2", "dev", "fake", 5, "test", {}
        )

        result = run_knowledge_evaluation([case], [chunk], config, retriever)

        self.assertEqual(result.summary["error_count"], 0)
        self.assertIsNone(result.case_results[0]["metrics"]["path_validity"])
        self.assertEqual(result.summary["metrics"]["recall_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
