import unittest
from pathlib import Path

from intern_rag.evaluation import (
    GraphRunConfig,
    load_chunks_jsonl,
    load_graph_challenge,
    run_graph_evaluation,
    validate_graph_challenge,
)
from intern_rag.retrieval import build_retriever_from_config


class GraphChallengeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = load_chunks_jsonl(
            Path("data/processed/chunks/evalrag_v0.2.jsonl")
        )
        cls.cases = load_graph_challenge(
            Path("data/evaluation/evalrag_graph_v0.1.jsonl")
        )

    def test_dataset_has_grounded_30_dev_10_frozen_cases(self) -> None:
        validation = validate_graph_challenge(
            self.cases, {chunk.id for chunk in self.chunks}
        )

        self.assertTrue(validation.is_valid, validation.errors)
        self.assertEqual(validation.split_counts, {"dev": 30, "test": 10})
        self.assertEqual(validation.reviewed_count, 40)

    def test_graph_only_runner_produces_real_predictions(self) -> None:
        config_data = {
            "retriever_name": "graph",
            "graph_index_path": (
                "data/processed/graphs/evalrag_v0.2/"
                "job-skill-experience-v0.1.json"
            ),
            "graph_max_hops": 2,
            "graph_max_nodes": 80,
            "graph_timeout_ms": 50.0,
        }
        result = run_graph_evaluation(
            self.cases,
            self.chunks,
            GraphRunConfig(
                run_id="graph-integration",
                dataset_version="evalrag_graph_v0.1",
                graph_version="job-skill-experience-v0.1",
                split="dev",
                retriever_name="graph",
                top_k=5,
                command="unit-test",
                retriever_config=config_data,
            ),
            build_retriever_from_config(config_data),
        )

        self.assertEqual(result.summary["case_count"], 30)
        self.assertEqual(len(result.case_results), 30)
        first_prediction = result.case_results[0]["predicted"]
        self.assertIn("retrieved", first_prediction)


if __name__ == "__main__":
    unittest.main()
