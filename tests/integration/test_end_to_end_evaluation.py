from pathlib import Path
import tempfile
import unittest

from intern_rag.agent import EvidenceConfig
from intern_rag.evaluation import (
    EndToEndRunConfig,
    EvaluationCase,
    run_extractive_end_to_end_evaluation,
    save_end_to_end_artifacts,
)
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult
from intern_rag.routing import RouteDecision


class EndToEndEvaluationIntegrationTests(unittest.TestCase):
    def test_runner_generates_prediction_trace_metrics_and_artifacts(self) -> None:
        chunk = Chunk(
            id="jd-python",
            source_type="jd",
            source_path="data/raw/jd/python.md",
            title="Python 实习岗位",
            text="岗位要求熟悉 Python。",
            metadata={"source_type": "jd"},
        )
        case = EvaluationCase(
            case_id="case-1",
            query="岗位要求什么语言？",
            category="single_source",
            split="dev",
            expected_intent="analyze_jd",
            expected_sources=["jd"],
            relevant_chunk_ids=[chunk.id],
            answerable=True,
            expected_points=["Python"],
            notes="集成测试 fixture",
            human_reviewed=True,
        )

        def router(query: str) -> RouteDecision:
            del query
            return RouteDecision("analyze_jd", ["jd"], ["岗位"])

        def retriever(query, chunks, top_k=5, source_types=None):
            del query, top_k, source_types
            return [RetrievalResult(chunks[0].id, 0.9, 1, chunks[0])]

        config = EndToEndRunConfig(
            run_id="fixture-run",
            dataset_version="fixture-v1",
            split="dev",
            router_name="fake",
            retriever_name="hybrid",
            top_k=5,
            context_max_chars=500,
            key_point_threshold=0.5,
            git_revision="fixture",
            command="unittest fixture",
        )
        result = run_extractive_end_to_end_evaluation(
            [case],
            [chunk],
            config,
            router,
            retriever,
            evidence_config=EvidenceConfig(require_source_coverage=True),
        )

        self.assertEqual(result.case_results[0]["status"], "answered")
        self.assertEqual(result.summary["metrics"]["citation_validity"], 1.0)
        self.assertEqual(result.summary["metrics"]["end_to_end_success_rate"], 1.0)
        self.assertEqual(len(result.traces), 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            save_end_to_end_artifacts(result, run_dir)
            self.assertTrue((run_dir / "case_results.jsonl").exists())
            self.assertTrue((run_dir / "traces.jsonl").exists())
            self.assertTrue((run_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
