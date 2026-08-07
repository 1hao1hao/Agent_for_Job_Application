import json
from pathlib import Path
import tempfile
import unittest

from intern_rag.agent import EvidenceConfig, FakeLlmClient, LlmClientError
from intern_rag.evaluation import (
    EvaluationCase,
    LiveLlmRunConfig,
    run_live_llm_end_to_end_evaluation,
)
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult
from intern_rag.routing import RouteDecision


class LiveLlmEvaluationIntegrationTests(unittest.TestCase):
    def test_runner_uses_pipeline_and_builds_live_llm_artifacts(self) -> None:
        chunk = Chunk(
            id="jd-python",
            source_type="jd",
            source_path="data/raw/jd/python.md",
            title="Python 岗位",
            text="岗位要求熟悉 Python。",
            metadata={"source_type": "jd"},
        )
        case = EvaluationCase(
            case_id="live-1",
            query="岗位要求什么语言？",
            category="single_source",
            split="dev",
            expected_intent="analyze_jd",
            expected_sources=["jd"],
            relevant_chunk_ids=[chunk.id],
            answerable=True,
            expected_points=["Python"],
            notes="fixture",
            human_reviewed=True,
        )
        raw = json.dumps({
            "answer": "岗位要求熟悉 Python。",
            "cited_chunk_ids": [chunk.id],
            "sufficient": True,
            "reason": "证据明确。",
        }, ensure_ascii=False)

        def router(query: str) -> RouteDecision:
            del query
            return RouteDecision("analyze_jd", ["jd"], ["岗位"])

        def retriever(query, chunks, top_k=5, source_types=None):
            del query, top_k, source_types
            return [RetrievalResult(chunks[0].id, 0.9, 1, chunks[0])]

        config = LiveLlmRunConfig(
            run_id="live-fixture",
            dataset_version="fixture-v1",
            split="dev",
            router_name="fake",
            retriever_name="hybrid",
            top_k=5,
            context_max_chars=500,
            key_point_threshold=0.5,
            model="fake-model",
            temperature=0.0,
            prompt_version="fixture-v1",
            max_source_retries=1,
            max_format_retries=1,
            input_cache_hit_usd_per_million=0.01,
            input_cache_miss_usd_per_million=0.10,
            output_usd_per_million=0.20,
            pricing_source="fixture",
            pricing_checked_at="2026-08-04",
            git_revision="fixture",
            command="unittest",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_live_llm_end_to_end_evaluation(
                [case],
                [chunk],
                config,
                router,
                retriever,
                FakeLlmClient([raw]),
                Path(temp_dir) / "traces.jsonl",
                evidence_config=EvidenceConfig(),
                support_labels={case.case_id: False},
            )

        self.assertEqual(result.case_results[0]["status"], "answered")
        self.assertEqual(result.summary["report_status"], "formal_live_llm")
        self.assertEqual(result.summary["metrics"]["citation_validity"], 1.0)
        self.assertEqual(result.summary["metrics"]["end_to_end_success_rate"], 1.0)
        self.assertEqual(len(result.traces), 1)

    def test_first_provider_error_stops_repeated_live_calls(self) -> None:
        class BrokenClient:
            def generate(self, prompt, *, model, temperature):
                del prompt, model, temperature
                raise LlmClientError("provider unavailable")

        chunk = Chunk(
            id="jd-1",
            source_type="jd",
            source_path="jd.md",
            title="JD",
            text="岗位要求 Python。",
            metadata={"source_type": "jd"},
        )
        case = EvaluationCase(
            case_id="provider-error",
            query="岗位要求什么？",
            category="single_source",
            split="dev",
            expected_intent="analyze_jd",
            expected_sources=["jd"],
            relevant_chunk_ids=[chunk.id],
            answerable=True,
            expected_points=["Python"],
            notes="fixture",
            human_reviewed=True,
        )
        config = LiveLlmRunConfig(
            run_id="provider-error",
            dataset_version="fixture",
            split="dev",
            router_name="fake",
            retriever_name="hybrid",
            top_k=1,
            context_max_chars=500,
            key_point_threshold=0.5,
            model="fake",
            temperature=0.0,
            prompt_version="fixture",
            max_source_retries=1,
            max_format_retries=1,
            input_cache_hit_usd_per_million=0.0,
            input_cache_miss_usd_per_million=0.0,
            output_usd_per_million=0.0,
            pricing_source="fixture",
            pricing_checked_at="fixture",
            git_revision="fixture",
            command="unittest",
        )
        router = lambda query: RouteDecision("analyze_jd", ["jd"], ["岗位"])
        retriever = lambda query, chunks, top_k=5, source_types=None: [
            RetrievalResult(chunks[0].id, 0.9, 1, chunks[0])
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(RuntimeError):
                run_live_llm_end_to_end_evaluation(
                    [case], [chunk], config, router, retriever, BrokenClient(),
                    Path(temp_dir) / "trace.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
