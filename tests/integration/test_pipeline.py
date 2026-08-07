from pathlib import Path
import json
import tempfile
import unittest

from intern_rag.agent import (
    EvidenceConfig,
    FakeLlmClient,
    PipelineConfig,
    RagPipeline,
    RagRequest,
)
from intern_rag.agent.generation import LlmTimeoutError
from intern_rag.ingestion import Chunk
from intern_rag.tracing import read_traces_jsonl
from intern_rag.retrieval import RetrievalResult


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="jd-1",
            source_type="jd",
            source_path="data/raw/jd/test.md",
            title="大模型应用研发实习生",
            text="岗位要求熟悉 Python、RAG 和效果评测。",
            metadata={"source_type": "jd", "status": "active"},
        ),
        Chunk(
            id="resume-1",
            source_type="resume",
            source_path="data/raw/resume/test.md",
            title="候选人简历",
            text="候选人使用 Python 开发过 RAG 项目。",
            metadata={"source_type": "resume"},
        ),
    ]


def _raw_generation(
    *,
    answer: str,
    cited_chunk_ids: list[str],
    sufficient: bool,
    reason: str,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "cited_chunk_ids": cited_chunk_ids,
            "sufficient": sufficient,
            "reason": reason,
        },
        ensure_ascii=False,
    )


class PipelineIntegrationTests(unittest.TestCase):
    def _run_pipeline(
        self,
        query: str,
        raw_response: str,
    ) -> tuple[object, list[object]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "pipeline.jsonl"
            pipeline = RagPipeline(
                chunks=_chunks(),
                llm_client=FakeLlmClient([raw_response, raw_response]),
                config=PipelineConfig(
                    model="fake-model",
                    temperature=0.0,
                    prompt_version="p0-v1",
                    context_max_chars=2000,
                ),
                trace_path=trace_path,
            )
            response = pipeline.run(
                RagRequest(query=query, request_id="req-integration", top_k=2)
            )
            traces = read_traces_jsonl(trace_path)
        return response, traces

    def test_answered_query_runs_all_stages_and_writes_one_trace(self) -> None:
        response, traces = self._run_pipeline(
            "分析这个岗位的 Python 要求",
            _raw_generation(
                answer="岗位要求熟悉 Python、RAG 和效果评测。",
                cited_chunk_ids=["jd-1"],
                sufficient=True,
                reason="JD 证据明确。",
            ),
        )

        self.assertEqual(response.status, "answered")
        self.assertEqual(response.error_type, None)
        self.assertEqual(response.citations[0].chunk_id, "jd-1")
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace.trace_id, response.trace_id)
        self.assertEqual(trace.response_status, "answered")
        self.assertEqual(trace.routing["intent"], "analyze_jd")
        self.assertEqual(trace.retrieval["chunk_ids"], ["jd-1"])
        self.assertEqual(trace.context["used_chunk_ids"], ["jd-1"])
        self.assertEqual(trace.generation["status"], "parsed")
        self.assertTrue(trace.validation["is_valid"])
        self.assertEqual(
            set(trace.latency_ms),
            {
                "routing",
                "retrieval",
                "context",
                "generation",
                "validation",
                "evidence",
                "total",
            },
        )

    def test_request_can_select_configured_dense_retriever(self) -> None:
        def dense_retriever(query, chunks, top_k=5, source_types=None):
            del query, top_k, source_types
            chunk = chunks[0]
            return [RetrievalResult(chunk.id, 0.9, 1, chunk, "fake dense")]

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RagPipeline(
                chunks=_chunks(),
                llm_client=FakeLlmClient([_raw_generation(
                    answer="岗位要求 Python。",
                    cited_chunk_ids=["jd-1"],
                    sufficient=True,
                    reason="证据充分",
                )]),
                config=PipelineConfig(model="fake-model"),
                trace_path=Path(temp_dir) / "trace.jsonl",
                retrievers={"dense": dense_retriever},
            )

            response = pipeline.run(RagRequest(
                query="分析岗位要求",
                retriever="dense",
            ))

        self.assertEqual(response.status, "answered")

    def test_insufficient_query_returns_controlled_abstention(self) -> None:
        response, traces = self._run_pipeline(
            "资料中有没有量子芯片流片经历？",
            _raw_generation(
                answer="当前资料中没有足够证据回答。",
                cited_chunk_ids=[],
                sufficient=False,
                reason="没有可用证据。",
            ),
        )

        self.assertEqual(response.status, "insufficient_evidence")
        self.assertEqual(response.citations, [])
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].response_status, "insufficient_evidence")
        self.assertEqual(traces[0].error_type, "retrieval_miss")

    def test_invalid_json_returns_error_and_still_writes_one_trace(self) -> None:
        response, traces = self._run_pipeline(
            "分析这个岗位要求",
            "not-json",
        )

        self.assertEqual(response.status, "error")
        self.assertEqual(response.error_type, "llm_format_error")
        self.assertEqual(response.citations, [])
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].generation["error_type"], "invalid_json")
        self.assertEqual(traces[0].error_type, "llm_format_error")

    def test_format_error_is_retried_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "retry.jsonl"
            client = FakeLlmClient([
                "not-json",
                _raw_generation(
                    answer="岗位要求熟悉 Python。",
                    cited_chunk_ids=["jd-1"],
                    sufficient=True,
                    reason="第二次输出格式合法。",
                ),
            ])
            pipeline = RagPipeline(
                chunks=_chunks(), llm_client=client,
                config=PipelineConfig(model="fake-model"),
                trace_path=trace_path,
            )
            response = pipeline.run(RagRequest(query="分析岗位要求", top_k=2))
            trace = read_traces_jsonl(trace_path)[0]

        self.assertEqual(response.status, "answered")
        generation_attempts = [
            item for item in trace.attempts
            if item["type"] in {"initial_generation", "format_repair"}
        ]
        self.assertEqual(len(generation_attempts), 2)
        self.assertEqual(generation_attempts[1]["type"], "format_repair")

    def test_source_expansion_is_retried_only_once(self) -> None:
        calls = []

        def expanding_retriever(query, chunks, top_k=5, source_types=None):
            del query, top_k
            calls.append(source_types)
            if source_types is not None:
                return []
            chunk = chunks[0]
            return [RetrievalResult(chunk.id, 0.9, 1, chunk)]

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "source-retry.jsonl"
            pipeline = RagPipeline(
                chunks=_chunks(),
                llm_client=FakeLlmClient([_raw_generation(
                    answer="岗位要求熟悉 Python。", cited_chunk_ids=["jd-1"],
                    sufficient=True, reason="扩源后找到证据。",
                )]),
                config=PipelineConfig(model="fake-model"),
                trace_path=trace_path,
                retriever=expanding_retriever,
            )
            response = pipeline.run(RagRequest(query="分析岗位要求"))
            trace = read_traces_jsonl(trace_path)[0]

        self.assertEqual(response.status, "answered")
        self.assertEqual(calls, [{"jd"}, None])
        self.assertEqual(len(trace.attempts), 3)

    def test_timeout_returns_controlled_error_without_loop(self) -> None:
        class TimeoutClient:
            def generate(self, prompt, *, model, temperature):
                del prompt, model, temperature
                raise LlmTimeoutError("fake timeout")

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "timeout.jsonl"
            pipeline = RagPipeline(
                chunks=_chunks(), llm_client=TimeoutClient(),
                config=PipelineConfig(model="fake-model"), trace_path=trace_path,
            )
            response = pipeline.run(RagRequest(query="分析岗位要求"))
            trace = read_traces_jsonl(trace_path)[0]

        self.assertEqual(response.error_type, "llm_timeout")
        self.assertEqual(trace.attempts[-1]["status"], "timeout")

    def test_source_retry_exhaustion_returns_retrieval_miss(self) -> None:
        calls = []

        def empty_retriever(query, chunks, top_k=5, source_types=None):
            del query, chunks, top_k
            calls.append(source_types)
            return []

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "exhausted.jsonl"
            pipeline = RagPipeline(
                chunks=_chunks(), llm_client=FakeLlmClient([]),
                config=PipelineConfig(model="fake-model"),
                trace_path=trace_path, retriever=empty_retriever,
            )
            response = pipeline.run(RagRequest(query="分析岗位职责"))
            trace = read_traces_jsonl(trace_path)[0]

        self.assertEqual(response.status, "insufficient_evidence")
        self.assertEqual(response.error_type, "retrieval_miss")
        self.assertEqual(len(calls), 2)
        self.assertEqual(trace.evidence["status"], "insufficient")

    def test_unknown_route_abstains_without_calling_model(self) -> None:
        client = FakeLlmClient([])
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "unknown.jsonl"
            pipeline = RagPipeline(
                chunks=_chunks(), llm_client=client,
                config=PipelineConfig(model="fake-model"), trace_path=trace_path,
            )
            response = pipeline.run(RagRequest(query="量子芯片流片编号是多少"))
            trace = read_traces_jsonl(trace_path)[0]

        self.assertEqual(response.status, "insufficient_evidence")
        self.assertIsNone(response.error_type)
        self.assertEqual(client.prompts, [])
        self.assertEqual(trace.evidence["reason"], "unanswerable_route")

    def test_invalid_citation_cannot_return_answered(self) -> None:
        response, traces = self._run_pipeline(
            "分析这个岗位要求",
            _raw_generation(
                answer="岗位要求掌握 Go。",
                cited_chunk_ids=["not-in-context"],
                sufficient=True,
                reason="模型返回了不存在的引用。",
            ),
        )

        self.assertEqual(response.status, "error")
        self.assertEqual(response.error_type, "citation_invalid")
        self.assertEqual(response.citations, [])
        self.assertEqual(len(traces), 1)
        self.assertFalse(traces[0].validation["is_valid"])
        self.assertEqual(
            traces[0].validation["issues"][0]["error_type"],
            "citation_not_found",
        )


if __name__ == "__main__":
    unittest.main()
