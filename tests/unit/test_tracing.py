from pathlib import Path
import tempfile
import unittest

from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult
from intern_rag.routing import RouteDecision
from intern_rag.tracing import (
    AgentTrace,
    build_agent_trace,
    read_traces_jsonl,
    retrieval_result_to_trace,
    write_trace_jsonl,
)


def _retrieval_result() -> RetrievalResult:
    chunk = Chunk(
        id="jd-1",
        source_type="jd",
        source_path="data/raw/jd/backend_intern.md",
        title="大模型应用研发实习生",
        text="岗位要求熟悉 Python、RAG 和向量检索。",
        metadata={"source_type": "jd", "status": "active"},
    )
    return RetrievalResult(
        chunk_id=chunk.id,
        score=0.8,
        rank=1,
        chunk=chunk,
        reason="rag, 向量, 检索",
    )


class TracingTests(unittest.TestCase):
    def test_build_agent_trace_records_single_turn_fields(self) -> None:
        route_decision = RouteDecision(
            intent="analyze_jd",
            routed_sources=["jd"],
            matched_keywords=["岗位"],
        )

        trace = build_agent_trace(
            query="分析这个岗位要求",
            route_decision=route_decision,
            retrieved_results=[_retrieval_result()],
            latency_ms={"routing": 1.0, "retrieval": 2.5},
            request_id="req-1",
            created_at="2026-06-18T00:00:00+00:00",
        )

        self.assertEqual(trace.request_id, "req-1")
        self.assertEqual(trace.query, "分析这个岗位要求")
        self.assertEqual(trace.intent, "analyze_jd")
        self.assertEqual(trace.routed_sources, ["jd"])
        self.assertEqual(trace.error_type, "none")
        self.assertEqual(trace.latency_ms["retrieval"], 2.5)
        self.assertEqual(trace.retrieved_chunks[0]["chunk_id"], "jd-1")
        self.assertEqual(trace.retrieved_chunks[0]["rank"], 1)
        self.assertEqual(trace.retrieved_chunks[0]["source_type"], "jd")

    def test_write_and_read_trace_jsonl(self) -> None:
        trace = AgentTrace(
            request_id="req-2",
            query="我的简历和岗位匹配吗",
            intent="match_resume",
            routed_sources=["jd", "resume"],
            retrieved_chunks=[retrieval_result_to_trace(_retrieval_result())],
            latency_ms={"total": 3.0},
            created_at="2026-06-18T00:00:00+00:00",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            write_trace_jsonl(trace, path)
            loaded_traces = read_traces_jsonl(path)

        self.assertEqual(len(loaded_traces), 1)
        self.assertEqual(loaded_traces[0].request_id, "req-2")
        self.assertEqual(loaded_traces[0].intent, "match_resume")
        self.assertEqual(loaded_traces[0].retrieved_chunks[0]["chunk_id"], "jd-1")

    def test_write_trace_jsonl_appends_multiple_traces(self) -> None:
        first_trace = AgentTrace(
            request_id="req-3",
            query="分析 JD",
            intent="analyze_jd",
            routed_sources=["jd"],
            retrieved_chunks=[],
            latency_ms={"total": 1.0},
        )
        second_trace = AgentTrace(
            request_id="req-4",
            query="准备面试",
            intent="interview_prepare",
            routed_sources=["interview", "jd", "resume"],
            retrieved_chunks=[],
            latency_ms={"total": 1.5},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            write_trace_jsonl(first_trace, path)
            write_trace_jsonl(second_trace, path)
            loaded_traces = read_traces_jsonl(path)

        self.assertEqual([trace.request_id for trace in loaded_traces], ["req-3", "req-4"])

    def test_error_type_is_recorded_and_unknown_values_are_normalized(self) -> None:
        trace = AgentTrace(
            request_id="req-5",
            query="找不到证据",
            intent="analyze_jd",
            routed_sources=["jd"],
            retrieved_chunks=[],
            latency_ms={"retrieval": 2.0},
            error_type="retrieval_miss",
        )
        trace_data = trace.to_dict() | {"error_type": "unexpected_error"}

        self.assertEqual(trace.error_type, "retrieval_miss")
        self.assertEqual(AgentTrace.from_dict(trace_data).error_type, "unknown_error")

    def test_read_missing_trace_file_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            traces = read_traces_jsonl(Path(temp_dir) / "missing.jsonl")

        self.assertEqual(traces, [])


if __name__ == "__main__":
    unittest.main()
