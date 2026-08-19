from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from intern_rag.agent import Citation, RagRequest, RagResponse
from intern_rag.runtime import (
    AgentRuntime,
    FileCheckpointStore,
    RunContext,
    RuntimeStage,
    SavedRun,
    replay_run,
)
from intern_rag.runtime.langgraph_adapter import run_with_langgraph
from intern_rag.tracing import AgentTrace


class _Executor:
    def execute(self, request: RagRequest) -> tuple[RagResponse, AgentTrace]:
        citation = Citation("chunk-1", "data/a.md", "jd", "岗位", 1, 0.9)
        response = RagResponse(
            request.request_id, "trace-1", "基于证据回答", [citation], ["jd"],
            "answered", 3.0, None,
        )
        trace = AgentTrace(
            request_id=request.request_id,
            query=request.query,
            intent="analyze_jd",
            routed_sources=["jd"],
            retrieved_chunks=[{"chunk_id": "chunk-1"}],
            latency_ms={"routing": 1.0, "retrieval": 1.0, "generation": 1.0},
            trace_id="trace-1",
            citations=[citation.to_dict()],
            answer=response.answer,
            response_status="answered",
            attempts=[{"attempt": 1, "type": "generation", "status": "succeeded"}],
            token_usage={"input_tokens": 12, "output_tokens": 6},
        )
        return response, trace


class _FailingSink:
    def write(self, event) -> None:
        raise OSError("sink unavailable")


class AgentRuntimeTests(unittest.TestCase):
    def test_execute_builds_hierarchy_and_isolates_sink_failure(self) -> None:
        runtime = AgentRuntime(_Executor(), span_sinks=[_FailingSink()])
        request = RagRequest("岗位要求是什么", request_id="request-1")

        execution = runtime.execute(request, RunContext(run_id="run-1", request_id="request-1"))

        self.assertEqual(execution.response.status, "answered")
        self.assertEqual(execution.spans[0].name, "agent.run")
        self.assertTrue(all(span.parent_span_id == execution.spans[0].span_id for span in execution.spans[1:]))
        self.assertGreater(len(execution.observability_errors), 0)
        generation = next(span for span in execution.spans if span.name == "generation")
        self.assertEqual(generation.attributes["token_usage"]["input_tokens"], 12)
        self.assertEqual(generation.input_refs["request_id"], "request-1")

    def test_fake_full_replay_and_external_generation_boundary(self) -> None:
        request = RagRequest("岗位要求是什么", request_id="request-1")
        context = RunContext(run_id="run-1", request_id="request-1", model_version="fake-v1")
        saved = SavedRun(
            context=context,
            request=request,
            response_summary={
                "status": "answered", "answer": "基于证据回答",
                "citation_ids": ["chunk-1"], "error_type": None,
                "attempt_types": ["generation"],
            },
            trace_summary={"retrieval": {"chunk_ids": ["chunk-1"]}},
            external_model_reproducible=True,
        )

        replayed = replay_run(saved, AgentRuntime(_Executor()))
        unavailable = replay_run(
            SavedRun(context, request, saved.response_summary, saved.trace_summary, False),
            AgentRuntime(_Executor()),
            stage="generation",
        )

        self.assertTrue(replayed.matched)
        self.assertFalse(unavailable.replayable)
        self.assertIsNone(unavailable.matched)

    def test_checkpoint_resume_skips_completed_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "index.json"
            artifact.write_text("{}", encoding="utf-8")
            runtime = AgentRuntime(checkpoint_store=FileCheckpointStore(Path(directory) / "checkpoints"))
            context = RunContext(run_id="resume-1", artifact_refs={"index": str(artifact)})
            calls = {"retrieve": 0, "generate": 0}

            def retrieve(state: dict[str, object]) -> dict[str, object]:
                calls["retrieve"] += 1
                return {**state, "chunks": ["chunk-1"]}

            def generate(state: dict[str, object]) -> dict[str, object]:
                calls["generate"] += 1
                if calls["generate"] == 1:
                    raise RuntimeError("interrupted")
                return {**state, "answer": "ok"}

            stages = [
                RuntimeStage("retrieval", retrieve, "retrieval-v1"),
                RuntimeStage("generation", generate, "generation-v1"),
            ]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                runtime.run_stages(context, stages, {}, resume=False)
            result = runtime.run_stages(context, stages, {}, resume=True)

            self.assertEqual(result["answer"], "ok")
            self.assertEqual(calls, {"retrieve": 1, "generate": 2})

    def test_config_or_artifact_drift_restarts_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "index.json"
            artifact.write_text("{}", encoding="utf-8")
            store = FileCheckpointStore(Path(directory) / "checkpoints")
            runtime = AgentRuntime(checkpoint_store=store)
            calls = {"count": 0}

            def stage(state: dict[str, object]) -> dict[str, object]:
                calls["count"] += 1
                return {"value": calls["count"]}

            old = RunContext(run_id="drift", config={"top_k": 3}, artifact_refs={"index": str(artifact)})
            runtime.run_stages(old, [RuntimeStage("retrieval", stage)], {})
            changed = RunContext(run_id="drift", config={"top_k": 5}, artifact_refs={"index": str(artifact)})
            runtime.run_stages(changed, [RuntimeStage("retrieval", stage)], {}, resume=True)
            artifact.unlink()
            runtime.run_stages(changed, [RuntimeStage("retrieval", stage)], {}, resume=True)

            self.assertEqual(calls["count"], 3)

    def test_resume_after_generation_does_not_repeat_model_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = AgentRuntime(
                checkpoint_store=FileCheckpointStore(Path(directory) / "checkpoints")
            )
            context = RunContext(run_id="after-generation")
            calls = {"generation": 0, "validation": 0}

            def generation(state: dict[str, object]) -> dict[str, object]:
                calls["generation"] += 1
                return {**state, "answer": "ok"}

            def validation(state: dict[str, object]) -> dict[str, object]:
                calls["validation"] += 1
                return {**state, "valid": True}

            runtime.run_stages(
                context, [RuntimeStage("generation", generation, "llm-call-1")], {}
            )
            result = runtime.run_stages(
                context,
                [
                    RuntimeStage("generation", generation, "llm-call-1"),
                    RuntimeStage("validation", validation),
                ],
                {},
                resume=True,
            )

            self.assertTrue(result["valid"])
            self.assertEqual(calls, {"generation": 1, "validation": 1})

    def test_resume_after_context_does_not_rebuild_completed_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = AgentRuntime(
                checkpoint_store=FileCheckpointStore(Path(directory) / "checkpoints")
            )
            context = RunContext(run_id="after-context")
            calls = {"context": 0, "generation": 0}

            def build_context(state: dict[str, object]) -> dict[str, object]:
                calls["context"] += 1
                return {**state, "context": "evidence"}

            def generation(state: dict[str, object]) -> dict[str, object]:
                calls["generation"] += 1
                if calls["generation"] == 1:
                    raise RuntimeError("interrupted after context")
                return {**state, "answer": "ok"}

            stages = [
                RuntimeStage("context", build_context),
                RuntimeStage("generation", generation, "llm-call-context-case"),
            ]
            with self.assertRaisesRegex(RuntimeError, "after context"):
                runtime.run_stages(context, stages, {})
            result = runtime.run_stages(context, stages, {}, resume=True)

            self.assertEqual(result["answer"], "ok")
            self.assertEqual(calls, {"context": 1, "generation": 2})

    def test_langgraph_adapter_keeps_fixed_case_state(self) -> None:
        def handler(state: dict[str, object]) -> dict[str, object]:
            return {**state, "status": "answered", "trace_id": "trace-fixed"}

        direct = handler({"query": "固定问题"})
        graph = run_with_langgraph({"query": "固定问题"}, handler)

        self.assertEqual(graph, direct)


if __name__ == "__main__":
    unittest.main()
