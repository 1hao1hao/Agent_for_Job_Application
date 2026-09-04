from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from intern_rag.agent import EvidenceConfig, FakeLlmClient, PipelineConfig, RagPipeline, RagRequest
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult
from intern_rag.routing import RouteDecision
from intern_rag.runtime import (
    AgentRuntime,
    PipelineRuntimeExecutor,
    RunContext,
    SavedRun,
    build_run_context_snapshot,
    replay_saved_run,
)
from intern_rag.runtime.replay import request_response_from_dict


MODEL_OUTPUT = json.dumps({
    "answer": "岗位要求熟悉 Python。",
    "cited_chunk_ids": ["jd-1"],
    "sufficient": True,
    "reason": "岗位证据明确",
}, ensure_ascii=False)


def _chunk() -> Chunk:
    return Chunk(
        "jd-1", "jd", "data/raw/jd/test.md", "后端实习生",
        "岗位要求熟悉 Python。", {"source_type": "jd"},
    )


def _route(_: str) -> RouteDecision:
    return RouteDecision("analyze_jd", ["jd"], ["岗位"], strategy="rule")


def _retriever(query, chunks, top_k=5, source_types=None):
    del query
    selected = [item for item in chunks if source_types is None or item.source_type in source_types]
    return [
        RetrievalResult(item.id, 1.0, rank, item, "fixed")
        for rank, item in enumerate(selected[:top_k], 1)
    ]


class DeterministicReplayTests(unittest.TestCase):
    def _pipeline(self, client, trace_path: Path, *, route=_route, retriever=_retriever) -> RagPipeline:
        return RagPipeline(
            [_chunk()], client,
            PipelineConfig(
                model="saved-model-v1", prompt_version="prompt-v1",
                evidence=EvidenceConfig(min_scores={"keyword": 0.0}),
            ),
            trace_path=trace_path,
            router=route,
            retriever=retriever,
        )

    def _saved(self, directory: Path) -> SavedRun:
        artifact = directory / "chunks.jsonl"
        artifact.write_text("snapshot", encoding="utf-8")
        request = RagRequest("分析岗位要求", request_id="request-original")
        context = RunContext(
            run_id="run-original", request_id=request.request_id,
            config={"router": "rule", "retriever": "keyword", "top_k": 5},
            artifact_refs={"chunks": str(artifact)}, dataset_version="test-v1",
            model_version="saved-model-v1", prompt_version="prompt-v1", index_version="index-v1",
        )
        execution = AgentRuntime(PipelineRuntimeExecutor(
            self._pipeline(FakeLlmClient([MODEL_OUTPUT]), directory / "original.jsonl")
        )).execute(request, context)
        trace = replace(execution.trace, run_context=build_run_context_snapshot(context))
        return SavedRun.from_persisted(request, execution.response, trace)

    def test_same_snapshot_matches_all_stages_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = self._saved(root)
            received_clients = []

            def factory(_saved, replay_client):
                received_clients.append(replay_client)
                return AgentRuntime(PipelineRuntimeExecutor(
                    self._pipeline(replay_client, root / "replayed.jsonl")
                ))

            result = replay_saved_run(saved, factory)

            self.assertTrue(result.replayable)
            self.assertTrue(result.matched)
            self.assertIsNone(result.first_divergent_stage)
            self.assertEqual([item.stage for item in result.stage_results], list((
                "routing", "retrieval", "evidence", "context", "generation", "validation"
            )))
            self.assertEqual(len(received_clients[0].prompts), 1)

    def test_changed_routing_locates_first_divergent_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = self._saved(root)

            def changed_route(_: str) -> RouteDecision:
                return RouteDecision("unknown", [], [], strategy="rule")

            result = replay_saved_run(
                saved,
                lambda _saved, client: AgentRuntime(PipelineRuntimeExecutor(
                    self._pipeline(client, root / "changed.jsonl", route=changed_route)
                )),
            )

            self.assertFalse(result.matched)
            self.assertEqual(result.first_divergent_stage, "routing")
            self.assertTrue(result.stage_results[0].differences)

    def test_changed_retrieval_locates_retrieval_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = self._saved(root)

            def empty_retriever(query, chunks, top_k=5, source_types=None):
                del query, chunks, top_k, source_types
                return []

            result = replay_saved_run(
                saved,
                lambda _saved, client: AgentRuntime(PipelineRuntimeExecutor(
                    self._pipeline(client, root / "retrieval-change.jsonl", retriever=empty_retriever)
                )),
            )

            self.assertEqual(result.first_divergent_stage, "retrieval")
            retrieval = next(item for item in result.stage_results if item.stage == "retrieval")
            self.assertTrue(any(diff.path.endswith("chunk_ids[0]") for diff in retrieval.differences))

    def test_volatile_ids_timestamps_and_latency_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = self._saved(root)
            changed_trace = replace(
                saved.trace,
                trace_id="other-trace",
                created_at="2099-01-01T00:00:00Z",
                latency_ms={"routing": 999.0, "total": 9999.0},
            )
            saved = replace(saved, trace=changed_trace)
            result = replay_saved_run(
                saved,
                lambda _saved, client: AgentRuntime(PipelineRuntimeExecutor(
                    self._pipeline(client, root / "volatile.jsonl")
                )),
                stage="routing",
            )
            self.assertTrue(result.matched)

    def test_missing_artifact_or_model_output_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = self._saved(root)
            missing_artifact = replace(
                saved,
                context=replace(saved.context, artifact_refs={"chunks": str(root / "missing")}),
            )
            missing_model = replace(saved, model_outputs=())
            missing_snapshot = replace(saved, trace=None)

            artifact_result = replay_saved_run(missing_artifact, lambda *_: self.fail("factory called"))
            model_result = replay_saved_run(missing_model, lambda *_: self.fail("factory called"))
            snapshot_result = replay_saved_run(missing_snapshot, lambda *_: self.fail("factory called"))

            self.assertFalse(artifact_result.replayable)
            self.assertIn("artifact missing", artifact_result.reason)
            self.assertFalse(model_result.replayable)
            self.assertIn("model output missing", model_result.reason)
            self.assertFalse(snapshot_result.replayable)
            self.assertIn("snapshot missing", snapshot_result.reason)

    def test_persisted_request_response_round_trip_keeps_replay_inputs(self) -> None:
        request_payload = {
            "query": "分析岗位要求", "request_id": "request-1", "top_k": 7,
            "retriever": "bm25", "user_id": None, "session_id": None,
        }
        response_payload = {
            "request_id": "request-1", "trace_id": "trace-1", "answer": "回答",
            "citations": [], "routed_sources": ["jd"], "status": "answered",
            "latency_ms": 10.0, "error_type": None,
        }

        request, response = request_response_from_dict(request_payload, response_payload)

        self.assertEqual((request.retriever, request.top_k), ("bm25", 7))
        self.assertEqual((response.trace_id, response.status), ("trace-1", "answered"))


if __name__ == "__main__":
    unittest.main()
