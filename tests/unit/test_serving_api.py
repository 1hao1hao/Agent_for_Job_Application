import json
import time
import unittest

from fastapi.testclient import TestClient

from intern_rag.agent import Citation, RagRequest, RagResponse
from intern_rag.routing import RouteDecision
from intern_rag.serving import AppServices, create_app
from intern_rag.tracing import build_agent_trace
from tests.support import InMemoryJobQueue, InMemoryPersistenceRepository


class FakeQueryService:
    def __init__(self, status: str = "answered", delay: float = 0.0) -> None:
        self.status = status
        self.delay = delay

    def execute(self, request: RagRequest) -> RagResponse:
        if self.delay:
            time.sleep(self.delay)
        return RagResponse(
            request_id=request.request_id,
            trace_id="trace-1",
            answer="基于证据回答" if self.status == "answered" else "证据不足",
            citations=(
                [Citation("chunk-1", "data/raw/jd/a.md", "jd", "岗位", 1, 1.0)]
                if self.status == "answered"
                else []
            ),
            routed_sources=["jd"],
            status=self.status,  # type: ignore[arg-type]
            latency_ms=12.5,
            error_type="llm_timeout" if self.status == "error" else None,
        )


class ServingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryPersistenceRepository()
        self.queue = InMemoryJobQueue()
        self.client = TestClient(
            create_app(
                AppServices(FakeQueryService(), self.repository, self.queue)
            )
        )

    def test_health_and_answered_query(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)

        response = self.client.post(
            "/v1/query", json={"query": "分析岗位", "retriever": "bm25"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "answered")
        self.assertEqual(response.json()["citations"][0]["chunk_id"], "chunk-1")

    def test_validation_timeout_and_pipeline_error_mapping(self) -> None:
        self.assertEqual(self.client.post("/v1/query", json={"query": ""}).status_code, 422)
        self.assertEqual(self.client.post("/v1/query", json={"query": "   "}).status_code, 422)
        timeout_client = TestClient(
            create_app(
                AppServices(
                    FakeQueryService(delay=0.05),
                    self.repository,
                    self.queue,
                    query_timeout_seconds=0.001,
                )
            )
        )
        self.assertEqual(
            timeout_client.post("/v1/query", json={"query": "超时"}).status_code,
            504,
        )
        error_client = TestClient(
            create_app(
                AppServices(FakeQueryService("error"), self.repository, self.queue)
            )
        )
        self.assertEqual(
            error_client.post("/v1/query", json={"query": "模型超时"}).status_code,
            504,
        )

    def test_trace_query_and_not_found(self) -> None:
        trace = build_agent_trace(
            "问题",
            RouteDecision("analyze_jd", ["jd"], []),
            [],
            {"total": 1.0},
            trace_id="trace-saved",
        )
        self.repository.save_trace(trace)

        self.assertEqual(
            self.client.get("/v1/traces/trace-saved").json()["trace_id"],
            "trace-saved",
        )
        self.assertEqual(self.client.get("/v1/traces/missing").status_code, 404)

    def test_evaluation_job_idempotency_status_and_retry(self) -> None:
        payload = {
            "dataset_version": "evalrag_v0.2",
            "split": "dev",
            "run_config": {
                "retriever_config_path": "configs/retrieval/bm25_v0.2.json"
            },
        }
        first = self.client.post(
            "/v1/evaluation-jobs",
            json=payload,
            headers={"Idempotency-Key": "same-job"},
        )
        second = self.client.post(
            "/v1/evaluation-jobs",
            json=payload,
            headers={"Idempotency-Key": "same-job"},
        )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])
        self.assertEqual(len(self.queue.job_ids), 1)
        job_id = first.json()["job_id"]
        self.assertEqual(
            self.client.get(f"/v1/evaluation-jobs/{job_id}").json()["status"],
            "queued",
        )

        self.repository.mark_job_failed(job_id, "evaluation_timeout", "timeout")
        retry = self.client.post(f"/v1/evaluation-jobs/{job_id}/retry")
        self.assertEqual(retry.status_code, 202)
        self.assertEqual(retry.json()["status"], "queued")

    def test_queue_failure_is_persisted_and_returns_503(self) -> None:
        self.queue.available = False
        response = self.client.post(
            "/v1/evaluation-jobs",
            json={
                "dataset_version": "evalrag_v0.2",
                "split": "dev",
                "run_config": {
                    "retriever_config_path": "configs/retrieval/bm25_v0.2.json"
                },
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(next(iter(self.repository.jobs.values())).status, "failed")


if __name__ == "__main__":
    unittest.main()
