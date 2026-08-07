import os
from pathlib import Path
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from intern_rag.agent import RagRequest, RagResponse
from intern_rag.persistence import PostgresRepository
from intern_rag.serving import AppServices, create_app
from intern_rag.serving.runtime import create_runtime_app
from intern_rag.worker import EvaluationWorker, RedisJobQueue, SubprocessEvaluationExecutor


class UnusedQueryService:
    def execute(self, request: RagRequest) -> RagResponse:
        raise AssertionError("this integration test only exercises evaluation jobs")


@unittest.skipUnless(
    os.environ.get("TEST_DATABASE_URL") and os.environ.get("TEST_REDIS_URL"),
    "TEST_DATABASE_URL/TEST_REDIS_URL not set",
)
class ServingStackIntegrationTests(unittest.TestCase):
    def test_runtime_query_persists_request_trace_and_citation(self) -> None:
        os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
        os.environ["REDIS_URL"] = os.environ["TEST_REDIS_URL"]
        os.environ["EVALRAG_LLM_BACKEND"] = "fake"
        client = TestClient(create_runtime_app())

        response = client.post(
            "/v1/query",
            json={
                "query": "请分析大模型应用研发实习生的岗位要求",
                "retriever": "bm25",
                "top_k": 5,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "answered")
        self.assertTrue(response.json()["citations"])
        trace = client.get(f"/v1/traces/{response.json()['trace_id']}")
        self.assertEqual(trace.status_code, 200)
        self.assertEqual(trace.json()["retrieval"]["retriever"], "bm25")

    def test_api_queue_worker_report_and_persistent_status(self) -> None:
        repository = PostgresRepository(
            os.environ["TEST_DATABASE_URL"], Path("migrations")
        )
        repository.initialize()
        queue = RedisJobQueue(
            os.environ["TEST_REDIS_URL"], f"evalrag:test:{uuid4()}"
        )
        client = TestClient(
            create_app(AppServices(UnusedQueryService(), repository, queue))
        )
        run_id = f"p1-d1-stack-{uuid4()}"
        created = client.post(
            "/v1/evaluation-jobs",
            headers={"Idempotency-Key": run_id},
            json={
                "dataset_version": "evalrag_v0.2",
                "split": "dev",
                "run_config": {
                    "run_id": run_id,
                    "retriever_config_path": "configs/retrieval/bm25_v0.2.json",
                },
            },
        )
        self.assertEqual(created.status_code, 202)
        job_id = created.json()["job_id"]

        worker = EvaluationWorker(
            repository,
            queue,
            SubprocessEvaluationExecutor(Path.cwd(), timeout_seconds=300),
        )
        self.assertTrue(worker.run_once(timeout_seconds=1))

        completed = client.get(f"/v1/evaluation-jobs/{job_id}")
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json()["status"], "succeeded")
        self.assertTrue(Path(completed.json()["report_path"]).joinpath("summary.json").exists())

        restarted = PostgresRepository(
            os.environ["TEST_DATABASE_URL"], Path("migrations")
        )
        self.assertEqual(restarted.get_job(job_id).status, "succeeded")


if __name__ == "__main__":
    unittest.main()
