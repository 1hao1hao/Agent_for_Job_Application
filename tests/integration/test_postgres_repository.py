import os
from pathlib import Path
import unittest
from uuid import uuid4

from intern_rag.agent import Citation, RagRequest, RagResponse
from intern_rag.persistence import EvaluationRunRecord, PostgresRepository
from intern_rag.routing import RouteDecision
from intern_rag.tracing import build_agent_trace


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "TEST_DATABASE_URL not set")
class PostgresRepositoryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = PostgresRepository(
            os.environ["TEST_DATABASE_URL"], Path("migrations")
        )
        self.repository.initialize()

    def test_request_trace_job_run_idempotency_and_restart_visibility(self) -> None:
        request = RagRequest("分析岗位", request_id="postgres-request", retriever="bm25")
        response = RagResponse(
            request_id=request.request_id,
            trace_id="postgres-trace",
            answer="回答",
            citations=[Citation("c1", "data/a.md", "jd", "岗位", 1, 1.0)],
            routed_sources=["jd"],
            status="answered",
            latency_ms=1.0,
        )
        trace = build_agent_trace(
            request.query,
            RouteDecision("analyze_jd", ["jd"], []),
            [],
            {"total": 1.0},
            request_id=request.request_id,
            trace_id=response.trace_id,
        )
        self.repository.save_trace(trace)
        self.repository.save_request(request, response)
        self.assertEqual(self.repository.get_trace("postgres-trace").request_id, request.request_id)

        idempotency_key = f"postgres-{uuid4()}"
        job, created = self.repository.create_job(
            dataset_version="evalrag_v0.2",
            split="dev",
            run_config={"retriever_config_path": "configs/retrieval/bm25_v0.2.json"},
            idempotency_key=idempotency_key,
            max_retries=1,
        )
        duplicate, duplicate_created = self.repository.create_job(
            dataset_version="evalrag_v0.2",
            split="dev",
            run_config={},
            idempotency_key=idempotency_key,
            max_retries=1,
        )
        self.assertTrue(created)
        self.assertFalse(duplicate_created)
        self.assertEqual(job.job_id, duplicate.job_id)

        running = self.repository.mark_job_running(job.job_id)
        self.assertEqual(running.attempt_count, 1)
        run_id = f"postgres-run-{uuid4()}"
        self.repository.save_run(
            EvaluationRunRecord(
                run_id,
                job.job_id,
                running.run_config,
                {"case_count": 1},
                "reports/runs/postgres-run",
            )
        )
        self.repository.mark_job_succeeded(job.job_id, "reports/runs/postgres-run")

        restarted_repository = PostgresRepository(
            os.environ["TEST_DATABASE_URL"], Path("migrations")
        )
        self.assertEqual(restarted_repository.get_job(job.job_id).status, "succeeded")


if __name__ == "__main__":
    unittest.main()
