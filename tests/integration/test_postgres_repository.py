import os
from pathlib import Path
import unittest
from uuid import uuid4

from intern_rag.agent import (
    Citation,
    ConversationMessage,
    MemoryItem,
    ProfileFact,
    RagRequest,
    RagResponse,
    UserProfile,
)
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

    def test_session_profile_memory_survive_repository_restart(self) -> None:
        suffix = str(uuid4())
        user_id = f"postgres-user-{suffix}"
        session = self.repository.create_session(user_id, "求职准备")
        self.repository.append_message(
            ConversationMessage(
                f"message-{suffix}", session.session_id, user_id, "user",
                "优先广州岗位", "2026-08-16T00:00:00+00:00",
            )
        )
        self.repository.save_summary(user_id, session.session_id, "用户优先广州岗位", 1)
        profile = self.repository.upsert_profile(
            UserProfile(user_id, (ProfileFact("城市", "广州", "explicit"),), 0, ""), 0
        )
        self.repository.save_memory(
            MemoryItem(
                f"memory-{suffix}", user_id, "preference", "优先广州岗位",
                "confirmed_chat", 1.0, "2026-08-16T00:00:00+00:00",
            )
        )

        restarted = PostgresRepository(os.environ["TEST_DATABASE_URL"], Path("migrations"))
        self.assertEqual(restarted.get_session(user_id, session.session_id).user_id, user_id)
        self.assertEqual(restarted.list_messages(user_id, session.session_id)[0].content, "优先广州岗位")
        self.assertEqual(restarted.get_summary(user_id, session.session_id), "用户优先广州岗位")
        self.assertEqual(restarted.get_profile(user_id).version, profile.version)
        self.assertEqual(restarted.list_memories(user_id)[0].content, "优先广州岗位")
        self.assertIsNone(restarted.get_session("other-user", session.session_id))


if __name__ == "__main__":
    unittest.main()
