import unittest

from intern_rag.worker.evaluation_worker import (
    EvaluationExecutionResult,
    EvaluationWorker,
    WorkerExecutionError,
)
from tests.support import InMemoryJobQueue, InMemoryPersistenceRepository
from intern_rag.runtime import AgentRuntime


class FakeExecutor:
    def __init__(self, error: WorkerExecutionError | None = None) -> None:
        self.error = error

    def execute(self, job):
        if self.error is not None:
            raise self.error
        return EvaluationExecutionResult(
            run_id=f"run-{job.job_id}",
            config=job.run_config,
            summary={"case_count": 80},
            report_path=f"reports/runs/run-{job.job_id}",
        )


def _create_job(repository, queue, *, max_retries=1):
    job, _ = repository.create_job(
        dataset_version="evalrag_v0.2",
        split="dev",
        run_config={"retriever_config_path": "configs/retrieval/bm25_v0.2.json"},
        idempotency_key=None,
        max_retries=max_retries,
    )
    queue.enqueue(job.job_id)
    return job


class EvaluationWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryPersistenceRepository()
        self.queue = InMemoryJobQueue()

    def test_success_persists_run_and_final_status(self) -> None:
        job = _create_job(self.repository, self.queue)
        worker = EvaluationWorker(self.repository, self.queue, FakeExecutor())

        self.assertTrue(worker.run_once(timeout_seconds=0))

        completed = self.repository.get_job(job.job_id)
        self.assertEqual(completed.status, "succeeded")
        self.assertEqual(completed.attempt_count, 1)
        self.assertIn(f"run-{job.job_id}", self.repository.runs)

    def test_worker_uses_shared_runtime_span(self) -> None:
        class CollectingSink:
            def __init__(self) -> None:
                self.events = []

            def write(self, event) -> None:
                self.events.append(event)

        sink = CollectingSink()
        job = _create_job(self.repository, self.queue)
        worker = EvaluationWorker(
            self.repository, self.queue, FakeExecutor(),
            runtime=AgentRuntime(span_sinks=[sink]),
        )

        self.assertTrue(worker.run_once(timeout_seconds=0))
        self.assertEqual(self.repository.get_job(job.job_id).status, "succeeded")
        self.assertEqual(sink.events[0].name, "evaluation.worker")
        self.assertEqual(sink.events[0].attributes["entrypoint"], "worker")

    def test_timeout_failure_and_retry_budget(self) -> None:
        job = _create_job(self.repository, self.queue, max_retries=1)
        worker = EvaluationWorker(
            self.repository,
            self.queue,
            FakeExecutor(WorkerExecutionError("evaluation_timeout", "timeout")),
        )
        worker.run_once(timeout_seconds=0)
        failed = self.repository.get_job(job.job_id)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error_type, "evaluation_timeout")

        retried = self.repository.retry_failed_job(job.job_id)
        self.queue.enqueue(retried.job_id)
        worker.run_once(timeout_seconds=0)
        with self.assertRaisesRegex(ValueError, "retry budget"):
            self.repository.retry_failed_job(job.job_id)

    def test_restart_recovers_running_job_once(self) -> None:
        job = _create_job(self.repository, self.queue)
        self.queue.dequeue(0)
        self.repository.mark_job_running(job.job_id)
        worker = EvaluationWorker(self.repository, self.queue, FakeExecutor())

        recovered = worker.recover_interrupted()

        self.assertEqual(recovered, [job.job_id])
        self.assertEqual(self.queue.job_ids, [job.job_id])
        self.assertTrue(worker.run_once(0))
        self.assertEqual(self.repository.get_job(job.job_id).status, "succeeded")

    def test_empty_queue_returns_false(self) -> None:
        worker = EvaluationWorker(self.repository, self.queue, FakeExecutor())
        self.assertFalse(worker.run_once(timeout_seconds=0))

    def test_restart_marks_exhausted_running_job_failed(self) -> None:
        job = _create_job(self.repository, self.queue, max_retries=0)
        self.queue.dequeue(0)
        self.repository.mark_job_running(job.job_id)
        worker = EvaluationWorker(self.repository, self.queue, FakeExecutor())

        self.assertEqual(worker.recover_interrupted(), [])
        failed = self.repository.get_job(job.job_id)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error_type, "retry_exhausted")


if __name__ == "__main__":
    unittest.main()
