import unittest

from intern_rag.worker.evaluation_worker import (
    EvaluationExecutionResult,
    EvaluationWorker,
    WorkerExecutionError,
)
from intern_rag.worker import QueueUnavailable
from tests.support import InMemoryJobQueue, InMemoryPersistenceRepository
from intern_rag.runtime import AgentRuntime


class FakeExecutor:
    def __init__(self, error: WorkerExecutionError | None = None) -> None:
        self.error = error
        self.calls = 0

    def execute(self, job):
        self.calls += 1
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
        self.assertEqual(self.queue.pending, {})
        self.assertEqual(len(self.queue.acked_message_ids), 1)

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

    def test_crash_after_dequeue_keeps_pending_and_other_consumer_reclaims(self) -> None:
        job = _create_job(self.repository, self.queue)
        message = self.queue.dequeue("crashed-worker", 0)

        self.assertIsNotNone(message)
        self.assertIn(message.message_id, self.queue.pending)
        self.assertEqual(self.repository.get_job(job.job_id).status, "queued")

        executor = FakeExecutor()
        worker = EvaluationWorker(
            self.repository, self.queue, executor,
            consumer_name="recovery-worker", stale_min_idle_ms=0,
        )
        self.assertTrue(worker.run_once(0))
        self.assertEqual(executor.calls, 1)
        self.assertEqual(self.repository.get_job(job.job_id).status, "succeeded")
        self.assertEqual(self.queue.pending, {})

    def test_stale_running_job_is_recovered_and_executed_once(self) -> None:
        job = _create_job(self.repository, self.queue)
        message = self.queue.dequeue("crashed-worker", 0)
        self.repository.mark_job_running(job.job_id)
        executor = FakeExecutor()
        worker = EvaluationWorker(
            self.repository, self.queue, executor,
            consumer_name="recovery-worker", stale_min_idle_ms=0,
        )

        self.assertTrue(worker.run_once(0))

        self.assertEqual(self.repository.get_job(job.job_id).status, "succeeded")
        self.assertEqual(self.repository.get_job(job.job_id).attempt_count, 2)
        self.assertEqual(executor.calls, 1)
        self.assertNotIn(message.message_id, self.queue.pending)

    def test_empty_queue_returns_false(self) -> None:
        worker = EvaluationWorker(self.repository, self.queue, FakeExecutor())
        self.assertFalse(worker.run_once(timeout_seconds=0))

    def test_duplicate_delivery_does_not_execute_running_job_twice(self) -> None:
        job = _create_job(self.repository, self.queue)
        original = self.queue.dequeue("healthy-worker", 0)
        self.repository.mark_job_running(job.job_id)
        self.queue.enqueue(job.job_id)
        executor = FakeExecutor()
        worker = EvaluationWorker(
            self.repository, self.queue, executor,
            consumer_name="duplicate-worker", stale_min_idle_ms=999_999,
        )

        self.assertTrue(worker.run_once(0))
        self.assertEqual(executor.calls, 0)
        self.assertEqual(self.repository.get_job(job.job_id).status, "running")
        self.assertIn(original.message_id, self.queue.pending)
        self.assertEqual(len(self.queue.pending), 1)

    def test_succeeded_job_duplicate_is_acked_without_execution(self) -> None:
        job = _create_job(self.repository, self.queue)
        original = self.queue.dequeue("old-worker", 0)
        self.repository.mark_job_running(job.job_id)
        self.repository.mark_job_succeeded(job.job_id, "reports/runs/already-done")
        executor = FakeExecutor()
        worker = EvaluationWorker(
            self.repository, self.queue, executor,
            consumer_name="recovery-worker", stale_min_idle_ms=0,
        )

        self.assertTrue(worker.run_once(0))
        self.assertEqual(executor.calls, 0)
        self.assertNotIn(original.message_id, self.queue.pending)

    def test_reclaimed_running_job_with_exhausted_budget_fails_and_acks(self) -> None:
        job = _create_job(self.repository, self.queue, max_retries=0)
        message = self.queue.dequeue("crashed-worker", 0)
        self.repository.mark_job_running(job.job_id)
        executor = FakeExecutor()
        worker = EvaluationWorker(
            self.repository, self.queue, executor,
            consumer_name="recovery-worker", stale_min_idle_ms=0,
        )

        self.assertTrue(worker.run_once(0))
        failed = self.repository.get_job(job.job_id)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error_type, "retry_exhausted")
        self.assertEqual(executor.calls, 0)
        self.assertNotIn(message.message_id, self.queue.pending)

    def test_queue_unavailable_is_controlled_and_message_is_not_lost(self) -> None:
        job = _create_job(self.repository, self.queue)
        self.queue.available = False
        worker = EvaluationWorker(self.repository, self.queue, FakeExecutor())

        with self.assertRaises(QueueUnavailable):
            worker.run_once(0)

        self.assertEqual(self.repository.get_job(job.job_id).status, "queued")
        self.queue.available = True
        self.assertEqual(self.queue.job_ids, [job.job_id])

    def test_ack_failure_after_success_is_reclaimed_without_reexecution(self) -> None:
        class AckFailOnceQueue(InMemoryJobQueue):
            def __init__(self) -> None:
                super().__init__()
                self.fail_ack = True

            def ack(self, message_id: str) -> None:
                if self.fail_ack:
                    self.fail_ack = False
                    raise QueueUnavailable("ack unavailable")
                super().ack(message_id)

        queue = AckFailOnceQueue()
        job = _create_job(self.repository, queue)
        first_executor = FakeExecutor()
        first_worker = EvaluationWorker(
            self.repository, queue, first_executor, consumer_name="worker-a"
        )

        with self.assertRaises(QueueUnavailable):
            first_worker.run_once(0)
        self.assertEqual(self.repository.get_job(job.job_id).status, "succeeded")
        self.assertEqual(first_executor.calls, 1)
        self.assertEqual(len(queue.pending), 1)

        recovery_executor = FakeExecutor()
        recovery_worker = EvaluationWorker(
            self.repository, queue, recovery_executor,
            consumer_name="worker-b", stale_min_idle_ms=0,
        )
        self.assertTrue(recovery_worker.run_once(0))
        self.assertEqual(recovery_executor.calls, 0)
        self.assertEqual(queue.pending, {})


if __name__ == "__main__":
    unittest.main()
