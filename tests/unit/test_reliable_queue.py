import unittest
from unittest.mock import MagicMock, patch

from intern_rag.worker import QueueUnavailable, RedisJobQueue


class RedisStreamsQueueTests(unittest.TestCase):
    def _queue(self):
        client = MagicMock()
        with patch("redis.Redis.from_url", return_value=client):
            queue = RedisJobQueue(
                "redis://localhost:6379/0", "jobs:test", "workers:test"
            )
        return queue, client

    def test_enqueue_dequeue_ack_use_stream_consumer_group(self) -> None:
        queue, client = self._queue()
        client.xreadgroup.return_value = [
            ("jobs:test", [("1-0", {"job_id": "job-1"})])
        ]

        queue.enqueue("job-1")
        message = queue.dequeue("worker-a", timeout_seconds=2)
        queue.ack(message.message_id)

        client.xgroup_create.assert_called_once_with(
            "jobs:test", "workers:test", id="0-0", mkstream=True
        )
        client.xadd.assert_called_once_with("jobs:test", {"job_id": "job-1"})
        client.xreadgroup.assert_called_once_with(
            "workers:test", "worker-a", {"jobs:test": ">"}, count=1, block=2000
        )
        client.xack.assert_called_once_with("jobs:test", "workers:test", "1-0")

    def test_reclaim_uses_xautoclaim_and_returns_message_identity(self) -> None:
        queue, client = self._queue()
        client.xautoclaim.return_value = [
            "0-0", [("7-0", {"job_id": "job-7"})], []
        ]

        messages = queue.reclaim_stale("worker-b", min_idle_ms=60_000, count=3)

        self.assertEqual(messages[0].message_id, "7-0")
        self.assertEqual(messages[0].job_id, "job-7")
        client.xautoclaim.assert_called_once_with(
            "jobs:test", "workers:test", "worker-b", 60_000,
            start_id="0-0", count=3,
        )

    def test_existing_group_is_idempotent_and_redis_error_is_controlled(self) -> None:
        queue, client = self._queue()
        client.xgroup_create.side_effect = RuntimeError("BUSYGROUP already exists")
        client.xadd.side_effect = RuntimeError("connection reset")

        with self.assertRaisesRegex(QueueUnavailable, "enqueue"):
            queue.enqueue("job-1")

        self.assertTrue(queue._group_ready)


if __name__ == "__main__":
    unittest.main()
