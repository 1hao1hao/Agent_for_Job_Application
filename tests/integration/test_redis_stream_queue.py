import os
import unittest
from uuid import uuid4

from intern_rag.worker import RedisJobQueue


@unittest.skipUnless(os.environ.get("TEST_REDIS_URL"), "TEST_REDIS_URL not set")
class RedisStreamQueueIntegrationTests(unittest.TestCase):
    def test_pending_message_can_be_reclaimed_and_acked(self) -> None:
        suffix = str(uuid4())
        stream = f"evalrag:test:stream:{suffix}"
        group = f"evalrag:test:group:{suffix}"
        queue = RedisJobQueue(os.environ["TEST_REDIS_URL"], stream, group)
        try:
            queue.enqueue("job-real-redis")
            first = queue.dequeue("worker-a", timeout_seconds=0)
            self.assertIsNotNone(first)
            self.assertEqual(queue.client.xpending(stream, group)["pending"], 1)

            reclaimed = queue.reclaim_stale("worker-b", min_idle_ms=0)
            self.assertEqual(reclaimed[0].message_id, first.message_id)
            queue.ack(reclaimed[0].message_id)

            self.assertEqual(queue.client.xpending(stream, group)["pending"], 0)
        finally:
            queue.client.delete(stream)


if __name__ == "__main__":
    unittest.main()
