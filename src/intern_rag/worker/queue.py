from __future__ import annotations

from typing import Protocol


class JobQueue(Protocol):
    """API 与 Worker 共同依赖的最小队列接口。"""

    def enqueue(self, job_id: str) -> None:
        """把 job id 放到队尾。"""

    def dequeue(self, timeout_seconds: int = 5) -> str | None:
        """阻塞读取一个 job id，超时返回 None。"""

    def ping(self) -> bool:
        """检查队列服务是否可用。"""


class RedisJobQueue:
    """使用 Redis List 实现只传 job id 的轻量持久队列。"""

    def __init__(self, redis_url: str, queue_name: str = "evalrag:evaluation_jobs") -> None:
        if not redis_url.strip() or not queue_name.strip():
            raise ValueError("redis_url and queue_name must not be empty")
        from redis import Redis

        self.client = Redis.from_url(redis_url, decode_responses=True)
        self.queue_name = queue_name

    def enqueue(self, job_id: str) -> None:
        """只保存 job id；配置和最终状态始终从 PostgreSQL 读取。"""

        self.client.rpush(self.queue_name, job_id)

    def dequeue(self, timeout_seconds: int = 5) -> str | None:
        result = self.client.blpop(self.queue_name, timeout=timeout_seconds)
        return str(result[1]) if result is not None else None

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False
