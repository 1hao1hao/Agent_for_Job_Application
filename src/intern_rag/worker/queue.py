from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class QueueMessage:
    """Consumer Group 交付的一条消息，ACK 使用 message_id。"""

    message_id: str
    job_id: str


class QueueUnavailable(RuntimeError):
    """Redis 操作失败时使用的稳定异常，不泄漏连接细节。"""


class JobQueue(Protocol):
    """API 与 Worker 共同依赖的可靠队列接口。"""

    def enqueue(self, job_id: str) -> None:
        """新增只包含 job id 的消息。"""

    def dequeue(
        self, consumer_name: str, timeout_seconds: int = 5
    ) -> QueueMessage | None:
        """读取新消息并放入 Consumer Group pending 列表。"""

    def ack(self, message_id: str) -> None:
        """确认消息对应任务的最终状态已经写入 PostgreSQL。"""

    def reclaim_stale(
        self, consumer_name: str, min_idle_ms: int, count: int = 1
    ) -> list[QueueMessage]:
        """把超时未 ACK 的 pending 消息转交给当前 consumer。"""

    def ping(self) -> bool:
        """检查队列服务是否可用。"""


class RedisJobQueue:
    """用 Redis Streams Consumer Group 提供 at-least-once Job 交付。

    Stream 只保存 ``job_id``。读取消息后它进入 Pending Entries List，只有
    Worker 确认 PostgreSQL 已写入 succeeded/failed 后才调用 ``ack``。
    """

    def __init__(
        self,
        redis_url: str,
        stream_name: str = "evalrag:evaluation_jobs",
        group_name: str = "evalrag:evaluation_workers",
    ) -> None:
        if not redis_url.strip() or not stream_name.strip() or not group_name.strip():
            raise ValueError("redis_url, stream_name and group_name must not be empty")
        from redis import Redis

        self.client = Redis.from_url(redis_url, decode_responses=True)
        self.stream_name = stream_name
        self.group_name = group_name
        self._group_ready = False

    def enqueue(self, job_id: str) -> None:
        """使用 XADD 追加消息；配置与最终状态仍只从 PostgreSQL 读取。"""

        if not job_id.strip():
            raise ValueError("job_id must not be empty")
        self._ensure_group()
        try:
            self.client.xadd(self.stream_name, {"job_id": job_id})
        except Exception as error:
            raise QueueUnavailable("failed to enqueue evaluation job") from error

    def dequeue(
        self, consumer_name: str, timeout_seconds: int = 5
    ) -> QueueMessage | None:
        """用 XREADGROUP 读取新消息，读取本身不会删除或 ACK。"""

        _validate_consumer(consumer_name)
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        self._ensure_group()
        options: dict[str, object] = {"count": 1}
        if timeout_seconds > 0:
            options["block"] = timeout_seconds * 1000
        try:
            rows = self.client.xreadgroup(
                self.group_name,
                consumer_name,
                {self.stream_name: ">"},
                **options,
            )
        except Exception as error:
            raise QueueUnavailable("failed to read evaluation queue") from error
        messages = _messages_from_stream_response(rows)
        return messages[0] if messages else None

    def ack(self, message_id: str) -> None:
        """使用 XACK 移出 pending；不执行 XDEL，保留 Stream 交付记录。"""

        if not message_id.strip():
            raise ValueError("message_id must not be empty")
        self._ensure_group()
        try:
            self.client.xack(self.stream_name, self.group_name, message_id)
        except Exception as error:
            raise QueueUnavailable("failed to acknowledge evaluation job") from error

    def reclaim_stale(
        self, consumer_name: str, min_idle_ms: int, count: int = 1
    ) -> list[QueueMessage]:
        """使用 XAUTOCLAIM 转移 stale pending 消息，供其他 Worker 恢复。"""

        _validate_consumer(consumer_name)
        if min_idle_ms < 0 or count <= 0:
            raise ValueError("min_idle_ms must be non-negative and count must be positive")
        self._ensure_group()
        try:
            response = self.client.xautoclaim(
                self.stream_name,
                self.group_name,
                consumer_name,
                min_idle_ms,
                start_id="0-0",
                count=count,
            )
        except Exception as error:
            raise QueueUnavailable("failed to reclaim stale evaluation jobs") from error
        rows = response[1] if len(response) > 1 else []
        return _messages_from_entries(rows)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def _ensure_group(self) -> None:
        """幂等创建 Consumer Group；从 0 开始接收创建前已有的消息。"""

        if self._group_ready:
            return
        try:
            self.client.xgroup_create(
                self.stream_name, self.group_name, id="0-0", mkstream=True
            )
        except Exception as error:
            if "BUSYGROUP" not in str(error):
                raise QueueUnavailable("failed to initialize evaluation queue") from error
        self._group_ready = True


def _messages_from_stream_response(rows: object) -> list[QueueMessage]:
    messages: list[QueueMessage] = []
    for _, entries in rows or []:  # type: ignore[union-attr]
        messages.extend(_messages_from_entries(entries))
    return messages


def _messages_from_entries(entries: object) -> list[QueueMessage]:
    messages: list[QueueMessage] = []
    for message_id, fields in entries or []:  # type: ignore[union-attr]
        job_id = fields.get("job_id") if isinstance(fields, dict) else None
        if job_id:
            messages.append(QueueMessage(str(message_id), str(job_id)))
    return messages


def _validate_consumer(consumer_name: str) -> None:
    if not consumer_name.strip():
        raise ValueError("consumer_name must not be empty")
