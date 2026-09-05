from __future__ import annotations

import logging
import os
from pathlib import Path
from time import sleep

from intern_rag.persistence import PostgresRepository
from intern_rag.worker import (
    EvaluationWorker,
    QueueUnavailable,
    RedisJobQueue,
    SubprocessEvaluationExecutor,
)
from intern_rag.runtime import AgentRuntime, JsonlSpanSink


LOGGER = logging.getLogger(__name__)


def main() -> int:
    """初始化真实 adapter，并持续消费或 reclaim Redis Stream 消息。"""

    project_root = Path(os.environ.get("EVALRAG_PROJECT_ROOT", ".")).resolve()
    repository = PostgresRepository(
        os.environ["DATABASE_URL"], project_root / "migrations"
    )
    repository.initialize()
    queue = RedisJobQueue(os.environ["REDIS_URL"])
    evaluation_timeout_seconds = int(
        os.environ.get("EVALUATION_TIMEOUT_SECONDS", "1800")
    )
    worker = EvaluationWorker(
        repository,
        queue,
        SubprocessEvaluationExecutor(
            project_root,
            timeout_seconds=evaluation_timeout_seconds,
        ),
        runtime=AgentRuntime(
            span_sinks=(
                JsonlSpanSink(project_root / "traces/service/runtime_spans.jsonl"),
            )
        ),
        consumer_name=os.environ.get("EVALUATION_WORKER_NAME"),
        stale_min_idle_ms=int(
            os.environ.get(
                "EVALUATION_STALE_MIN_IDLE_MS",
                str((evaluation_timeout_seconds + 60) * 1000),
            )
        ),
    )
    while True:
        try:
            worker.run_once(timeout_seconds=5)
        except QueueUnavailable:
            # 临时 Redis 故障不 ACK；短暂退避后 pending/new message 均可继续消费。
            LOGGER.warning("evaluation queue unavailable; retrying")
            sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
