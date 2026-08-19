from __future__ import annotations

import os
from pathlib import Path

from intern_rag.persistence import PostgresRepository
from intern_rag.worker import EvaluationWorker, RedisJobQueue, SubprocessEvaluationExecutor
from intern_rag.runtime import AgentRuntime, JsonlSpanSink


def main() -> int:
    """初始化真实 adapter，恢复中断任务并持续消费 Redis Queue。"""

    project_root = Path(os.environ.get("EVALRAG_PROJECT_ROOT", ".")).resolve()
    repository = PostgresRepository(
        os.environ["DATABASE_URL"], project_root / "migrations"
    )
    repository.initialize()
    queue = RedisJobQueue(os.environ["REDIS_URL"])
    worker = EvaluationWorker(
        repository,
        queue,
        SubprocessEvaluationExecutor(
            project_root,
            timeout_seconds=int(os.environ.get("EVALUATION_TIMEOUT_SECONDS", "1800")),
        ),
        runtime=AgentRuntime(
            span_sinks=(
                JsonlSpanSink(project_root / "traces/service/runtime_spans.jsonl"),
            )
        ),
    )
    worker.recover_interrupted()
    while True:
        worker.run_once(timeout_seconds=5)


if __name__ == "__main__":
    raise SystemExit(main())
