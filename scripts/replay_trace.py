#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from intern_rag.persistence import PostgresRepository
from intern_rag.runtime import SavedRun, replay_saved_run
from intern_rag.serving.runtime import build_runtime_for_replay


def main() -> int:
    """从 PostgreSQL 恢复历史快照，执行离线 Replay 并输出结构化结果。"""

    parser = argparse.ArgumentParser(description="Replay one persisted EvalRAG trace")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument(
        "--stage",
        default="full",
        choices=["full", "routing", "retrieval", "evidence", "context", "generation", "validation"],
    )
    parser.add_argument("--project-root", default=os.environ.get("EVALRAG_PROJECT_ROOT", "."))
    args = parser.parse_args()
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print(json.dumps({"replayable": False, "reason": "DATABASE_URL is not configured"}))
        return 2
    root = Path(args.project_root).resolve()
    repository = PostgresRepository(database_url, root / "migrations")
    pair = repository.get_request_response(args.trace_id)
    trace = repository.get_trace(args.trace_id)
    if pair is None or trace is None:
        print(json.dumps({"replayable": False, "reason": "persisted request or trace snapshot missing"}))
        return 2
    try:
        saved = SavedRun.from_persisted(pair[0], pair[1], trace)
    except ValueError as error:
        print(json.dumps({"replayable": False, "reason": str(error)}, ensure_ascii=False))
        return 2
    result = replay_saved_run(saved, build_runtime_for_replay, stage=args.stage)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.replayable and result.matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
