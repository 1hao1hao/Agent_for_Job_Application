from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import ConversationMessage, MemoryItem, ProfileFact, UserProfile  # noqa: E402
from intern_rag.persistence import (  # noqa: E402
    PostgresRepository,
    RedisRecentHistoryCache,
    SessionMemoryService,
)


STATE_PATH = Path("/tmp/evalrag-p1-d5-context-state.json")


def main() -> int:
    """创建或复查 P1-D5 持久化状态，并可验证 Redis 故障时回源 PostgreSQL。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expect-cache-fallback", action="store_true")
    args = parser.parse_args()
    repository = PostgresRepository(os.environ["DATABASE_URL"], ROOT / "migrations")
    repository.initialize()
    cache = RedisRecentHistoryCache(os.environ["REDIS_URL"], ttl_seconds=300)
    service = SessionMemoryService(repository, cache)
    if args.verify_only:
        _verify(service, args.expect_cache_fallback)
    else:
        _create(service, repository)
    return 0


def _create(service: SessionMemoryService, repository: PostgresRepository) -> None:
    """写入 Session、Message、Summary、Profile 和带向量的确认 Memory。"""

    now = datetime.now(timezone.utc)
    user_id = "p1-d5-ci-user"
    session = service.create_session(user_id, "P1-D5 persistence validation")
    message = ConversationMessage(
        "p1-d5-ci-message",
        session.session_id,
        user_id,
        "user",
        "已确认优先广州岗位。",
        now.isoformat(),
    )
    service.append_message(message)
    repository.save_summary(user_id, session.session_id, "用户确认优先广州岗位。", 1)
    service.update_profile(
        UserProfile(
            user_id,
            (ProfileFact("城市", "广州", "ci_confirmed"),),
            0,
            now.isoformat(),
        ),
        0,
    )
    service.add_memory(
        MemoryItem(
            "p1-d5-ci-memory",
            user_id,
            "preference",
            "优先广州岗位",
            "confirmed_ci_message",
            0.9,
            now.isoformat(),
            session_id=session.session_id,
            expires_at=(now + timedelta(days=30)).isoformat(),
        ),
        [1.0] + [0.0] * 511,
    )
    STATE_PATH.write_text(
        json.dumps({"user_id": user_id, "session_id": session.session_id}),
        encoding="utf-8",
    )
    print(json.dumps({"created": True, "session_id": session.session_id}))


def _verify(service: SessionMemoryService, expect_cache_fallback: bool) -> None:
    """重连后读取全部状态；Redis 不可用时还断言 history 来自 PostgreSQL。"""

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    context = service.load_context(state["user_id"], state["session_id"])
    assert context.session.session_id == state["session_id"]
    assert context.messages and context.messages[0].message_id == "p1-d5-ci-message"
    assert context.summary == "用户确认优先广州岗位。"
    assert context.profile is not None and context.profile.facts[0].value == "广州"
    assert any(item.memory_id == "p1-d5-ci-memory" for item in context.memories)
    if expect_cache_fallback:
        assert context.history_source == "postgres", context.history_source
    print(
        json.dumps(
            {
                "verified": True,
                "history_source": context.history_source,
                "memory_count": len(context.memories),
            }
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
