from __future__ import annotations

import json
from typing import Protocol, Sequence

from intern_rag.agent.context_engine import ConversationMessage, MemoryItem, UserProfile
from intern_rag.persistence.base import PersistenceRepository
from intern_rag.persistence.models import SessionContext, SessionRecord


class RecentHistoryCache(Protocol):
    """Redis 与 Fake Cache 共用的最小最近历史接口。"""

    def get(self, user_id: str, session_id: str) -> list[ConversationMessage] | None: ...
    def set(self, user_id: str, session_id: str, messages: list[ConversationMessage]) -> None: ...
    def invalidate(self, user_id: str, session_id: str) -> None: ...


class MemoryExtractor(Protocol):
    """从已经人工确认的对话中提取候选长期记忆。"""

    def extract(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[ConversationMessage],
    ) -> list[MemoryItem]:
        """返回带来源、类型和重要性的 MemoryItem，不直接写稳定 Profile。"""


class MemoryEmbeddingProvider(Protocol):
    """把 Memory 文本编码成与 pgvector schema 一致的向量。"""

    def encode_one(self, text: str) -> list[float]: ...


class RedisRecentHistoryCache:
    """Redis 只缓存最近消息；缓存故障由 SessionMemoryService 回源 PostgreSQL。"""

    def __init__(self, redis_url: str, *, ttl_seconds: int = 1800, prefix: str = "evalrag:history") -> None:
        import redis

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{self.prefix}:{user_id}:{session_id}"

    def get(self, user_id: str, session_id: str) -> list[ConversationMessage] | None:
        value = self.client.get(self._key(user_id, session_id))
        if value is None:
            return None
        return [ConversationMessage(**item) for item in json.loads(value)]

    def set(self, user_id: str, session_id: str, messages: list[ConversationMessage]) -> None:
        self.client.setex(
            self._key(user_id, session_id),
            self.ttl_seconds,
            json.dumps([item.__dict__ for item in messages], ensure_ascii=False),
        )

    def invalidate(self, user_id: str, session_id: str) -> None:
        self.client.delete(self._key(user_id, session_id))


class SessionMemoryService:
    """统一 Session/Profile/History/Memory 读取，并提供 Redis 故障回退。

    Service 先校验 session 属于 user，再尝试 Redis 最近历史；cache miss 或 Redis 异常
    都回源 repository。Profile 与 Memory 始终读取 PostgreSQL，避免缓存成为事实来源。
    """

    def __init__(
        self,
        repository: PersistenceRepository,
        cache: RecentHistoryCache | None = None,
        memory_embedder: MemoryEmbeddingProvider | None = None,
    ) -> None:
        self.repository = repository
        self.cache = cache
        self.memory_embedder = memory_embedder

    def load_context(
        self,
        user_id: str,
        session_id: str,
        *,
        history_limit: int = 50,
        memory_query_embedding: list[float] | None = None,
    ) -> SessionContext:
        session = self.repository.get_session(user_id, session_id)
        if session is None:
            raise PermissionError("session does not exist or belongs to another user")
        messages: list[ConversationMessage] | None = None
        source = "postgres"
        if self.cache is not None:
            try:
                messages = self.cache.get(user_id, session_id)
                if messages is not None:
                    source = "redis"
            except Exception:
                messages = None
        if messages is None:
            messages = self.repository.list_messages(user_id, session_id, history_limit)
            if self.cache is not None:
                try:
                    self.cache.set(user_id, session_id, messages)
                except Exception:
                    pass
        memories = (
            self.repository.search_memories(user_id, memory_query_embedding)
            if memory_query_embedding is not None
            else self.repository.list_memories(user_id)
        )
        return SessionContext(
            session=session,
            profile=self.repository.get_profile(user_id),
            messages=tuple(messages),
            summary=self.repository.get_summary(user_id, session_id),
            memories=tuple(memories),
            history_source=source,  # type: ignore[arg-type]
        )

    def append_message(self, message: ConversationMessage) -> None:
        session = self.repository.get_session(message.user_id, message.session_id)
        if session is None:
            raise PermissionError("session does not exist or belongs to another user")
        self.repository.append_message(message)
        if self.cache is not None:
            try:
                self.cache.invalidate(message.user_id, message.session_id)
            except Exception:
                pass

    def update_profile(self, profile: UserProfile, expected_version: int | None) -> UserProfile:
        """只接受显式调用；摘要和未确认 Memory 不会触发此方法。"""

        if any(not fact.confirmed for fact in profile.facts):
            raise ValueError("profile only accepts confirmed facts")
        return self.repository.upsert_profile(profile, expected_version)

    def add_memory(self, item: MemoryItem, embedding: list[float] | None = None) -> bool:
        """去除同用户同类型的完全重复记忆；冲突内容使用不同 ID 并存。"""

        if not item.confirmed:
            raise ValueError("unconfirmed memory must not be persisted")
        normalized = " ".join(item.content.lower().split())
        for existing in self.repository.list_memories(item.user_id, limit=1000):
            if (
                existing.memory_type == item.memory_type
                and " ".join(existing.content.lower().split()) == normalized
                and existing.memory_id != item.memory_id
            ):
                return False
        if embedding is None and self.memory_embedder is not None:
            embedding = self.memory_embedder.encode_one(item.content)
        self.repository.save_memory(item, embedding)
        return True

    def extract_confirmed_memories(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[ConversationMessage],
        extractor: MemoryExtractor,
    ) -> list[MemoryItem]:
        """调用可注入提取器并保存已确认记忆。

        输入必须来自已经确认的对话。方法再次校验 user/session scope、确认状态和
        extractor 返回的归属；任一候选不合法时受控失败，不会写入 Profile。
        """

        if self.repository.get_session(user_id, session_id) is None:
            raise PermissionError("session does not exist or belongs to another user")
        extracted = extractor.extract(user_id, session_id, messages)
        persisted: list[MemoryItem] = []
        for item in extracted:
            if item.user_id != user_id or item.session_id not in {None, session_id}:
                raise PermissionError("extracted memory crosses user or session scope")
            if self.add_memory(item):
                persisted.append(item)
        return persisted

    def create_session(self, user_id: str, title: str) -> SessionRecord:
        return self.repository.create_session(user_id, title)
