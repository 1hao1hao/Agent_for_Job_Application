from __future__ import annotations

from datetime import datetime, timezone
import unittest

from intern_rag.agent import ConversationMessage, MemoryItem, ProfileFact, UserProfile
from intern_rag.persistence import SessionMemoryService
from tests.support import InMemoryPersistenceRepository


class FakeCache:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.values = {}

    def get(self, user_id, session_id):
        if self.fail:
            raise RuntimeError("redis unavailable")
        return self.values.get((user_id, session_id))

    def set(self, user_id, session_id, messages):
        if self.fail:
            raise RuntimeError("redis unavailable")
        self.values[(user_id, session_id)] = messages

    def invalidate(self, user_id, session_id):
        if self.fail:
            raise RuntimeError("redis unavailable")
        self.values.pop((user_id, session_id), None)


class FakeMemoryExtractor:
    def __init__(self, memory: MemoryItem) -> None:
        self.memory = memory

    def extract(self, user_id, session_id, messages):
        del user_id, session_id, messages
        return [self.memory]


class SessionMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryPersistenceRepository()
        self.cache = FakeCache()
        self.service = SessionMemoryService(self.repository, self.cache)
        self.session = self.service.create_session("u1", "求职准备")
        self.now = datetime.now(timezone.utc).isoformat()

    def test_cache_miss_and_redis_failure_fall_back_to_repository(self) -> None:
        message = ConversationMessage("m1", self.session.session_id, "u1", "user", "优先广州", self.now)
        self.service.append_message(message)
        loaded = self.service.load_context("u1", self.session.session_id)
        self.assertEqual(loaded.messages, (message,))
        self.assertEqual(loaded.history_source, "postgres")

        failed_cache_service = SessionMemoryService(self.repository, FakeCache(fail=True))
        loaded_again = failed_cache_service.load_context("u1", self.session.session_id)
        self.assertEqual(loaded_again.messages, (message,))
        self.assertEqual(loaded_again.history_source, "postgres")

    def test_cross_user_access_is_rejected(self) -> None:
        with self.assertRaises(PermissionError):
            self.service.load_context("u2", self.session.session_id)

    def test_profile_version_and_unconfirmed_fact_are_guarded(self) -> None:
        profile = UserProfile("u1", (ProfileFact("城市", "广州", "explicit"),), 0, "")
        saved = self.service.update_profile(profile, 0)
        self.assertEqual(saved.version, 1)
        with self.assertRaises(ValueError):
            self.service.update_profile(profile, 0)
        with self.assertRaises(ValueError):
            self.service.update_profile(
                UserProfile("u1", (ProfileFact("技能", "Go", "model", False),), 1, ""), 1
            )

    def test_memory_scope_version_and_delete(self) -> None:
        item = MemoryItem("mem1", "u1", "preference", "优先广州", "confirmed_chat", 0.9, self.now)
        self.service.add_memory(item)
        self.assertEqual([value.memory_id for value in self.repository.list_memories("u1")], ["mem1"])
        self.assertEqual(self.repository.list_memories("u2"), [])
        self.assertTrue(self.repository.delete_memory("u1", "mem1"))
        self.assertEqual(self.repository.list_memories("u1"), [])
        with self.assertRaises(ValueError):
            self.service.add_memory(
                MemoryItem("bad", "u1", "fact", "未确认", "model", 1.0, self.now, confirmed=False)
            )

    def test_memory_dedup_keeps_conflicting_confirmed_facts(self) -> None:
        first = MemoryItem("city-1", "u1", "preference", "优先广州", "chat-1", 0.8, self.now)
        duplicate = MemoryItem("city-2", "u1", "preference", "  优先广州 ", "chat-2", 0.9, self.now)
        conflict = MemoryItem("city-3", "u1", "preference", "优先深圳", "chat-3", 0.9, self.now)

        self.assertTrue(self.service.add_memory(first))
        self.assertFalse(self.service.add_memory(duplicate))
        self.assertTrue(self.service.add_memory(conflict))
        self.assertEqual(
            {item.memory_id for item in self.repository.list_memories("u1")},
            {"city-1", "city-3"},
        )

    def test_confirmed_memory_extraction_checks_scope_before_write(self) -> None:
        valid = MemoryItem(
            "mem-confirmed", "u1", "experience", "做过 RAG 评测",
            "confirmed_chat", 0.8, self.now, session_id=self.session.session_id,
        )
        extracted = self.service.extract_confirmed_memories(
            "u1", self.session.session_id, [], FakeMemoryExtractor(valid)
        )
        self.assertEqual(extracted, [valid])
        self.assertEqual(self.repository.list_memories("u1"), [valid])

        cross_user = MemoryItem(
            "bad-scope", "u2", "fact", "跨用户内容", "confirmed_chat", 0.8,
            self.now, session_id=self.session.session_id,
        )
        with self.assertRaises(PermissionError):
            self.service.extract_confirmed_memories(
                "u1", self.session.session_id, [], FakeMemoryExtractor(cross_user)
            )


if __name__ == "__main__":
    unittest.main()
