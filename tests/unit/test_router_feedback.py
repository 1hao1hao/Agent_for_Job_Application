from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from pathlib import Path

from intern_rag.routing import DEFAULT_INTENT_PROTOTYPES
from intern_rag.routing.factory import build_active_router_from_registry
from intern_rag.routing.intent_router import route_query
from intern_rag.routing.feedback import (
    FeedbackRouter,
    JsonlRouterFeedbackStore,
    RouterFeedback,
    RouterVersion,
    RouterVersionRegistry,
    compare_router_versions,
    prototypes_from_feedback,
)


class RouterFeedbackTests(unittest.TestCase):
    def test_feedback_is_versioned_and_deduplicates_prototype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlRouterFeedbackStore(Path(directory) / "feedback.jsonl")
            feedback = RouterFeedback(
                "f-1", "岗位分析：帮我分析职位", "unknown", (), "analyze_jd", ("jd",),
                "router_wrong", "hybrid-v1", "evaluation",
                datetime.now(timezone.utc).isoformat(),
            )
            store.append(feedback)
            updated = prototypes_from_feedback(DEFAULT_INTENT_PROTOTYPES, store.read_all() * 2)

            self.assertEqual(updated["analyze_jd"].count("岗位分析"), 1)

    def test_shadow_gate_publish_and_rollback(self) -> None:
        baseline = {
            "accuracy": 0.9, "unknown_precision": 1.0, "unknown_recall": 1.0,
            "latency_ms": {"p95": 10.0},
            "cases": [{"case_id": "c1", "correct": False, "predicted_intent": "unknown", "predicted_sources": []}],
        }
        candidate = {
            "accuracy": 1.0, "unknown_precision": 1.0, "unknown_recall": 1.0,
            "latency_ms": {"p95": 11.0},
            "cases": [{"case_id": "c1", "correct": True, "predicted_intent": "analyze_jd", "predicted_sources": ["jd"]}],
        }
        gate = compare_router_versions(baseline, candidate)
        with tempfile.TemporaryDirectory() as directory:
            registry = RouterVersionRegistry(Path(directory) / "registry.json")
            first = RouterVersion("v1", "hybrid", {}, "feedback-v1", "old.json", None, "2026-08-16")
            second = RouterVersion("v2", "hybrid", {}, "feedback-v1", "new.json", "v1", "2026-08-16")
            registry.publish(first, gate_passed=True)
            registry.publish(second, gate_passed=bool(gate["passed"]))
            registry.rollback("v1")

            self.assertTrue(gate["passed"])
            self.assertEqual(registry.load()["active_version"], "v1")
            with self.assertRaises(ValueError):
                registry.publish(second, gate_passed=False)

    def test_feedback_router_only_overrides_confirmed_query(self) -> None:
        feedback = RouterFeedback(
            "f-1", "确认问题", "unknown", (), "analyze_jd", ("jd",),
            "router_wrong", "v1", "evaluation", "2026-08-16",
        )

        router = FeedbackRouter(route_query, [feedback])

        self.assertEqual(router("确认问题").reason, "confirmed_feedback_anchor_override")
        self.assertEqual(router("今天吃什么").intent, "unknown")

    def test_factory_builds_active_feedback_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback_path = root / "feedback.jsonl"
            store = JsonlRouterFeedbackStore(feedback_path)
            store.append(RouterFeedback(
                "f-1", "确认问题", "unknown", (), "analyze_jd", ("jd",),
                "router_wrong", "v1", "evaluation", "2026-08-16",
            ))
            registry = RouterVersionRegistry(root / "registry.json")
            registry.publish(RouterVersion(
                "v2", "hybrid", {
                    "router_name": "hybrid",
                    "feedback_strategy": "confirmed_anchor_override",
                    "feedback_dataset": str(feedback_path),
                }, str(feedback_path), "report.json", None, "2026-08-16",
            ), gate_passed=True)

            with patch("intern_rag.routing.factory.build_router_from_config", return_value=route_query):
                router = build_active_router_from_registry(root / "registry.json")

            self.assertEqual(router("确认问题").intent, "analyze_jd")


if __name__ == "__main__":
    unittest.main()
