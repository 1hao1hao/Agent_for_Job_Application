from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

from intern_rag.routing.base import Router
from intern_rag.routing.intent_router import Intent, RouteDecision

if TYPE_CHECKING:
    from intern_rag.evaluation.dataset import EvaluationCase


@dataclass(frozen=True)
class RouterFeedback:
    """一条可审计误路由反馈，不直接修改线上 Router。"""

    feedback_id: str
    query: str
    original_intent: str
    original_sources: tuple[str, ...]
    corrected_intent: Intent
    corrected_sources: tuple[str, ...]
    failure_type: str
    router_version: str
    source: Literal["evaluation", "user_confirmed"]
    created_at: str


class JsonlRouterFeedbackStore:
    """版本化 JSONL Feedback Store。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, feedback: RouterFeedback) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(asdict(feedback), ensure_ascii=False) + "\n")

    def read_all(self) -> list[RouterFeedback]:
        if not self.path.exists():
            return []
        values = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item["original_sources"] = tuple(item["original_sources"])
            item["corrected_sources"] = tuple(item["corrected_sources"])
            values.append(RouterFeedback(**item))
        return values


class FeedbackRouter:
    """从已确认误路由提取短意图锚点，其余 Query 委托给基础 Router。

    该层由离线审核后的 feedback 构建，不在请求期间学习；只匹配“岗位分析：”这类
    明确前缀，避免把共享长正文记入单一意图。发布前仍需运行完整 shadow gate。
    """

    def __init__(self, base_router: Router, feedback: Sequence[RouterFeedback]) -> None:
        self.base_router = base_router
        self.corrections: dict[str, RouterFeedback] = {}
        for item in feedback:
            anchor = _feedback_prototype(item.query)
            existing = self.corrections.get(anchor)
            if existing is not None and (
                existing.corrected_intent != item.corrected_intent
                or existing.corrected_sources != item.corrected_sources
            ):
                raise ValueError(f"conflicting router feedback anchor: {anchor}")
            self.corrections[anchor] = item

    def __call__(self, query: str) -> RouteDecision:
        normalized = query.strip()
        matched_anchor = next(
            (
                anchor for anchor in sorted(self.corrections, key=len, reverse=True)
                if normalized == anchor
                or normalized.startswith(f"{anchor}：")
                or normalized.startswith(f"{anchor}:")
            ),
            None,
        )
        item = self.corrections.get(matched_anchor or "")
        if item is None:
            return self.base_router(query)
        return RouteDecision(
            intent=item.corrected_intent,
            routed_sources=list(item.corrected_sources),
            matched_keywords=[],
            strategy="hybrid",
            confidence=1.0,
            reason="confirmed_feedback_anchor_override",
            details={
                "feedback_id": item.feedback_id,
                "feedback_anchor": matched_anchor,
                "parent_router_version": item.router_version,
            },
        )


@dataclass(frozen=True)
class RouterVersion:
    """Router 配置、feedback dataset 和 shadow 报告的发布记录。"""

    version: str
    router_name: str
    config: dict[str, object]
    feedback_dataset: str
    report_path: str
    parent_version: str | None
    created_at: str


class RouterVersionRegistry:
    """持久化 active/history，失败候选不会污染 active，支持显式回滚。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, object]:
        if not self.path.exists():
            return {"active_version": None, "versions": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def publish(self, version: RouterVersion, *, gate_passed: bool) -> None:
        if not gate_passed:
            raise ValueError("router candidate did not pass shadow gate")
        state = self.load()
        versions = list(state["versions"])
        if any(item["version"] == version.version for item in versions):
            raise ValueError("router version already exists")
        versions.append(asdict(version))
        self._save({"active_version": version.version, "versions": versions})

    def rollback(self, version: str) -> None:
        state = self.load()
        if not any(item["version"] == version for item in state["versions"]):
            raise ValueError("router version does not exist")
        state["active_version"] = version
        self._save(state)

    def _save(self, state: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prototypes_from_feedback(
    base: Mapping[Intent, Sequence[str]], feedback: Sequence[RouterFeedback]
) -> dict[Intent, tuple[str, ...]]:
    """离线提取已确认 Query 的短意图锚点，追加为正确 intent prototype。

    评测 Query 常采用“意图锚点：主题正文”的形式。直接记忆整句会把跨意图复用的
    长正文也学入 prototype，造成反馈污染；因此优先保留冒号前的短锚点。没有明确
    锚点时才使用完整 Query，候选仍需经过 shadow gate 才能发布。
    """

    output = {intent: list(values) for intent, values in base.items()}
    for item in feedback:
        prototype = _feedback_prototype(item.query)
        if prototype not in output[item.corrected_intent]:
            output[item.corrected_intent].append(prototype)
    return {intent: tuple(values) for intent, values in output.items()}


def _feedback_prototype(query: str) -> str:
    for separator in ("：", ":"):
        prefix, found, _ = query.partition(separator)
        if found and 2 <= len(prefix.strip()) <= 16:
            return prefix.strip()
    return query.strip()


def evaluate_router_shadow(
    router: Router, cases: Sequence[EvaluationCase]
) -> dict[str, object]:
    """在同一 reference 集输出 Accuracy、unknown precision/recall、P50/P95 和逐 Case。"""

    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    for case in cases:
        started = perf_counter()
        decision = router(case.query)
        latency = (perf_counter() - started) * 1000
        latencies.append(latency)
        correct = (
            decision.intent == case.expected_intent
            and set(decision.routed_sources) == set(case.expected_sources)
        )
        rows.append({
            "case_id": case.case_id,
            "query": case.query,
            "expected_intent": case.expected_intent,
            "expected_sources": case.expected_sources,
            "predicted_intent": decision.intent,
            "predicted_sources": decision.routed_sources,
            "correct": correct,
            "latency_ms": latency,
            "reason": decision.reason,
        })
    predicted_unknown = [row for row in rows if row["predicted_intent"] == "unknown"]
    expected_unknown = [row for row in rows if row["expected_intent"] == "unknown"]
    true_unknown = [row for row in rows if row["predicted_intent"] == row["expected_intent"] == "unknown"]
    ordered = sorted(latencies)
    return {
        "case_count": len(rows),
        "accuracy": mean(float(row["correct"]) for row in rows) if rows else 0.0,
        "unknown_precision": len(true_unknown) / len(predicted_unknown) if predicted_unknown else 0.0,
        "unknown_recall": len(true_unknown) / len(expected_unknown) if expected_unknown else 0.0,
        "latency_ms": {
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
        },
        "cases": rows,
    }


def compare_router_versions(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    max_accuracy_drop: float = 0.0,
    max_unknown_drop: float = 0.0,
    max_p95_ratio: float = 1.25,
) -> dict[str, object]:
    """执行 shadow gate，并列出改善、退化和漂移 Case。"""

    base_cases = {item["case_id"]: item for item in baseline["cases"]}  # type: ignore[index]
    candidate_cases = {item["case_id"]: item for item in candidate["cases"]}  # type: ignore[index]
    differences = []
    for case_id, old in base_cases.items():
        new = candidate_cases[case_id]
        if (old["predicted_intent"], old["predicted_sources"]) != (
            new["predicted_intent"], new["predicted_sources"]
        ):
            differences.append({
                "case_id": case_id,
                "before": old,
                "after": new,
                "change": "improved" if new["correct"] and not old["correct"] else "regressed" if old["correct"] and not new["correct"] else "drift",
            })
    checks = {
        "accuracy": float(candidate["accuracy"]) >= float(baseline["accuracy"]) - max_accuracy_drop,
        "unknown_precision": float(candidate["unknown_precision"]) >= float(baseline["unknown_precision"]) - max_unknown_drop,
        "unknown_recall": float(candidate["unknown_recall"]) >= float(baseline["unknown_recall"]) - max_unknown_drop,
        "p95": float(candidate["latency_ms"]["p95"]) <= max(float(baseline["latency_ms"]["p95"]) * max_p95_ratio, 1.0),  # type: ignore[index]
        "no_case_regression": not any(item["change"] == "regressed" for item in differences),
    }
    return {"passed": all(checks.values()), "checks": checks, "differences": differences}


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return values[min(int((len(values) - 1) * quantile), len(values) - 1)]
