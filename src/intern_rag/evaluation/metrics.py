from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalEvalCase:
    """一条检索评测样例。

    relevant_chunk_ids 是人工标注的正确证据；retrieved_chunk_ids 是系统实际
    返回的结果顺序。Recall@k 会检查前 k 个结果覆盖了多少正确证据。
    """

    query: str
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]


@dataclass(frozen=True)
class RouterEvalCase:
    """一条路由评测样例。

    expected_* 是人工标注结果；predicted_* 是当前 router 的实际输出。
    第一版 Router Accuracy 要求 intent 和 sources 都匹配才算正确。
    """

    query: str
    expected_intent: str
    predicted_intent: str
    expected_sources: list[str]
    predicted_sources: list[str]


@dataclass(frozen=True)
class EvaluationReport:
    """基础评测报告。"""

    recall_at_k: float
    router_accuracy: float
    retrieval_case_count: int
    router_case_count: int
    k: int

    def to_dict(self) -> dict[str, float | int]:
        """转换成普通 dict，方便 demo 打印或写入 JSON。"""

        return asdict(self)


def calculate_recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """计算单条样例的 Recall@k（前 k 召回率）。

    公式：前 k 个检索结果命中的 relevant chunk 数量 / relevant chunk 总数。
    如果没有人工标注的 relevant chunk，返回 0.0，避免伪造高分。
    """

    if k <= 0 or not relevant_chunk_ids:
        return 0.0

    top_k_chunk_ids = set(retrieved_chunk_ids[:k])
    relevant_ids = set(relevant_chunk_ids)
    matched_count = len(top_k_chunk_ids & relevant_ids)
    return matched_count / len(relevant_ids)


def calculate_average_recall_at_k(
    cases: list[RetrievalEvalCase],
    k: int,
) -> float:
    """计算多条检索样例的平均 Recall@k。"""

    if not cases:
        return 0.0

    total_recall = sum(
        calculate_recall_at_k(
            case.retrieved_chunk_ids,
            case.relevant_chunk_ids,
            k,
        )
        for case in cases
    )
    return total_recall / len(cases)


def calculate_router_accuracy(cases: list[RouterEvalCase]) -> float:
    """计算 Router Accuracy（路由准确率）。

    第一版采用严格口径：predicted intent 与 expected intent 相同，并且
    predicted sources 与 expected sources 集合相同，才算该 query 路由正确。
    """

    if not cases:
        return 0.0

    correct_count = 0
    for case in cases:
        intent_is_correct = case.predicted_intent == case.expected_intent
        sources_are_correct = set(case.predicted_sources) == set(case.expected_sources)
        if intent_is_correct and sources_are_correct:
            correct_count += 1

    return correct_count / len(cases)


def evaluate_cases(
    retrieval_cases: list[RetrievalEvalCase],
    router_cases: list[RouterEvalCase],
    k: int,
) -> EvaluationReport:
    """对检索和路由样例运行基础评测。"""

    return EvaluationReport(
        recall_at_k=calculate_average_recall_at_k(retrieval_cases, k),
        router_accuracy=calculate_router_accuracy(router_cases),
        retrieval_case_count=len(retrieval_cases),
        router_case_count=len(router_cases),
        k=k,
    )


def load_evaluation_cases(
    path: Path,
) -> tuple[list[RetrievalEvalCase], list[RouterEvalCase]]:
    """从 JSON 文件读取最小评测样例。

    文件格式包含 `retrieval_cases` 和 `router_cases` 两个列表。这里不执行
    系统流程，只读取已记录的 expected/predicted 字段用于指标计算。
    """

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    retrieval_cases = [
        _retrieval_case_from_dict(raw_case)
        for raw_case in raw_data.get("retrieval_cases", [])
    ]
    router_cases = [
        _router_case_from_dict(raw_case)
        for raw_case in raw_data.get("router_cases", [])
    ]
    return retrieval_cases, router_cases


def _retrieval_case_from_dict(raw_case: dict[str, Any]) -> RetrievalEvalCase:
    """把原始 dict 转成 RetrievalEvalCase。"""

    return RetrievalEvalCase(
        query=str(raw_case["query"]),
        relevant_chunk_ids=list(raw_case["relevant_chunk_ids"]),
        retrieved_chunk_ids=list(raw_case["retrieved_chunk_ids"]),
    )


def _router_case_from_dict(raw_case: dict[str, Any]) -> RouterEvalCase:
    """把原始 dict 转成 RouterEvalCase。"""

    return RouterEvalCase(
        query=str(raw_case["query"]),
        expected_intent=str(raw_case["expected_intent"]),
        predicted_intent=str(raw_case["predicted_intent"]),
        expected_sources=list(raw_case["expected_sources"]),
        predicted_sources=list(raw_case["predicted_sources"]),
    )
