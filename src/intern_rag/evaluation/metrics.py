from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean
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


def calculate_ndcg_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """计算二值相关标签下的 NDCG@k，衡量正确证据是否排在前面。"""

    if k <= 0 or not relevant_chunk_ids:
        return 0.0
    relevant_ids = set(relevant_chunk_ids)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_count = min(k, len(relevant_ids))
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


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


def calculate_citation_validity(
    citation_ids: list[str],
    context_ids: list[str],
    *,
    status: str,
) -> float | None:
    """计算回答引用 ID 在本轮 Context 中存在的比例。"""

    if status != "answered":
        return None
    if not citation_ids:
        return 0.0
    valid_count = sum(chunk_id in set(context_ids) for chunk_id in citation_ids)
    return valid_count / len(citation_ids)


def calculate_key_point_coverage(
    answer: str,
    expected_points: list[str],
) -> tuple[float | None, list[str]]:
    """用规范化关键词匹配计算要点覆盖率，并返回已覆盖要点。"""

    if not expected_points:
        return None, []
    normalized_answer = _normalize_text(answer)
    covered = [
        point
        for point in expected_points
        if _normalize_text(point) in normalized_answer
    ]
    return len(covered) / len(expected_points), covered


def summarize_end_to_end_results(
    case_results: list[dict[str, object]],
    *,
    key_point_threshold: float,
) -> dict[str, object]:
    """按协议汇总 Grounding、Abstention、Safety 与端到端成功率。"""

    if not 0.0 <= key_point_threshold <= 1.0:
        raise ValueError("key_point_threshold must be between 0 and 1")
    evaluated = [
        _evaluate_end_to_end_case(result, key_point_threshold)
        for result in case_results
    ]
    answerable = [item for item in evaluated if bool(item["answerable"])]
    unanswerable = [item for item in evaluated if not bool(item["answerable"])]
    answered = [item for item in evaluated if item["status"] == "answered"]
    unsupported_reviewed = [
        item for item in answered if item["unsupported_answer"] is not None
    ]
    e2e_known = [item for item in evaluated if item["end_to_end_success"] is not None]

    return {
        "case_metrics": evaluated,
        "metrics": {
            "citation_validity": _mean_optional(
                item["citation_validity"] for item in answerable
            ),
            "key_point_coverage": _mean_optional(
                item["key_point_coverage"] for item in answerable
            ),
            "abstention_accuracy": (
                mean(float(bool(item["correct_abstention"])) for item in unanswerable)
                if unanswerable else None
            ),
            "unsupported_answer_rate": (
                mean(
                    float(bool(item["unsupported_answer"]))
                    for item in unsupported_reviewed
                )
                if unsupported_reviewed else None
            ),
            "end_to_end_success_rate": (
                mean(float(bool(item["end_to_end_success"])) for item in e2e_known)
                if len(e2e_known) == len(evaluated) and evaluated else None
            ),
        },
        "counts": {
            "total": len(evaluated),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "answered": len(answered),
            "unexpected_abstention": sum(
                bool(item["unexpected_abstention"]) for item in evaluated
            ),
            "should_abstain_but_answered": sum(
                bool(item["should_abstain_but_answered"]) for item in evaluated
            ),
            "unsupported_reviewed": len(unsupported_reviewed),
        },
        "category_metrics": _summarize_categories(evaluated),
        "key_point_threshold": key_point_threshold,
        "metric_formulas": {
            "citation_validity": "valid citation ids / all returned citation ids",
            "key_point_coverage": "matched expected points / all expected points",
            "abstention_accuracy": "correct abstentions / all unanswerable cases",

            "unsupported_answer_rate": "reviewed unsupported answers / reviewed answered cases",#编造不存在证据的回答比例

            #   分子：满足全部成功条件的 Case 数量 “整个 RAG 请求是否在路由、检索、回答、引用和拒答方面都达到要求。”
            #   分母：全部可计算端到端结果的 EvaluationCase 数量
            "end_to_end_success_rate": "successful cases / all evaluated cases",
        },
    }


def _evaluate_end_to_end_case(
    result: dict[str, object],
    key_point_threshold: float,
) -> dict[str, object]:
    answerable = bool(result["answerable"])
    status = str(result["status"])
    citation_validity = result.get("citation_validity")
    key_point_coverage = result.get("key_point_coverage")
    unsupported = result.get("unsupported_answer")
    correct_abstention = not answerable and status == "insufficient_evidence"
    unexpected_abstention = answerable and status != "answered"
    should_abstain_but_answered = not answerable and status == "answered"

    if answerable:
        if status != "answered":
            # 可回答样例被拒答是明确失败，不依赖答案支持性标签。
            success: bool | None = False
        elif unsupported is None:
            success = None
        else:
            success = (
                bool(result["router_correct"])
                and float(result["recall_at_5"] or 0.0) > 0.0
                and citation_validity == 1.0
                and key_point_coverage is not None
                and float(key_point_coverage) >= key_point_threshold
                and unsupported is False
            )
    else:
        success = (
            correct_abstention
            and not list(result["citation_ids"])  # type: ignore[arg-type]
            and unsupported is not True
        )
    return {
        **result,
        "correct_abstention": correct_abstention,
        "unexpected_abstention": unexpected_abstention,
        "should_abstain_but_answered": should_abstain_but_answered,
        "end_to_end_success": success,
    }


def _summarize_categories(
    evaluated: list[dict[str, object]],
) -> dict[str, dict[str, float | int | None]]:
    categories = sorted({str(item["category"]) for item in evaluated})
    summary: dict[str, dict[str, float | int | None]] = {}
    for category in categories:
        items = [item for item in evaluated if item["category"] == category]
        successes = [item["end_to_end_success"] for item in items]
        summary[category] = {
            "case_count": len(items),
            "citation_validity": _mean_optional(
                item["citation_validity"] for item in items
            ),
            "key_point_coverage": _mean_optional(
                item["key_point_coverage"] for item in items
            ),
            "end_to_end_success_rate": (
                mean(float(bool(value)) for value in successes)
                if all(value is not None for value in successes) and successes
                else None
            ),
        }
    return summary


def _mean_optional(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return mean(numbers) if numbers else None


def _normalize_text(text: str) -> str:
    return "".join(text.casefold().split())


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
