from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Iterable, Mapping

from intern_rag.evaluation.knowledge_dataset import KnowledgeEvaluationCase


@dataclass(frozen=True)
class ThresholdPoint:
    """阈值扫描中的一个混淆矩阵与派生指标。"""

    threshold: float
    true_accept: int
    true_reject: int
    false_accept: int
    false_reject: int
    far: float
    frr: float
    precision: float
    recall: float


@dataclass(frozen=True)
class CalibrationResult:
    """单个 Retriever 的分布、完整曲线和最终 operating point。"""

    retriever: str
    split: str
    case_count: int
    positive_count: int
    negative_count: int
    positive_stats: dict[str, float | int | None]
    negative_stats: dict[str, float | int | None]
    curve: tuple[ThresholdPoint, ...]
    enabled: bool
    threshold: float | None
    status: str
    operating_point: dict[str, float | int] | None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["curve"] = [asdict(point) for point in self.curve]
        return value


def calibrate_score_gate(
    retriever: str,
    cases: Iterable[KnowledgeEvaluationCase],
    case_results: Iterable[Mapping[str, object]],
    *,
    far_target: float = 0.05,
    max_frr: float = 0.25,
) -> CalibrationResult:
    """使用 dev prediction 校准 top-1 score gate。

    函数拒绝任何 test Case。正标签表示当前 top-5 已满足该问题的 gold/结构证据：
    普通问题需命中 gold，多来源需覆盖全部来源，关系问题还需覆盖标注边且路径
    有效；不可回答与 hard negative 为负标签。随后扫描全部候选阈值，并按 FAR
    预算、最低 FRR、最低 FAR、较低阈值的顺序选择 operating point。
    """

    case_by_id = {case.case_id: case for case in cases}
    if any(case.split != "dev" for case in case_by_id.values()):
        raise ValueError("calibration only accepts dev cases")
    records: list[tuple[bool, float | None]] = []
    for row in case_results:
        case_id = str(row["case_id"])
        case = case_by_id.get(case_id)
        if case is None:
            continue
        if str(row.get("split", "dev")) != "dev":
            raise ValueError("calibration result contains non-dev prediction")
        predicted = row.get("predicted", {})
        retrieved = list(dict(predicted).get("retrieved", [])) if isinstance(predicted, Mapping) else []
        score = float(dict(retrieved[0])["score"]) if retrieved else None
        records.append((is_structurally_positive(case, retrieved[:5]), score))

    positives = [score for label, score in records if label and score is not None]
    negatives = [score for label, score in records if not label and score is not None]
    thresholds = _candidate_thresholds([score for _, score in records if score is not None])
    curve = tuple(_evaluate_threshold(records, value) for value in thresholds)
    feasible = [point for point in curve if point.far <= far_target]
    selected = min(feasible, key=lambda p: (p.frr, p.far, p.threshold)) if feasible else None
    enabled = bool(selected and selected.frr <= max_frr and positives and negatives)
    status = "calibrated" if enabled else "raw_score_not_separable"
    return CalibrationResult(
        retriever=retriever,
        split="dev",
        case_count=len(records),
        positive_count=sum(label for label, _ in records),
        negative_count=sum(not label for label, _ in records),
        positive_stats=_distribution(positives),
        negative_stats=_distribution(negatives),
        curve=curve,
        enabled=enabled,
        threshold=selected.threshold if enabled and selected else None,
        status=status,
        operating_point=asdict(selected) if selected else None,
    )


def is_structurally_positive(
    case: KnowledgeEvaluationCase,
    retrieved: list[object],
) -> bool:
    if not case.answerable or case.category in {"unanswerable", "hard_negative"}:
        return False
    rows = [dict(item) for item in retrieved]
    retrieved_ids = {str(item.get("chunk_id", "")) for item in rows}
    gold_hits = retrieved_ids & set(case.relevant_chunk_ids)
    if not gold_hits:
        return False
    if case.category == "cross_source":
        # relevant Chunk 本身已按来源标注，全部 gold 命中等价于覆盖标注来源。
        return set(case.relevant_chunk_ids).issubset(retrieved_ids)
    if case.category in {"two_hop", "three_hop"}:
        predicted_edges: set[str] = set()
        valid_path = False
        for row in rows:
            details = dict(row.get("details", {}))
            edge_value = str(details.get("graph_edge_ids", ""))
            predicted_edges.update(edge for edge in edge_value.split("|") if edge)
            valid_path = valid_path or bool(details.get("path_valid"))
        return valid_path and set(case.graph_edge_ids).issubset(predicted_edges)
    return True


def _candidate_thresholds(scores: list[float]) -> list[float]:
    if not scores:
        return []
    unique = sorted(set(scores))
    epsilon = max(1e-12, (unique[-1] - unique[0]) * 1e-9)
    return [unique[0] - epsilon, *unique, unique[-1] + epsilon]


def _evaluate_threshold(
    records: list[tuple[bool, float | None]], threshold: float
) -> ThresholdPoint:
    ta = tr = fa = fr = 0
    for positive, score in records:
        accepted = score is not None and score >= threshold
        if positive and accepted:
            ta += 1
        elif positive:
            fr += 1
        elif accepted:
            fa += 1
        else:
            tr += 1
    return ThresholdPoint(
        threshold=threshold, true_accept=ta, true_reject=tr,
        false_accept=fa, false_reject=fr,
        far=fa / (fa + tr) if fa + tr else 0.0,
        frr=fr / (ta + fr) if ta + fr else 0.0,
        precision=ta / (ta + fa) if ta + fa else 0.0,
        recall=ta / (ta + fr) if ta + fr else 0.0,
    )


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None,
                "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    ordered = sorted(values)
    return {
        "count": len(values), "min": ordered[0], "max": ordered[-1],
        "mean": mean(ordered), "median": median(ordered),
        "p10": _percentile(ordered, 0.10), "p25": _percentile(ordered, 0.25),
        "p50": _percentile(ordered, 0.50), "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
    }


def _percentile(values: list[float], fraction: float) -> float:
    index = max(0, min(len(values) - 1, int(len(values) * fraction + 0.999999) - 1))
    return values[index]
