from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MetricGate:
    """一项 CI 指标约束；quality 不得明显下降，latency 不得超比例增长。"""

    name: str
    direction: str
    max_drop: float = 0.0
    max_increase_ratio: float = 1.0


@dataclass(frozen=True)
class EvaluationGateResult:
    """CI Evaluation Gate 的结构化结果。"""

    passed: bool
    checks: list[dict[str, object]]
    failed_case_ids: list[str]
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_ci_gate(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    gates: Sequence[MetricGate],
    *,
    fixed_regression_pass_rate: float,
    failed_case_ids: Sequence[str] = (),
) -> EvaluationGateResult:
    """比较版本化 reference 与候选指标，并把回归失败并入发布结论。

    输入为同一数据版本上的 reference/candidate 指标和 fixed regression 结果；
    质量指标检查最大允许下降，延迟指标检查最大增长比例。缺失指标直接失败，
    返回逐项检查、失败 Case ID 和原因，供 CI 保存为工件。
    """

    checks: list[dict[str, object]] = []
    reasons: list[str] = []
    for gate in gates:
        old = reference.get(gate.name)
        new = candidate.get(gate.name)
        if old is None or new is None:
            passed = False
            reason = f"missing metric: {gate.name}"
        elif gate.direction == "higher_is_better":
            passed = new >= old - gate.max_drop
            reason = "" if passed else f"{gate.name} dropped from {old:.6f} to {new:.6f}"
        elif gate.direction == "lower_is_better":
            limit = old * gate.max_increase_ratio
            passed = new <= limit
            reason = "" if passed else f"{gate.name} increased from {old:.6f} to {new:.6f}"
        else:
            raise ValueError(f"unsupported metric direction: {gate.direction}")
        checks.append({
            "metric": gate.name,
            "reference": old,
            "candidate": new,
            "passed": passed,
            "reason": reason,
        })
        if reason:
            reasons.append(reason)

    regression_passed = fixed_regression_pass_rate == 1.0
    checks.append({
        "metric": "fixed_regression_pass_rate",
        "reference": 1.0,
        "candidate": fixed_regression_pass_rate,
        "passed": regression_passed,
        "reason": "" if regression_passed else "fixed regression did not fully pass",
    })
    if not regression_passed:
        reasons.append("fixed regression did not fully pass")
    return EvaluationGateResult(
        passed=all(bool(item["passed"]) for item in checks),
        checks=checks,
        failed_case_ids=sorted(set(failed_case_ids)),
        reasons=reasons,
    )
