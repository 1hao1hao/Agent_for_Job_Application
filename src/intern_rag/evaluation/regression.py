from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Literal, Mapping


RegressionStatus = Literal["open", "fixed"]
RegressionHandler = Callable[[str], dict[str, object]]


@dataclass(frozen=True)
class RegressionCase:
    """一个 open 或 fixed 失败样例及其可执行断言。"""

    case_id: str
    status: RegressionStatus
    failure_type: str
    query: str
    assertion: str
    expected: dict[str, object]
    source_case_id: str
    note: str


@dataclass(frozen=True)
class RegressionResult:
    """Regression Suite 的逐 Case 结果和固定用例通过率。"""

    case_results: list[dict[str, object]]
    fixed_count: int
    fixed_passed: int
    open_count: int

    @property
    def fixed_pass_rate(self) -> float:
        """open 不进入分母；没有 fixed 时返回 0。"""

        return self.fixed_passed / self.fixed_count if self.fixed_count else 0.0

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "fixed_pass_rate": self.fixed_pass_rate}


def load_regression_cases(path: Path) -> list[RegressionCase]:
    """读取版本化 JSONL Regression Case。"""

    cases: list[RegressionCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        case = RegressionCase(**payload)
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate regression case: {case.case_id}")
        if case.status not in {"open", "fixed"}:
            raise ValueError(f"{path}:{line_number} has invalid status")
        seen_ids.add(case.case_id)
        cases.append(case)
    return cases


def run_regression_suite(
    cases: list[RegressionCase],
    handlers: Mapping[str, RegressionHandler],
) -> RegressionResult:
    """只执行 fixed 断言，open 用例记录但不计入 pass rate。"""

    results: list[dict[str, object]] = []
    fixed_passed = 0
    for case in cases:
        if case.status == "open":
            results.append({
                "case_id": case.case_id,
                "status": "open",
                "failure_type": case.failure_type,
                "passed": None,
                "note": "open regression is not included in pass rate",
            })
            continue
        handler = handlers.get(case.assertion)
        if handler is None:
            actual: dict[str, object] = {}
            passed = False
            error = f"missing handler: {case.assertion}"
        else:
            try:
                actual = handler(case.query)
                passed = all(actual.get(key) == value for key, value in case.expected.items())
                error = "" if passed else "expected fields do not match"
            except Exception as exc:  # pragma: no cover - defensive artifact path
                actual = {}
                passed = False
                error = f"{type(exc).__name__}: {exc}"
        fixed_passed += int(passed)
        results.append({
            "case_id": case.case_id,
            "status": "fixed",
            "failure_type": case.failure_type,
            "passed": passed,
            "expected": case.expected,
            "actual": actual,
            "error": error,
        })
    fixed_count = sum(case.status == "fixed" for case in cases)
    return RegressionResult(
        case_results=results,
        fixed_count=fixed_count,
        fixed_passed=fixed_passed,
        open_count=sum(case.status == "open" for case in cases),
    )
