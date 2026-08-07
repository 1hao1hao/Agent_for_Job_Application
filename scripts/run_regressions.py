from __future__ import annotations

import json
from pathlib import Path

from intern_rag.evaluation import load_regression_cases, run_regression_suite
from intern_rag.routing import route_query


def main() -> int:
    """运行 fixed Regression，保存结果；open 只计数不计入通过率。"""

    cases = load_regression_cases(Path("tests/regression/cases_v0.2.jsonl"))
    result = run_regression_suite(cases, {"route": _route_handler})
    report_dir = Path("reports/regression/p0-d5-v0.2")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "case_results.jsonl").open("w", encoding="utf-8") as output:
        for case_result in result.case_results:
            output.write(json.dumps(case_result, ensure_ascii=False) + "\n")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.fixed_pass_rate == 1.0 else 1


def _route_handler(query: str) -> dict[str, object]:
    decision = route_query(query)
    return {"intent": decision.intent, "sources": decision.routed_sources}


if __name__ == "__main__":
    raise SystemExit(main())
