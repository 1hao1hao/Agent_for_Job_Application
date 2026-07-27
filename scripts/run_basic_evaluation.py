from __future__ import annotations

import json
from pathlib import Path

from intern_rag.evaluation import evaluate_cases, load_evaluation_cases


def main() -> None:
    """运行最小评测 demo，并打印 JSON 格式报告。"""

    cases_path = Path("tests/fixtures/evaluation_cases.json")
    retrieval_cases, router_cases = load_evaluation_cases(cases_path)
    report = evaluate_cases(retrieval_cases, router_cases, k=2)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
