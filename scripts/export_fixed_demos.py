from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path(
    "reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash"
)
DEFAULT_OUTPUT_DIR = Path("examples/fixed_demos")
DEMO_CASES = {
    "single_source": "v02_single_022",
    "multi_source": "v02_multi_027",
    "abstention": "v02_unanswerable_021",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并忽略空行。"""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sanitize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """提取演示所需 Trace 字段，并移除原始 Chunk 正文和请求标识。

    输入是一条完整请求级 AgentTrace 字典；函数保留路由、检索排名、Evidence
    Gate、Context、生成、校验、尝试、延迟和 token 使用，返回可公开展示的精简
    字典。缺失的可选阶段保持为空，不把拒答误写为执行过生成。
    """

    retrieved_chunks = [
        {
            "chunk_id": item["chunk_id"],
            "rank": item["rank"],
            "score": item["score"],
            "source_type": item["source_type"],
            "source_path": item["source_path"],
            "reason": item.get("reason", ""),
        }
        for item in trace.get("retrieved_chunks", [])
    ]
    return {
        "query": trace.get("query", ""),
        "routing": trace.get("routing", {}),
        "retrieval": trace.get("retrieval", {}),
        "retrieved_chunks": retrieved_chunks,
        "evidence": trace.get("evidence", {}),
        "context": trace.get("context", {}),
        "generation": trace.get("generation", {}),
        "validation": trace.get("validation", {}),
        "attempts": trace.get("attempts", []),
        "response_status": trace.get("response_status", ""),
        "error_type": trace.get("error_type", "none"),
        "latency_ms": trace.get("latency_ms", {}),
        "token_usage": trace.get("token_usage", {}),
    }


def build_demo(
    case_result: dict[str, Any],
    trace: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """把一条保存的 prediction、Trace 和配置组合为固定 Demo。

    citation 通过 Trace 中的检索结果补全 source type/path；输出包含 Case、最终
    Response、可追溯 Citation、精简 Trace 和冻结配置。若回答为拒答，citation
    列表自然为空。
    """

    retrieved_by_id = {
        item["chunk_id"]: item for item in trace.get("retrieved_chunks", [])
    }
    citations = []
    for chunk_id in case_result.get("citation_ids", []):
        evidence = retrieved_by_id.get(chunk_id, {})
        citations.append(
            {
                "chunk_id": chunk_id,
                "source_type": evidence.get("source_type", "unknown"),
                "source_path": evidence.get("source_path", "unknown"),
            }
        )
    return {
        "case": {
            "case_id": case_result["case_id"],
            "category": case_result["category"],
            "query": case_result["query"],
            "answerable": case_result["answerable"],
        },
        "response": {
            "status": case_result["status"],
            "answer": case_result["answer"],
            "citations": citations,
            "error_type": case_result.get("error_type"),
        },
        "trace": sanitize_trace(trace),
        "config": run_config,
        "provenance": {
            "source_run": str(DEFAULT_RUN_DIR),
            "note": "由已保存 frozen prediction 导出，未重新调用 LLM。",
        },
    }


def export_fixed_demos(run_dir: Path, output_dir: Path) -> list[Path]:
    """从真实 frozen Run 导出单来源、多来源和拒答三个固定 Demo。

    函数读取保存的 case results、Trace 和 run config，按稳定 Case ID 配对，写出
    三个独立 JSON。Case 或 Trace 缺失时立即报错，避免静默生成不完整演示。
    """

    case_path = run_dir / "case_results_before_support_review.jsonl"
    if not case_path.exists():
        case_path = run_dir / "case_results.jsonl"
    cases = {row["case_id"]: row for row in read_jsonl(case_path)}
    traces = {row["request_id"]: row for row in read_jsonl(run_dir / "traces.jsonl")}
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for demo_name, case_id in DEMO_CASES.items():
        if case_id not in cases or case_id not in traces:
            raise KeyError(f"missing saved case or trace: {case_id}")
        demo = build_demo(cases[case_id], traces[case_id], run_config)
        output_path = output_dir / f"{demo_name}.json"
        output_path.write_text(
            json.dumps(demo, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written_paths.append(output_path)
    return written_paths


def main() -> int:
    """解析命令行参数并导出固定 Demo。"""

    parser = argparse.ArgumentParser(description="导出 EvalRAG 固定演示工件")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in export_fixed_demos(args.run_dir, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
