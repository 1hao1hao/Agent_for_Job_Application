from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import ContextEngine, ContextEngineConfig  # noqa: E402
from intern_rag.ingestion import Chunk  # noqa: E402
from intern_rag.retrieval import RetrievalResult  # noqa: E402


TARGET_CASES = {"v03_cross_source_010", "v03_cross_source_026"}


def main() -> int:
    """以旧 Trace 的相同候选和预算回放两个 context evidence drop。

    输入历史 E2E Trace 中的最终 RetrievalResult、required sources 和原预算；分别
    运行旧 memory-first 优先级与新 Evidence-first 优先级，输出实际进入 Prompt 的
    Chunk、来源覆盖和 token。该审计不调用 Retriever/LLM，也不读取 test 标签。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-path",
        default=(
            "reports/runs/p1-e2e-abstention-v03-dev-release-candidate-20261005/"
            "traces.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default="reports/ablations/p1-evidence-first-context-v03-dev-20261006",
    )
    args = parser.parse_args()
    rows = []
    for line in (ROOT / args.trace_path).read_text(encoding="utf-8").splitlines():
        trace = json.loads(line)
        if trace.get("request_id") in TARGET_CASES:
            rows.append(_audit_trace(trace))
    if {row["case_id"] for row in rows} != TARGET_CASES:
        raise ValueError("historical trace does not contain both target cases")

    summary = {
        "dataset_version": "evalrag_v0.3",
        "split": "dev",
        "source_trace": args.trace_path,
        "case_count": len(rows),
        "before_drop_count": sum(
            not row["before"]["required_sources_covered"] for row in rows
        ),
        "after_drop_count": sum(
            not row["after"]["required_sources_covered"] for row in rows
        ),
        "citation_whitelist_rule": "only evidence chunks actually kept in ManagedContext",
        "cases": rows,
    }
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence_drop_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "evidence_drop_audit.md").write_text(
        _report(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _audit_trace(trace: dict[str, object]) -> dict[str, object]:
    retrieval = dict(trace["retrieval"])
    requirement = dict(dict(retrieval["decision"])["evidence_requirement"])
    required_sources = tuple(str(item) for item in requirement["required_source_types"])
    results = [_to_result(item) for item in trace["retrieved_chunks"]]
    reserved = int(dict(trace["context"])["reserved_token_count"])
    return {
        "case_id": trace["request_id"],
        "query": trace["query"],
        "required_sources": list(required_sources),
        "before": _build(
            trace, results, required_sources, reserved, evidence_first=False
        ),
        "after": _build(
            trace, results, required_sources, reserved, evidence_first=True
        ),
    }


def _build(trace, results, required_sources, reserved, *, evidence_first):
    managed = ContextEngine().build(
        query=str(trace["query"]),
        system_prompt="仅依据提供的证据回答；证据不足时明确拒答。",
        retrieved_results=results,
        config=ContextEngineConfig(
            token_budget=1800,
            evidence_char_budget=4000,
            mode="no_memory",
            evidence_strategy="source_balanced",
            reserved_token_count=reserved,
            evidence_first=evidence_first,
        ),
        required_source_types=required_sources,
    )
    covered = set(managed.evidence.covered_source_types)
    return {
        "evidence_first": evidence_first,
        "used_chunk_ids": managed.evidence.used_chunk_ids,
        "covered_sources": sorted(covered),
        "required_sources_covered": set(required_sources) <= covered,
        "token_count": managed.token_count,
        "token_budget": managed.token_budget,
    }


def _to_result(raw: dict[str, object]) -> RetrievalResult:
    chunk = Chunk(
        id=str(raw["chunk_id"]),
        source_type=str(raw["source_type"]),
        source_path=str(raw["source_path"]),
        title=str(raw["title"]),
        text=str(raw["text"]),
        metadata=dict(raw["metadata"]),
    )
    return RetrievalResult(
        chunk.id,
        float(raw["score"]),
        int(raw["rank"]),
        chunk,
        str(raw.get("reason", "")),
        dict(raw.get("details", {})),
    )


def _report(summary: dict[str, object]) -> str:
    lines = [
        "# Evidence-first Context Audit",
        "",
        "- Dataset: `evalrag_v0.3/dev`.",
        "- Same candidates and token budget as the historical run.",
        "",
        "| Case | Required sources | Before covered | After covered | Before tokens | After tokens |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in summary["cases"]:
        lines.append(
            f"| {row['case_id']} | {', '.join(row['required_sources'])} | "
            f"{row['before']['required_sources_covered']} | "
            f"{row['after']['required_sources_covered']} | "
            f"{row['before']['token_count']} | {row['after']['token_count']} |"
        )
    lines.extend([
        "",
        f"Context evidence drops: {summary['before_drop_count']} -> {summary['after_drop_count']}.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
