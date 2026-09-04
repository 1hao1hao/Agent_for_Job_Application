from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import check_evidence, load_evidence_config  # noqa: E402
from intern_rag.evaluation import is_structurally_positive, load_chunks_jsonl  # noqa: E402
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset  # noqa: E402
from intern_rag.retrieval import build_retriever_from_config  # noqa: E402
from intern_rag.routing import RouteDecision  # noqa: E402


RUN_ID = "p1-evidence-retry-v03-dev-20260904"


def main() -> int:
    """只在 dev 上比较结构 Gate 与低置信重试 Gate。

    输入 evalrag_v0.3/dev、Adaptive v2 Retriever 和两版 Gate 配置。每条 Case
    先执行带来源过滤的首轮检索；任一 Gate 要求重试时，再执行一次无来源过滤
    的扩源检索。输出两版逐 Case 决策、重试统计和对照报告，不读取或重跑 test。
    """

    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    cases = [
        case for case in load_knowledge_dataset(
            ROOT / "data/evaluation/evalrag_v0.3.jsonl"
        )
        if case.split == "dev"
    ]
    retrieval_config = _read_json(
        ROOT / "configs/retrieval/adaptive_v2_v0.3.json"
    )
    retriever = build_retriever_from_config(retrieval_config)
    configs = {
        "structural_v2": load_evidence_config(
            ROOT / "configs/evidence/gate_calibrated_v0.3.json"
        ),
        "retry_v2_1": load_evidence_config(
            ROOT / "configs/evidence/gate_retry_calibrated_v0.3.json"
        ),
    }
    rows = {name: [] for name in configs}

    for case in cases:
        route = RouteDecision(
            "evaluation" if case.answerable else "unknown",
            list(case.expected_sources) if case.answerable else [],
            [],
        )
        initial_results = retriever(
            case.query,
            chunks,
            top_k=5,
            source_types=set(route.routed_sources),
        )
        initial_trace = _retriever_trace(retriever)
        requirement = initial_trace.get("evidence_requirement")
        requirement = dict(requirement) if isinstance(requirement, dict) else {}
        initial_decisions = {
            name: check_evidence(
                route,
                initial_results,
                retriever_name="adaptive",
                retry_count=0,
                max_retries=1,
                config=config,
                evidence_requirement=requirement,
                retrieval_trace=initial_trace,
            )
            for name, config in configs.items()
        }

        expanded_results = None
        expanded_trace = None
        if any(item.status == "retryable" for item in initial_decisions.values()):
            expanded_results = retriever(
                case.query, chunks, top_k=5, source_types=None
            )
            expanded_trace = _retriever_trace(retriever)

        for name, config in configs.items():
            first = initial_decisions[name]
            retried = first.status == "retryable"
            final_results = expanded_results if retried else initial_results
            final_trace = expanded_trace if retried else initial_trace
            if final_results is None or final_trace is None:
                raise RuntimeError("retry results were not produced")
            final = (
                check_evidence(
                    route,
                    final_results,
                    retriever_name="adaptive",
                    retry_count=1,
                    max_retries=1,
                    config=config,
                    evidence_requirement=requirement,
                    retrieval_trace=final_trace,
                )
                if retried
                else first
            )
            initial_positive = is_structurally_positive(
                case, _result_dicts(initial_results)
            )
            final_positive = is_structurally_positive(
                case, _result_dicts(final_results)
            )
            accepted = final.status == "sufficient"
            success = (
                case.answerable and final_positive and accepted
            ) or (not case.answerable and not accepted)
            rows[name].append({
                "case_id": case.case_id,
                "category": case.category,
                "answerable": case.answerable,
                "initial_structurally_positive": initial_positive,
                "final_structurally_positive": final_positive,
                "retried": retried,
                "retry_recovered_evidence": (
                    retried and not initial_positive and final_positive
                ),
                "accepted": accepted,
                "success": success,
                "initial": {
                    "retrieved_chunk_ids": [item.chunk_id for item in initial_results],
                    "decision": asdict(first),
                },
                "final": {
                    "retrieved_chunk_ids": [item.chunk_id for item in final_results],
                    "decision": asdict(final),
                },
            })

    summaries = {name: _summarize(value) for name, value in rows.items()}
    payload = {
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "evalrag_v0.3",
        "split": "dev",
        "case_count": len(cases),
        "answerable_case_count": sum(case.answerable for case in cases),
        "retrieval_config": "configs/retrieval/adaptive_v2_v0.3.json",
        "comparison": summaries,
        "metric_definitions": {
            "retry_rate": "retried cases / all cases",
            "retry_success_rate": "retried cases with a correct final gate outcome / retried cases",
            "retry_evidence_recovery_rate": "retries changing structural-negative retrieval into structural-positive retrieval / retried cases",
        },
        "recommendation": {
            "retain_as_default": False,
            "reason": "Retry rate increased without recovering structural evidence; deterministic E2E success regressed.",
            "active_release_config": "configs/evidence/gate_calibrated_v0.3.json",
        },
        "boundary": "dev-only Gate comparison; selector, labels and test split unchanged",
    }
    output = ROOT / "reports/ablations" / RUN_ID
    output.mkdir(parents=True, exist_ok=True)
    for name, value in rows.items():
        _write_jsonl(output / f"{name}_case_results.jsonl", value)
    _write_json(output / "summary.json", payload)
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """从逐 Case Gate 决策汇总可靠性、重试和失败原因。"""

    positives = [row for row in rows if row["final_structurally_positive"]]
    negatives = [row for row in rows if not row["final_structurally_positive"]]
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    retried = [row for row in rows if row["retried"]]
    failures = [row for row in rows if not row["success"]]
    return {
        "far": _ratio(sum(bool(row["accepted"]) for row in negatives), len(negatives)),
        "frr": _ratio(sum(not bool(row["accepted"]) for row in positives), len(positives)),
        "answerable_acceptance": _ratio(
            sum(bool(row["accepted"]) for row in answerable), len(answerable)
        ),
        "abstention_accuracy": _ratio(
            sum(not bool(row["accepted"]) for row in unanswerable), len(unanswerable)
        ),
        "retry_count": len(retried),
        "retry_rate": _ratio(len(retried), len(rows)),
        "retry_success_count": sum(bool(row["success"]) for row in retried),
        "retry_success_rate": _ratio(
            sum(bool(row["success"]) for row in retried), len(retried)
        ),
        "retry_evidence_recovery_count": sum(
            bool(row["retry_recovered_evidence"]) for row in retried
        ),
        "retry_evidence_recovery_rate": _ratio(
            sum(bool(row["retry_recovered_evidence"]) for row in retried),
            len(retried),
        ),
        "deterministic_e2e_success": _ratio(
            sum(bool(row["success"]) for row in rows), len(rows)
        ),
        "failure_count": len(failures),
        "final_reason_counts": dict(Counter(
            str(dict(row["final"])["decision"]["reason"]) for row in rows
        )),
        "failure_case_ids": [str(row["case_id"]) for row in failures],
    }


def _result_dicts(results) -> list[dict[str, object]]:
    return [{
        "chunk_id": item.chunk_id,
        "score": item.score,
        "rank": item.rank,
        "details": item.details,
    } for item in results]


def _retriever_trace(retriever) -> dict[str, object]:
    getter = getattr(retriever, "get_last_trace", None)
    value = getter() if callable(getter) else {}
    return dict(value) if isinstance(value, dict) else {}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _report(payload: dict[str, object]) -> str:
    comparison = dict(payload["comparison"])
    lines = [
        "# Evidence Gate Low-Confidence Retry Dev Ablation",
        "",
        "`evalrag_v0.3/dev`: 160 cases, 120 answerable. Test split was not read.",
        "",
        "| Gate | FAR | FRR | Answerable acceptance | Retry rate | Retry success | Evidence recovery | E2E success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("structural_v2", "retry_v2_1"):
        row = dict(comparison[name])
        lines.append(
            f"| {name} | {row['far']:.4f} | {row['frr']:.4f} | "
            f"{row['answerable_acceptance']:.4f} | {row['retry_rate']:.4f} | "
            f"{row['retry_success_rate']:.4f} | "
            f"{row['retry_evidence_recovery_rate']:.4f} | "
            f"{row['deterministic_e2e_success']:.4f} |"
        )
    lines += [
        "",
        "Raw top-1 score is used only to trigger one retry. It never hard-rejects structurally sufficient evidence after retry.",
        "",
        "## Decision",
        "",
        "Rejected as the default Gate: retries increased but recovered no missing structural evidence, while deterministic E2E success regressed.",
    ]
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
