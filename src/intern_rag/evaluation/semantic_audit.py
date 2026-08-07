from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Callable

from intern_rag.evaluation.end_to_end import build_end_to_end_failures
from intern_rag.evaluation.grading import (
    ClaimJudgment,
    GroundingGrade,
    GroundingGrader,
    KeyPointGrade,
    KeyPointGrader,
    PointJudgment,
    grade_to_dict,
)
from intern_rag.evaluation.metrics import summarize_end_to_end_results


@dataclass(frozen=True)
class SemanticAuditConfig:
    """Saved Prediction 语义审核的模型、价格和版本配置。"""

    run_id: str
    dataset_version: str
    split: str
    source_prediction_run_id: str
    model: str
    temperature: float
    key_point_prompt_version: str
    grounding_prompt_version: str
    key_point_threshold: float
    input_cache_hit_usd_per_million: float
    input_cache_miss_usd_per_million: float
    output_usd_per_million: float
    pricing_source: str
    pricing_checked_at: str
    command: str
    grader_independence: str = "same_provider_model_not_independent_human"


@dataclass(frozen=True)
class SemanticAuditResult:
    """语义审核的逐 Case、逐 point、逐 claim 与汇总工件。"""

    config: SemanticAuditConfig
    summary: dict[str, object]
    case_results: list[dict[str, object]]
    point_verdicts: list[dict[str, object]]
    claim_verdicts: list[dict[str, object]]
    grader_calls: list[dict[str, object]]
    failures: list[dict[str, object]]
    comparison_cases: list[dict[str, object]]


def run_saved_prediction_audit(
    case_results: list[dict[str, object]],
    traces: list[dict[str, object]],
    config: SemanticAuditConfig,
    key_point_grader: KeyPointGrader,
    grounding_grader: GroundingGrader,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> SemanticAuditResult:
    """审核已保存 predictions，并用语义 verdict 重算端到端指标。

    输入必须是 P0-D5 原始 ``case_results_before_support_review`` 和同 Run 的 Trace；
    函数不会调用 Router、Retriever 或 Generator。
    每个 answered case 先做语义要点判断，再把 Answer 拆成 factual claims 并只对照实际 cited Chunk。
    随后将 semantic coverage 与 claim-level unsupported 标签写回 case 副本，调用现有 Metrics 汇总，
    最终返回可落盘的逐 point、逐 claim、失败和对照工件。

    Grader API、JSON 或证据 span 校验失败时保留 unknown/unavailable。
    此时 Unsupported Answer Rate 的审核分母会减少，完整 E2E 指标可能不可用，
    绝不会把失败默认记成 supported。
    """

    selected = [item for item in case_results if item.get("split") == config.split]
    if not selected:
        raise ValueError(f"predictions have no cases for split={config.split}")
    if len({str(item["case_id"]) for item in selected}) != len(selected):
        raise ValueError("prediction case ids must be unique")
    traces_by_case = {str(trace["request_id"]): trace for trace in traces}
    missing_traces = sorted(
        str(item["case_id"])
        for item in selected
        if str(item["case_id"]) not in traces_by_case
    )
    if missing_traces:
        raise ValueError(f"predictions reference missing traces: {missing_traces}")

    audited_cases: list[dict[str, object]] = []
    point_verdicts: list[dict[str, object]] = []
    claim_verdicts: list[dict[str, object]] = []
    grader_calls: list[dict[str, object]] = []
    for index, case in enumerate(selected, start=1):
        trace = traces_by_case[str(case["case_id"])]
        audited, point_grade, grounding_grade = _audit_case(
            case,
            trace,
            key_point_grader,
            grounding_grader,
        )
        audited_cases.append(audited)
        grader_calls.extend(
            _grader_call_records(str(case["case_id"]), point_grade, grounding_grade)
        )
        point_verdicts.extend(
            _point_records(str(case["case_id"]), point_grade)
        )
        claim_verdicts.extend(
            _claim_records(str(case["case_id"]), grounding_grade)
        )
        if progress_callback is not None:
            progress_callback(index, len(selected), str(case["case_id"]))

    metric_summary = summarize_end_to_end_results(
        audited_cases,
        key_point_threshold=config.key_point_threshold,
    )
    comparison = _build_coverage_comparison(audited_cases)
    usage = _summarize_grader_usage(grader_calls, config)
    grounding_known = sum(
        item["status"] == "answered" and item["unsupported_answer"] is not None
        for item in audited_cases
    )
    answered = sum(item["status"] == "answered" for item in audited_cases)
    summary = {
        "run_id": config.run_id,
        "report_status": (
            "formal_semantic_audit"
            if grounding_known == answered
            else "semantic_audit_with_unknowns"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config.dataset_version,
        "split": config.split,
        "source_prediction_run_id": config.source_prediction_run_id,
        "prediction_reused_without_generation": True,
        "case_count": len(audited_cases),
        **{
            key: value
            for key, value in metric_summary.items()
            if key != "case_metrics"
        },
        "coverage_comparison": comparison,
        "grader_usage": usage,
        "grounding_review": {
            "answered_count": answered,
            "known_count": grounding_known,
            "unknown_count": answered - grounding_known,
            "complete": grounding_known == answered,
            "independent_human_review": False,
            "independence": config.grader_independence,
        },
        "limitations": [
            "Semantic and grounding verdicts are model-based, not independent human labels.",
            "The same provider/model family generated and graded answers, so self-evaluation bias may remain.",
            "Lexical coverage remains in each case as a deterministic baseline.",
            "Frozen predictions were reused; only evaluation verdicts and metrics were recomputed.",
        ],
    }
    failures = build_end_to_end_failures(metric_summary["case_metrics"])  # type: ignore[arg-type]
    return SemanticAuditResult(
        config=config,
        summary=summary,
        case_results=audited_cases,
        point_verdicts=point_verdicts,
        claim_verdicts=claim_verdicts,
        grader_calls=grader_calls,
        failures=failures,
        comparison_cases=list(comparison["analysis_cases"]),  # type: ignore[arg-type]
    )


def save_semantic_audit_artifacts(
    result: SemanticAuditResult,
    run_dir: Path,
) -> None:
    """保存配置、逐条 verdict、重算指标、失败和 Markdown 对照报告。"""

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"audit run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_config.json", asdict(result.config))
    _write_json(run_dir / "summary.json", result.summary)
    _write_jsonl(run_dir / "case_results.jsonl", result.case_results)
    _write_jsonl(run_dir / "point_verdicts.jsonl", result.point_verdicts)
    _write_jsonl(run_dir / "claim_verdicts.jsonl", result.claim_verdicts)
    _write_jsonl(run_dir / "grader_calls.jsonl", result.grader_calls)
    _write_jsonl(run_dir / "failures.jsonl", result.failures)
    _write_jsonl(run_dir / "comparison_cases.jsonl", result.comparison_cases)
    (run_dir / "report.md").write_text(
        _build_report(result.summary), encoding="utf-8"
    )


def _audit_case(
    case: dict[str, object],
    trace: dict[str, object],
    key_point_grader: KeyPointGrader,
    grounding_grader: GroundingGrader,
) -> tuple[dict[str, object], KeyPointGrade, GroundingGrade]:
    expected_points = [str(point) for point in case.get("expected_points", [])]
    answer = str(case.get("answer", ""))
    status = str(case.get("status", ""))
    if expected_points and status == "answered":
        point_grade = key_point_grader.grade(answer, expected_points)
    elif expected_points:
        point_grade = KeyPointGrade(
            status="completed",
            judgments=[
                PointJudgment(
                    point=point,
                    verdict="not_covered",
                    reason="system_did_not_answer",
                    answer_evidence="",
                )
                for point in expected_points
            ],
            grader_name="system-status",
            grader_version="v1",
        )
    else:
        point_grade = KeyPointGrade(
            "skipped", [], "system-status", "v1"
        )

    if status == "answered":
        cited_context = _extract_cited_context(case, trace)
        grounding_grade = grounding_grader.grade(answer, cited_context)
    else:
        grounding_grade = GroundingGrade(
            "skipped", [], "system-status", "v1"
        )

    semantic_covered = [
        item.point for item in point_grade.judgments if item.verdict == "covered"
    ]
    audited = {
        **case,
        "lexical_key_point_coverage": case.get("key_point_coverage"),
        "lexical_covered_points": list(case.get("covered_points", [])),
        "key_point_coverage": point_grade.coverage,
        "covered_points": semantic_covered,
        "key_point_grader": {
            "name": point_grade.grader_name,
            "version": point_grade.grader_version,
            "status": point_grade.status,
            "error_type": point_grade.error_type,
        },
        "unsupported_answer": grounding_grade.unsupported_answer,
        "unsupported_grader": grounding_grade.grader_name,
        "grounding_grader": {
            "name": grounding_grade.grader_name,
            "version": grounding_grade.grader_version,
            "status": grounding_grade.status,
            "error_type": grounding_grade.error_type,
        },
    }
    return audited, point_grade, grounding_grade


def _extract_cited_context(
    case: dict[str, object],
    trace: dict[str, object],
) -> dict[str, str]:
    citation_ids = {str(item) for item in case.get("citation_ids", [])}
    context: dict[str, str] = {}
    for item in trace.get("retrieved_chunks", []):  # type: ignore[union-attr]
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", ""))
        text = item.get("text")
        if chunk_id in citation_ids and isinstance(text, str):
            context[chunk_id] = text
    return context


def _point_records(case_id: str, grade: KeyPointGrade) -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            **asdict(judgment),
            "grader_status": grade.status,
            "grader_name": grade.grader_name,
            "grader_version": grade.grader_version,
            "error_type": grade.error_type,
        }
        for judgment in grade.judgments
    ]


def _claim_records(case_id: str, grade: GroundingGrade) -> list[dict[str, object]]:
    if not grade.claims and grade.status == "unavailable":
        return [{
            "case_id": case_id,
            "claim": "",
            "verdict": "unknown",
            "citation_ids": [],
            "evidence": [],
            "reason": grade.error_message or "grounding_grader_unavailable",
            "grader_status": grade.status,
            "grader_name": grade.grader_name,
            "grader_version": grade.grader_version,
            "error_type": grade.error_type,
        }]
    return [
        {
            "case_id": case_id,
            **asdict(claim),
            "grader_status": grade.status,
            "grader_name": grade.grader_name,
            "grader_version": grade.grader_version,
            "error_type": grade.error_type,
        }
        for claim in grade.claims
    ]


def _grader_call_records(
    case_id: str,
    point_grade: KeyPointGrade,
    grounding_grade: GroundingGrade,
) -> list[dict[str, object]]:
    """只记录真实 LLM grader 调用，排除系统拒答产生的零成本判断。"""

    records: list[dict[str, object]] = []
    for grade_type, grade in (
        ("key_point", point_grade),
        ("grounding", grounding_grade),
    ):
        if not grade.grader_name.startswith("llm-"):
            continue
        records.append({
            "case_id": case_id,
            "grade_type": grade_type,
            "status": grade.status,
            "grader_name": grade.grader_name,
            "grader_version": grade.grader_version,
            "latency_ms": grade.latency_ms,
            "token_usage": grade.token_usage,
            "error_type": grade.error_type,
            "error_message": grade.error_message,
        })
    return records


def _build_coverage_comparison(
    cases: list[dict[str, object]],
) -> dict[str, object]:
    comparable = [
        item
        for item in cases
        if item.get("answerable")
        and item.get("lexical_key_point_coverage") is not None
        and item.get("key_point_coverage") is not None
    ]
    differences = [
        _comparison_case(item)
        for item in comparable
        if float(item["lexical_key_point_coverage"])
        != float(item["key_point_coverage"])
    ]
    agreements = [
        _comparison_case(item)
        for item in comparable
        if float(item["lexical_key_point_coverage"])
        == float(item["key_point_coverage"])
    ]
    analysis_cases = (differences + agreements)[:10]
    return {
        "comparable_case_count": len(comparable),
        "unknown_case_count": sum(
            item.get("answerable") and item.get("key_point_coverage") is None
            for item in cases
        ),
        "lexical_macro_coverage": _mean_field(
            comparable, "lexical_key_point_coverage"
        ),
        "semantic_macro_coverage": _mean_field(comparable, "key_point_coverage"),
        "improved_case_count": sum(float(item["delta"]) > 0 for item in differences),
        "regressed_case_count": sum(float(item["delta"]) < 0 for item in differences),
        "agreement_case_count": len(agreements),
        "difference_case_count": len(differences),
        "analysis_case_count": len(analysis_cases),
        "analysis_cases": analysis_cases,
    }


def _comparison_case(item: dict[str, object]) -> dict[str, object]:
    lexical = float(item["lexical_key_point_coverage"])
    semantic = float(item["key_point_coverage"])
    return {
        "case_id": item["case_id"],
        "category": item["category"],
        "query": item["query"],
        "lexical_coverage": lexical,
        "semantic_coverage": semantic,
        "delta": semantic - lexical,
        "lexical_covered_points": item["lexical_covered_points"],
        "semantic_covered_points": item["covered_points"],
    }


def _summarize_grader_usage(
    calls: list[dict[str, object]],
    config: SemanticAuditConfig,
) -> dict[str, object]:
    completed = [item for item in calls if item["status"] == "completed"]
    unavailable = [item for item in calls if item["status"] == "unavailable"]
    latencies = [float(item["latency_ms"]) for item in calls]
    token_fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    tokens = {
        field: sum(
            int(dict(item["token_usage"]).get(field, 0))
            for item in calls
        )
        for field in token_fields
    }
    cache_hit = tokens["prompt_cache_hit_tokens"]
    cache_miss = tokens["prompt_cache_miss_tokens"] or (
        tokens["input_tokens"] - cache_hit
    )
    cost = (
        cache_hit * config.input_cache_hit_usd_per_million
        + cache_miss * config.input_cache_miss_usd_per_million
        + tokens["output_tokens"] * config.output_usd_per_million
    ) / 1_000_000
    return {
        "call_count": len(latencies),
        "completed_count": len(completed),
        "unavailable_count": len(unavailable),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "tokens": tokens,
        "estimated_cost_usd": cost,
        "cost_status": "estimated_from_client_reported_usage",
    }


def _mean_field(items: list[dict[str, object]], field: str) -> float | None:
    return mean(float(item[field]) for item in items) if items else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * percentile + 0.999999))
    return ordered[rank - 1]


def _build_report(summary: dict[str, object]) -> str:
    metrics = dict(summary["metrics"])  # type: ignore[arg-type]
    comparison = dict(summary["coverage_comparison"])  # type: ignore[arg-type]
    usage = dict(summary["grader_usage"])  # type: ignore[arg-type]
    review = dict(summary["grounding_review"])  # type: ignore[arg-type]
    return f"""# Semantic Metrics 与 Claim-Level Grounding Audit

## Run

- Run ID：`{summary['run_id']}`
- Dataset：`{summary['dataset_version']}` / `{summary['split']}`
- Source predictions：`{summary['source_prediction_run_id']}`
- Prediction reused without generation：`{summary['prediction_reused_without_generation']}`

## Key-Point Coverage

- Comparable cases：{comparison['comparable_case_count']}
- Lexical macro：{_format_metric(comparison['lexical_macro_coverage'])}
- Semantic macro：{_format_metric(comparison['semantic_macro_coverage'])}
- Improved / regressed / agreement：{comparison['improved_case_count']} / {comparison['regressed_case_count']} / {comparison['agreement_case_count']}
- Unknown cases：{comparison['unknown_case_count']}
- Analysis cases：{comparison['analysis_case_count']}

## Grounding 与 E2E

- Unsupported Answer Rate：{_format_metric(metrics.get('unsupported_answer_rate'))}
- Key-Point Coverage：{_format_metric(metrics.get('key_point_coverage'))}
- End-to-End Success：{_format_metric(metrics.get('end_to_end_success_rate'))}
- Grounding known / answered：{review['known_count']} / {review['answered_count']}
- Independent human review：{review['independent_human_review']}

## Judge Cost

- Calls：{usage['call_count']}，unavailable：{usage['unavailable_count']}
- P50 / P95：{usage['p50_latency_ms']} / {usage['p95_latency_ms']} ms
- Tokens：{dict(usage['tokens']).get('total_tokens', 0)}
- Estimated cost：${float(usage['estimated_cost_usd']):.6f}

## 边界

本 Run 复用已保存 predictions，只重新审核和计算指标。Judge 不是独立人工标注，
且与 Generator 使用同一模型家族，可能存在自评偏差；逐 point/claim verdict、reason
和 evidence span 已落盘供复查。unknown 不按 supported 处理。
"""


def _format_metric(value: object) -> str:
    return "unavailable" if value is None else f"{float(value):.2%}"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
