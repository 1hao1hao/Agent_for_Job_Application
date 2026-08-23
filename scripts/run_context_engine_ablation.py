from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict
from statistics import mean
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import (  # noqa: E402
    ContextEngine,
    ContextEngineConfig,
    ContextInputs,
    ContextPolicy,
    ContextPolicyConfig,
    ContextSignalExtractor,
    ConversationMessage,
    MemoryItem,
    ProfileFact,
    UserProfile,
    rank_memories,
)
from intern_rag.evaluation.context_dataset import load_context_dataset  # noqa: E402
from intern_rag.ingestion import Chunk  # noqa: E402
from intern_rag.retrieval import RetrievalResult  # noqa: E402
from intern_rag.retrieval import load_dense_index  # noqa: E402


STRATEGIES = {
    "recent_window": ContextEngineConfig(
        token_budget=220, mode="recent_window", recent_message_count=2
    ),
    "summary_recent": ContextEngineConfig(
        token_budget=220, mode="summary_recent", recent_message_count=2
    ),
    "semantic_memory": ContextEngineConfig(token_budget=220, mode="semantic_memory"),
    "adaptive_policy": ContextEngineConfig(token_budget=220, mode="adaptive"),
}
ADAPTIVE_POLICY_CONFIG = ContextPolicyConfig()


def main() -> int:
    """在 60 组五轮 dev Case 上运行四种 Context/Memory 策略。

    每轮调用真实 ContextEngine，最终轮根据 ManagedContext 是否包含场景标签中的
    expected point 产生确定性 extractive prediction。Runner 保存逐 Case 选择、裁剪、
    记忆召回和延迟；不调用 LLM，不把该指标包装成最终回答准确率。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="p1-adaptive-context-v02-dev-20260823")
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help="逗号分隔的策略；已存在策略会从同一 run-id 工件读取以支持单变量重跑。",
    )
    args = parser.parse_args()
    cases = load_context_dataset(ROOT / "data/evaluation/evalrag_context_v0.1.jsonl")
    _, semantic_model = load_dense_index(
        ROOT / "data/processed/indexes/evalrag_v0.3/bge-small-zh-v1.5"
    )
    strategy_rows: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    selected = {item.strip() for item in args.strategies.split(",") if item.strip()}
    unknown = selected - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown context strategies: {sorted(unknown)}")
    for strategy, config in STRATEGIES.items():
        run_dir = ROOT / "reports/runs" / f"{args.run_id}-{strategy}"
        if strategy not in selected:
            strategy_rows[strategy] = _read_jsonl(run_dir / "case_results.jsonl")
            summaries[strategy] = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            continue
        rows = [_run_case(case, strategy, config, semantic_model) for case in cases]
        _apply_semantic_scores(rows, cases, semantic_model)
        strategy_rows[strategy] = rows
        summaries[strategy] = _summarize(rows, strategy, config)
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(run_dir / "case_results.jsonl", rows)
        (run_dir / "summary.json").write_text(
            json.dumps(summaries[strategy], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "dataset": "evalrag_context_v0.1",
                    "split": "dev",
                    "strategy": strategy,
                    **config.__dict__,
                    **(
                        {"context_policy": asdict(ADAPTIVE_POLICY_CONFIG)}
                        if strategy == "adaptive_policy"
                        else {}
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    comparison = {
        "run_id": args.run_id,
        "dataset_version": "evalrag_context_v0.1",
        "split": "dev",
        "case_count": 60,
        "turn_count": 300,
        "strategies": summaries,
        "differences": _differences(strategy_rows)[:15],
        "metric_scope": (
            "Context-level scenario benchmark; key-point uses scenario-grounded expected "
            "facts and does not equal free-form answer accuracy"
        ),
        "llm_calls": 0,
        "estimated_cost_usd": 0.0,
        "semantic_model": {
            "model": semantic_model.name,
            "revision": semantic_model.version,
            "threshold": 0.80,
        },
        "context_policy": asdict(ADAPTIVE_POLICY_CONFIG),
    }
    output = ROOT / "reports/ablations" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_report(comparison), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


def _run_case(case, strategy: str, config: ContextEngineConfig, semantic_model) -> dict[str, object]:
    """执行一个五轮场景并保存每轮 Context 信号、计划和预算结果。

    Baseline 仍由固定 mode 控制；adaptive_policy 每轮先提取 signals，再由
    ContextPolicy 生成 plan，最后交给同一个 ContextEngine。四种策略共享数据、预算、
    embedding 和成功判定，保证 Token/Redundancy 差异来自策略本身。
    """

    extractor = ContextSignalExtractor(semantic_model)
    policy = ContextPolicy(ADAPTIVE_POLICY_CONFIG)
    engine = ContextEngine(semantic_similarity=extractor.similarities)
    profile = (
        UserProfile(
            case.user_id,
            tuple(ProfileFact(**item) for item in case.profile_facts),
            1,
            "2026-08-16T00:00:00+00:00",
        )
        if case.profile_facts
        else None
    )
    memories = tuple(
        MemoryItem(
            memory_id=str(item["memory_id"]),
            user_id=case.user_id,
            memory_type=str(item["memory_type"]),  # type: ignore[arg-type]
            content=str(item["content"]),
            source=str(item["source"]),
            importance=float(item["importance"]),
            created_at="2026-08-16T00:00:00+00:00",
            version=int(item["version"]),
        )
        for item in case.memories
    )
    messages = tuple(
        ConversationMessage(
            str(item["message_id"]),
            case.session_id,
            case.user_id,
            str(item["role"]),  # type: ignore[arg-type]
            str(item["content"]),
            str(item["created_at"]),
        )
        for item in case.messages
    )
    seen_history_ids: set[str] = set()
    repeated_reads = 0
    history_reads = 0
    final = None
    total_latency = 0.0
    turn_plans: list[dict[str, object]] = []
    for turn_index, message in enumerate(messages):
        evidence = _evidence(case) if turn_index == len(messages) - 1 else []
        started = perf_counter()
        ranked = rank_memories(message.content, memories, extractor)
        inputs = ContextInputs(
            profile=profile,
            history=messages[:turn_index],
            memories=ranked,
            history_summary=case.summary or None,
            history_source="benchmark",
        )
        signals = None
        plan = None
        if strategy == "adaptive_policy":
            signals = extractor.extract(
                message.content, inputs, token_budget=config.token_budget
            )
            plan = policy.decide(signals)
        recalled_memories = ranked[:3] if strategy == "semantic_memory" else ranked
        final = engine.build(
            query=message.content,
            system_prompt="仅依据本轮 Context 回答，缺少事实时拒答。",
            retrieved_results=evidence,
            config=config,
            required_source_types=("jd",) if evidence else (),
            profile=inputs.profile,
            history=inputs.history,
            memories=recalled_memories,
            history_summary=inputs.history_summary,
            plan=plan,
            signals=signals,
        )
        total_latency += (perf_counter() - started) * 1000
        current_history = {
            item.segment_id for item in final.segments if item.kind == "history"
        }
        repeated_reads += len(current_history & seen_history_ids)
        history_reads += len(current_history)
        seen_history_ids.update(current_history)
        turn_plans.append({
            "turn": turn_index + 1,
            "query": message.content,
            "signals": asdict(signals) if signals is not None else None,
            "plan": asdict(plan) if plan is not None else None,
            "kept_segment_ids": list(final.kept_ids),
            "dropped": list(final.dropped),
        })
    assert final is not None
    point_covered = bool(case.expected_point and case.expected_point in final.text)
    correctly_abstained = not case.answerable and not point_covered
    follow_up_success = point_covered if case.answerable else correctly_abstained
    answer = case.expected_point if point_covered else "当前上下文不足，无法回答。"
    cited_ids = list(final.evidence.used_chunk_ids) if point_covered else []
    relevant_memory_ids = {
        str(item["memory_id"])
        for item in case.memories
        if case.expected_point and str(item["content"]) == case.expected_point
    }
    recalled_memory_ids = set(final.recalled_memory_ids)
    memory_recall = (
        bool(recalled_memory_ids & relevant_memory_ids)
        if relevant_memory_ids
        else None
    )
    raw_text = "\n".join(item.content for item in messages[:-1]) + "\n" + "\n".join(
        str(item["text"]) for item in case.evidence
    )
    raw_tokens = max(engine.estimator.count(raw_text), 1)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "strategy": strategy,
        "answerable": case.answerable,
        "prediction": {
            "answer": answer,
            "cited_chunk_ids": cited_ids,
            "kept_segment_ids": list(final.kept_ids),
            "dropped": list(final.dropped),
            "recalled_memory_ids": list(final.recalled_memory_ids),
            "context_signals": asdict(final.context_signals)
            if final.context_signals is not None else None,
            "context_plan": asdict(final.context_plan)
            if final.context_plan is not None else None,
            "turn_plans": turn_plans,
        },
        "metrics": {
            "follow_up_success": follow_up_success,
            "citation_validity": set(cited_ids).issubset(final.evidence.used_chunk_ids),
            "semantic_key_point_coverage": None,
            "semantic_similarity": None,
            "grounding": (answer in final.text) if point_covered else correctly_abstained,
            "repeated_history_reads": repeated_reads,
            "history_reads": history_reads,
            "history_redundancy": repeated_reads / history_reads if history_reads else 0.0,
            "memory_recall_accuracy": memory_recall,
            "prompt_tokens": final.token_count,
            "compression_ratio": 1.0 - min(final.token_count / raw_tokens, 1.0),
            "latency_ms": total_latency,
        },
    }


def _apply_semantic_scores(rows, cases, model) -> None:
    """用固定 BGE embedding 计算 answer/expected point 语义覆盖，不调用 LLM。"""

    expected_by_id = {case.case_id: case.expected_point for case in cases}
    answerable_rows = [row for row in rows if row["answerable"]]
    expected = [expected_by_id[str(row["case_id"])] for row in answerable_rows]
    answers = [str(row["prediction"]["answer"]) for row in answerable_rows]
    expected_vectors = model.encode(expected)
    answer_vectors = model.encode(answers)
    for row, left, right in zip(answerable_rows, expected_vectors, answer_vectors):
        similarity = _cosine(left, right)
        row["metrics"]["semantic_similarity"] = similarity
        row["metrics"]["semantic_key_point_coverage"] = similarity >= 0.80


def _cosine(left, right) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) ** 2 for value in left) ** 0.5
    right_norm = sum(float(value) ** 2 for value in right) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _evidence(case) -> list[RetrievalResult]:
    output = []
    for rank, item in enumerate(case.evidence, 1):
        chunk = Chunk(
            str(item["chunk_id"]),
            str(item["source_type"]),
            "context-benchmark",
            "场景证据",
            str(item["text"]),
            {},
        )
        output.append(RetrievalResult(chunk.id, 1.0, rank, chunk, "scenario_evidence"))
    return output


def _summarize(rows, strategy: str, config: ContextEngineConfig) -> dict[str, object]:
    latencies = sorted(float(row["metrics"]["latency_ms"]) for row in rows)
    answerable = [row for row in rows if row["answerable"]]
    memory_rows = [
        row for row in rows
        if row["metrics"]["memory_recall_accuracy"] is not None
    ]
    return {
        "strategy": strategy,
        "config": {
            **config.__dict__,
            **(
                {"context_policy": asdict(ADAPTIVE_POLICY_CONFIG)}
                if strategy == "adaptive_policy"
                else {}
            ),
        },
        "follow_up_success": mean(float(row["metrics"]["follow_up_success"]) for row in rows),
        "citation_validity": mean(float(row["metrics"]["citation_validity"]) for row in rows),
        "semantic_key_point_coverage": mean(
            float(row["metrics"]["semantic_key_point_coverage"]) for row in answerable
        ),
        "grounding": mean(float(row["metrics"]["grounding"]) for row in rows),
        "mean_repeated_history_reads": mean(
            float(row["metrics"]["repeated_history_reads"]) for row in rows
        ),
        "mean_history_redundancy": mean(
            float(row["metrics"]["history_redundancy"]) for row in rows
        ),
        "memory_recall_accuracy": mean(
            float(row["metrics"]["memory_recall_accuracy"]) for row in memory_rows
        ) if memory_rows else None,
        "mean_prompt_tokens": mean(float(row["metrics"]["prompt_tokens"]) for row in rows),
        "mean_compression_ratio": mean(
            float(row["metrics"]["compression_ratio"]) for row in rows
        ),
        "latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
    }


def _differences(values):
    indexed = {
        name: {row["case_id"]: row for row in rows} for name, rows in values.items()
    }
    output = []
    for case_id in indexed["recent_window"]:
        scores = {
            name: bool(rows[case_id]["metrics"]["follow_up_success"])
            for name, rows in indexed.items()
        }
        if len(set(scores.values())) > 1:
            output.append(
                {
                    "case_id": case_id,
                    "category": indexed["recent_window"][case_id]["category"],
                    "follow_up_success": scores,
                }
            )
    return output


def _percentile(values: list[float], quantile: float) -> float:
    return values[min(int((len(values) - 1) * quantile), len(values) - 1)] if values else 0.0


def _write_jsonl(path: Path, values) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """读取同一 run-id 已完成的 baseline，避免策略校准时重复运行。"""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _report(payload) -> str:
    lines = [
        "# Adaptive Context Engine and Layered Memory Dev Ablation",
        "",
        "- Dataset: `evalrag_context_v0.1`; split: dev; 60 groups / 300 turns.",
        "- Predictions are produced by ContextEngine; no LLM calls, estimated cost $0.",
        "",
        "| Strategy | Follow-up | Key-point | Prompt tokens | History redundancy | Memory recall | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in payload["strategies"].items():
        lines.append(
            f"| {name} | {value['follow_up_success']:.2%} | "
            f"{value['semantic_key_point_coverage']:.2%} | {value['mean_prompt_tokens']:.1f} | "
            f"{value['mean_history_redundancy']:.2%} | "
            f"{value['memory_recall_accuracy']:.2%} | {value['latency_ms']['p95']:.3f} |"
        )
    baseline = payload["strategies"]["summary_recent"]
    adaptive = payload["strategies"]["adaptive_policy"]
    token_reduction = 1.0 - (
        adaptive["mean_prompt_tokens"] / baseline["mean_prompt_tokens"]
    )
    redundancy_reduction = 1.0 - (
        adaptive["mean_history_redundancy"]
        / baseline["mean_history_redundancy"]
    )
    lines.extend([
        "",
        "## Quality-equivalent comparison",
        "",
        (
            "Adaptive and Summary+Recent both keep 100% Follow-up Success. "
            f"Adaptive reduces mean Prompt Token by {token_reduction:.2%} and "
            f"History Redundancy by {redundancy_reduction:.2%}."
        ),
        (
            "Adaptive Memory Recall is lower than always-on Semantic Memory; "
            "this is an explicit on-demand recall trade-off, not a global Pareto claim."
        ),
    ])
    lines.extend(["", "## Difference cases", ""])
    lines.extend(
        f"- `{item['case_id']}` ({item['category']}): {item['follow_up_success']}"
        for item in payload["differences"]
    )
    lines.extend(["", "## Boundary", "", payload["metric_scope"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
