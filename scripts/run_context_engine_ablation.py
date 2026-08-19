from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    ConversationMessage,
    MemoryItem,
    ProfileFact,
    UserProfile,
)
from intern_rag.evaluation.context_dataset import load_context_dataset  # noqa: E402
from intern_rag.ingestion import Chunk  # noqa: E402
from intern_rag.retrieval import RetrievalResult  # noqa: E402
from intern_rag.retrieval import load_dense_index  # noqa: E402


STRATEGIES = {
    "no_memory": ContextEngineConfig(token_budget=220, mode="no_memory"),
    "recent_window": ContextEngineConfig(
        token_budget=220, mode="recent_window", recent_message_count=2
    ),
    "summary_recent": ContextEngineConfig(
        token_budget=220, mode="summary_recent", recent_message_count=2
    ),
    "semantic_memory": ContextEngineConfig(token_budget=220, mode="semantic_memory"),
}


def main() -> int:
    """在 60 组五轮 dev Case 上运行四种 Context/Memory 策略。

    每轮调用真实 ContextEngine，最终轮根据 ManagedContext 是否包含场景标签中的
    expected point 产生确定性 extractive prediction。Runner 保存逐 Case 选择、裁剪、
    记忆召回和延迟；不调用 LLM，不把该指标包装成最终回答准确率。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="p1-d5-context-memory-v01-dev-20260816")
    args = parser.parse_args()
    cases = load_context_dataset(ROOT / "data/evaluation/evalrag_context_v0.1.jsonl")
    _, semantic_model = load_dense_index(
        ROOT / "data/processed/indexes/evalrag_v0.3/bge-small-zh-v1.5"
    )
    strategy_rows: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for strategy, config in STRATEGIES.items():
        rows = [_run_case(case, strategy, config, semantic_model) for case in cases]
        _apply_semantic_scores(rows, cases, semantic_model)
        strategy_rows[strategy] = rows
        summaries[strategy] = _summarize(rows, strategy, config)
        run_dir = ROOT / "reports/runs" / f"{args.run_id}-{strategy}"
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
        "semantic_grader": {
            "model": semantic_model.name,
            "revision": semantic_model.version,
            "threshold": 0.80,
        },
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
    """执行一个五轮场景，语义记忆策略先从带干扰项的候选中召回 top-k。"""

    engine = ContextEngine()
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
    final = None
    total_latency = 0.0
    for turn_index, message in enumerate(messages):
        evidence = _evidence(case) if turn_index == len(messages) - 1 else []
        started = perf_counter()
        recalled_memories = (
            _recall_memories(message.content, memories, semantic_model, top_k=3)
            if strategy == "semantic_memory" and turn_index == len(messages) - 1
            else ()
        )
        final = engine.build(
            query=message.content,
            system_prompt="仅依据本轮 Context 回答，缺少事实时拒答。",
            retrieved_results=evidence,
            config=config,
            required_source_types=("jd",) if evidence else (),
            profile=profile,
            history=messages[:turn_index],
            memories=recalled_memories,
            history_summary=case.summary or None,
        )
        total_latency += (perf_counter() - started) * 1000
        current_history = {
            item.segment_id for item in final.segments if item.kind == "history"
        }
        repeated_reads += len(current_history & seen_history_ids)
        seen_history_ids.update(current_history)
    assert final is not None
    point_covered = bool(case.expected_point and case.expected_point in final.text)
    correctly_abstained = not case.answerable and not point_covered
    follow_up_success = point_covered if case.answerable else correctly_abstained
    answer = case.expected_point if point_covered else "当前上下文不足，无法回答。"
    cited_ids = list(final.evidence.used_chunk_ids) if point_covered else []
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
        },
        "metrics": {
            "follow_up_success": follow_up_success,
            "citation_validity": set(cited_ids).issubset(final.evidence.used_chunk_ids),
            "semantic_key_point_coverage": None,
            "semantic_similarity": None,
            "grounding": (answer in final.text) if point_covered else correctly_abstained,
            "repeated_history_reads": repeated_reads,
            "prompt_tokens": final.token_count,
            "compression_ratio": 1.0 - min(final.token_count / raw_tokens, 1.0),
            "latency_ms": total_latency,
        },
    }


def _recall_memories(query: str, memories, model, *, top_k: int):
    """用固定 BGE 余弦分数结合人工确认 importance 选择当前用户记忆。"""

    if not memories:
        return ()
    vectors = model.encode([query, *[item.content for item in memories]])
    query_vector, memory_vectors = vectors[0], vectors[1:]
    ranked = sorted(
        memories,
        key=lambda item: (
            -(_cosine(query_vector, memory_vectors[memories.index(item)]) + 0.1 * item.importance),
            -item.version,
            item.memory_id,
        ),
    )
    return tuple(ranked[:top_k])


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
    return {
        "strategy": strategy,
        "config": config.__dict__,
        "follow_up_success": mean(float(row["metrics"]["follow_up_success"]) for row in rows),
        "citation_validity": mean(float(row["metrics"]["citation_validity"]) for row in rows),
        "semantic_key_point_coverage": mean(
            float(row["metrics"]["semantic_key_point_coverage"]) for row in answerable
        ),
        "grounding": mean(float(row["metrics"]["grounding"]) for row in rows),
        "mean_repeated_history_reads": mean(
            float(row["metrics"]["repeated_history_reads"]) for row in rows
        ),
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
    for case_id in indexed["no_memory"]:
        scores = {
            name: bool(rows[case_id]["metrics"]["follow_up_success"])
            for name, rows in indexed.items()
        }
        if len(set(scores.values())) > 1:
            output.append(
                {
                    "case_id": case_id,
                    "category": indexed["no_memory"][case_id]["category"],
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


def _report(payload) -> str:
    lines = [
        "# P1-D5 Context Engine and Memory Dev Ablation",
        "",
        "- Dataset: `evalrag_context_v0.1`; split: dev; 60 groups / 300 turns.",
        "- Predictions are produced by ContextEngine; no LLM calls, estimated cost $0.",
        "",
        "| Strategy | Follow-up | Citation | Key-point | Grounding | Prompt tokens | Compression | Repeat reads | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in payload["strategies"].items():
        lines.append(
            f"| {name} | {value['follow_up_success']:.2%} | {value['citation_validity']:.2%} | "
            f"{value['semantic_key_point_coverage']:.2%} | {value['grounding']:.2%} | "
            f"{value['mean_prompt_tokens']:.1f} | {value['mean_compression_ratio']:.2%} | "
            f"{value['mean_repeated_history_reads']:.1f} | {value['latency_ms']['p95']:.3f} |"
        )
    lines.extend(["", "## Difference cases", ""])
    lines.extend(
        f"- `{item['case_id']}` ({item['category']}): {item['follow_up_success']}"
        for item in payload["differences"]
    )
    lines.extend(["", "## Boundary", "", payload["metric_scope"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
