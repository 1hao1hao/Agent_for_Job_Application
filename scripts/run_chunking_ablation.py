from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from intern_rag.evaluation import (
    EvaluationCase,
    EvaluationRunConfig,
    load_chunks_jsonl,
    load_evaluation_dataset,
    run_retrieval_evaluation,
    save_run_artifacts,
)
from intern_rag.ingestion import Chunk, load_chunks_from_raw_dir
from intern_rag.retrieval import DenseRetriever, HybridRetriever, retrieve_top_k
from intern_rag.retrieval.dense import (
    DenseIndex,
    DenseIndexMetadata,
    SentenceTransformerEmbedder,
)


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
SENTENCE_ENDINGS = frozenset("。！？!?；;")


def main() -> int:
    """在相同 dev Query 和 Hybrid 配置下比较两种长文本切分策略。

    输入是 v0.2 原始语料、原始评测标签和固定 BGE 模型；脚本分别生成固定字符与
    句子边界 Chunk，把原 relevant ids 按来源和文本要点映射到候选 Chunk，然后运行
    真实 Keyword+BGE RRF 检索。输出两组标准 Run、Chunk 质量统计、逐 Case 差异和
    Markdown 报告。该映射是自动派生标签，因此工件明确标记为 dev-only candidate。
    """

    args = _parse_args()
    ablation_id = args.run_id or datetime.now(timezone.utc).strftime(
        "p1-chunking-v02-dev-%Y%m%dT%H%M%SZ"
    )
    report_dir = Path("reports/ablations") / ablation_id
    report_dir.mkdir(parents=True, exist_ok=True)

    original_chunks = load_chunks_jsonl(
        Path("data/processed/chunks/evalrag_v0.2.jsonl")
    )
    original_cases = load_evaluation_dataset(
        Path("data/evaluation/evalrag_v0.2.jsonl")
    )
    strategies = {
        "paragraph_fixed": load_chunks_from_raw_dir(
            Path("data/raw"),
            max_chars=args.max_chars,
            strategy="paragraph_fixed",
        ),
        "sentence_boundary": load_chunks_from_raw_dir(
            Path("data/raw"),
            max_chars=args.max_chars,
            strategy="sentence_boundary",
            min_chunk_ratio=args.min_chunk_ratio,
        ),
    }

    model = SentenceTransformerEmbedder(
        MODEL_NAME,
        MODEL_REVISION,
        device=args.device,
        local_files_only=True,
    )
    summaries: dict[str, dict[str, object]] = {}
    case_results: dict[str, list[dict[str, object]]] = {}
    quality: dict[str, dict[str, object]] = {}
    run_dirs: dict[str, str] = {}

    for strategy, chunks in strategies.items():
        remapped_cases, mapping_stats = _remap_cases(
            original_cases,
            original_chunks,
            chunks,
        )
        index_started = perf_counter()
        retriever = _build_hybrid_retriever(
            chunks,
            model,
            dataset_version=f"evalrag_v0.2-{strategy}-{args.max_chars}",
        )
        index_seconds = perf_counter() - index_started
        run_id = f"{ablation_id}-{strategy}"
        config = EvaluationRunConfig(
            run_id=run_id,
            dataset_version="evalrag_v0.2-dev-derived-labels",
            split="dev",
            retriever_name="hybrid",
            top_k=5,
            chunk_max_chars=args.max_chars,
            git_commit="working-tree",
            command=(
                "PYTHONPATH=src python scripts/run_chunking_ablation.py "
                f"--max-chars {args.max_chars} --min-chunk-ratio "
                f"{args.min_chunk_ratio} --run-id {ablation_id}"
            ),
            candidate_run=True,
            retriever_config={
                "chunking_strategy": strategy,
                "min_chunk_ratio": args.min_chunk_ratio,
                "embedding_name": MODEL_NAME,
                "embedding_revision": MODEL_REVISION,
                "rrf_k": 60,
                "candidate_multiplier": 4,
                "label_mapping": "same-source expected-point or text-overlap",
            },
        )
        result = run_retrieval_evaluation(
            remapped_cases,
            chunks,
            config,
            retriever,
        )
        run_dir = Path("reports/runs") / run_id
        save_run_artifacts(result, run_dir)
        summaries[strategy] = result.summary
        case_results[strategy] = result.case_results
        quality[strategy] = _chunk_quality(
            chunks,
            max_chars=args.max_chars,
            index_seconds=index_seconds,
            mapping_stats=mapping_stats,
        )
        run_dirs[strategy] = str(run_dir)
        _write_chunks(report_dir / f"chunks_{strategy}.jsonl", chunks)

    differences = _case_differences(case_results)
    payload = {
        "ablation_id": ablation_id,
        "dataset_version": "evalrag_v0.2",
        "split": "dev",
        "case_count": 80,
        "max_chars": args.max_chars,
        "min_chunk_ratio": args.min_chunk_ratio,
        "label_status": "derived_candidate_not_frozen_ground_truth",
        "controlled_variables": {
            "raw_corpus": "data/raw",
            "queries": "evalrag_v0.2 dev",
            "router": "rule",
            "retriever": "Keyword + BGE Dense + RRF",
            "top_k": 5,
            "embedding_name": MODEL_NAME,
            "embedding_revision": MODEL_REVISION,
        },
        "run_dirs": run_dirs,
        "chunk_quality": quality,
        "summaries": summaries,
        "case_differences": differences,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text(
        _format_report(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Ablation report written to {report_dir}")
    return 0


def _build_hybrid_retriever(
    chunks: list[Chunk],
    model: SentenceTransformerEmbedder,
    *,
    dataset_version: str,
) -> HybridRetriever:
    """用同一个已加载 BGE 模型为一套 Chunk 建索引并构造 RRF Hybrid。"""

    vectors = model.encode([chunk.text for chunk in chunks])
    index = DenseIndex(
        metadata=DenseIndexMetadata(
            dataset_version=dataset_version,
            embedding_name=model.name,
            embedding_version=model.version,
            dimensions=len(vectors[0]),
            chunk_count=len(chunks),
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        chunk_ids=[chunk.id for chunk in chunks],
        vectors=vectors,
    )
    return HybridRetriever(
        retrieve_top_k,
        DenseRetriever(index, model),
        rrf_k=60,
        candidate_multiplier=4,
    )


def _remap_cases(
    cases: list[EvaluationCase],
    original_chunks: list[Chunk],
    candidate_chunks: list[Chunk],
) -> tuple[list[EvaluationCase], dict[str, int]]:
    """把原 relevant ids 映射到同一来源的新 Chunk，并统计兜底次数。"""

    original_by_id = {chunk.id: chunk for chunk in original_chunks}
    candidates_by_path: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in candidate_chunks:
        candidates_by_path[chunk.source_path].append(chunk)

    fallback_count = 0
    mapped_cases: list[EvaluationCase] = []
    for case in cases:
        mapped_ids: list[str] = []
        for relevant_id in case.relevant_chunk_ids:
            original = original_by_id[relevant_id]
            candidates = candidates_by_path[original.source_path]
            normalized_original = _normalize_text(original.text)
            point_matches = [
                chunk
                for chunk in candidates
                if _normalize_text(chunk.text) in normalized_original
                if any(
                    point.strip() and point.casefold() in chunk.text.casefold()
                    for point in case.expected_points
                )
            ]
            selected = point_matches
            if not selected:
                fallback_count += 1
                selected = [
                    max(
                        candidates,
                        key=lambda chunk: (
                            _character_overlap(original.text, chunk.text),
                            -int(chunk.metadata["chunk_index"]),
                        ),
                    )
                ]
            mapped_ids.extend(chunk.id for chunk in selected)
        mapped_cases.append(
            replace(case, relevant_chunk_ids=list(dict.fromkeys(mapped_ids)))
        )
    return mapped_cases, {
        "case_count": len(cases),
        "relevant_id_count": sum(
            len(case.relevant_chunk_ids) for case in mapped_cases
        ),
        "text_overlap_fallbacks": fallback_count,
    }


def _character_overlap(left: str, right: str) -> float:
    """计算字符二元组 Jaccard，仅用于无法按 expected point 映射时兜底。"""

    left_pairs = {left[index : index + 2] for index in range(len(left) - 1)}
    right_pairs = {right[index : index + 2] for index in range(len(right) - 1)}
    union = left_pairs | right_pairs
    return len(left_pairs & right_pairs) / len(union) if union else 0.0


def _normalize_text(text: str) -> str:
    """移除空白，仅用于判断新 Chunk 是否仍位于原 relevant 文本范围。"""

    return "".join(text.split()).casefold()


def _chunk_quality(
    chunks: list[Chunk],
    *,
    max_chars: int,
    index_seconds: float,
    mapping_stats: dict[str, int],
) -> dict[str, object]:
    """统计长度、短块和疑似句中截断，作为检索指标之外的结构质量证据。"""

    by_path: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_path[chunk.source_path].append(chunk)
    non_final: list[Chunk] = []
    for path_chunks in by_path.values():
        ordered = sorted(path_chunks, key=lambda item: int(item.metadata["chunk_index"]))
        non_final.extend(ordered[:-1])
    lengths = [len(chunk.text) for chunk in chunks]
    mid_sentence_cuts = sum(
        bool(chunk.text) and chunk.text[-1] not in SENTENCE_ENDINGS
        for chunk in non_final
    )
    short_threshold = int(max_chars * 0.5)
    return {
        "chunk_count": len(chunks),
        "length_min": min(lengths, default=0),
        "length_mean": round(mean(lengths), 3) if lengths else 0.0,
        "length_max": max(lengths, default=0),
        "short_chunk_count": sum(length < short_threshold for length in lengths),
        "short_chunk_rate": (
            sum(length < short_threshold for length in lengths) / len(lengths)
            if lengths
            else 0.0
        ),
        "non_final_chunk_count": len(non_final),
        "suspected_mid_sentence_cut_count": mid_sentence_cuts,
        "suspected_mid_sentence_cut_rate": (
            mid_sentence_cuts / len(non_final) if non_final else 0.0
        ),
        "offline_index_seconds": index_seconds,
        "label_mapping": mapping_stats,
    }


def _case_differences(
    by_strategy: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """保存最多十条 Recall/MRR 有变化的 Case，便于解释平均指标。"""

    baseline = {
        str(item["case_id"]): item for item in by_strategy["paragraph_fixed"]
    }
    sentence = {
        str(item["case_id"]): item for item in by_strategy["sentence_boundary"]
    }
    differences: list[dict[str, object]] = []
    for case_id, baseline_case in baseline.items():
        sentence_case = sentence[case_id]
        before = dict(baseline_case["metrics"])
        after = dict(sentence_case["metrics"])
        if before == after:
            continue
        differences.append(
            {
                "case_id": case_id,
                "category": baseline_case["category"],
                "query": baseline_case["query"],
                "paragraph_fixed": before,
                "sentence_boundary": after,
            }
        )
    return differences[:10]


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def _format_report(payload: dict[str, object]) -> str:
    summaries = dict(payload["summaries"])
    quality = dict(payload["chunk_quality"])
    lines = [
        "# Chunking Dev 消融实验",
        "",
        f"- Dataset: `{payload['dataset_version']}` / dev (80 cases)",
        f"- Max chars: `{payload['max_chars']}`; min ratio: `{payload['min_chunk_ratio']}`",
        "- 只改变超长文本边界策略；Router、Retriever、模型和 top-k 保持不变。",
        "- Relevant ids 按来源和 expected point/文本范围自动映射，因此这是 dev-only",
        "  candidate 实验，不是新的 frozen ground truth。",
        "",
        "| 策略 | Chunks | 短块率 | 疑似句中截断率 | Recall@3 | Recall@5 | MRR | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in ("paragraph_fixed", "sentence_boundary"):
        item = dict(quality[strategy])
        summary = dict(summaries[strategy])
        metrics = dict(summary["metrics"])
        latency = dict(dict(summary["latency_ms"])["retrieval"])
        lines.append(
            f"| {strategy} | {item['chunk_count']} | "
            f"{float(item['short_chunk_rate']):.2%} | "
            f"{float(item['suspected_mid_sentence_cut_rate']):.2%} | "
            f"{float(metrics['recall_at_3']):.2%} | "
            f"{float(metrics['recall_at_5']):.2%} | "
            f"{float(metrics['mrr']):.2%} | {float(latency['p95']):.2f} |"
        )
    baseline_metrics = dict(dict(summaries["paragraph_fixed"])["metrics"])
    sentence_metrics = dict(dict(summaries["sentence_boundary"])["metrics"])
    baseline_categories = dict(
        dict(summaries["paragraph_fixed"])["category_metrics"]
    )
    sentence_categories = dict(
        dict(summaries["sentence_boundary"])["category_metrics"]
    )
    lines.extend([
        "",
        "## 结论",
        "",
        "句子边界策略明显减少短 Chunk 和句中截断，但整体检索退化，因此不切换默认配置：",
        "",
        f"- Recall@5: {float(baseline_metrics['recall_at_5']):.2%} -> "
        f"{float(sentence_metrics['recall_at_5']):.2%}.",
        f"- MRR: {float(baseline_metrics['mrr']):.2%} -> "
        f"{float(sentence_metrics['mrr']):.2%}.",
        f"- Single-source Recall@5: "
        f"{float(dict(baseline_categories['single_source'])['recall_at_5']):.2%} -> "
        f"{float(dict(sentence_categories['single_source'])['recall_at_5']):.2%}.",
        f"- Multi-source Recall@5: "
        f"{float(dict(baseline_categories['multi_source'])['recall_at_5']):.2%} -> "
        f"{float(dict(sentence_categories['multi_source'])['recall_at_5']):.2%}.",
        "",
        "可能的取舍是：完整句子有利于单证据问题，但更多竞争片段使多来源问题更难在 "
        "top-5 内覆盖完整证据。当前保留为可选策略；后续若继续实验，应联合比较 "
        "candidate_k/top_k 或 parent-child retrieval，不能只因结构统计改善就启用。",
    ])
    lines.extend(["", "## 指标变化 Case", ""])
    differences = list(payload["case_differences"])
    if not differences:
        lines.append("- 没有 dev Case 的检索指标发生变化。")
    else:
        for item in differences:
            before = dict(item["paragraph_fixed"])
            after = dict(item["sentence_boundary"])
            lines.append(
                f"- `{item['case_id']}`: Recall@5 "
                f"{float(before['recall_at_5'] or 0):.2f} -> "
                f"{float(after['recall_at_5'] or 0):.2f}, MRR "
                f"{float(before['reciprocal_rank'] or 0):.2f} -> "
                f"{float(after['reciprocal_rank'] or 0):.2f}."
            )
    lines.extend([
        "",
        "## 口径边界",
        "",
        "本实验测试 220 字符压力预算下的句子边界行为，不替换 frozen 420 字符数据，",
        "也不能证明最终答案准确率发生变化。",
    ])
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=220)
    parser.add_argument("--min-chunk-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
