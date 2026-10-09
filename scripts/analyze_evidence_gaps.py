from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/runs/p1-task2-evidence-rerank-context-v03-dev-20261006"
OUTPUT = ROOT / "reports/ablations/p1-evidence-gap-v03-dev-20261009-r2"


def main() -> int:
    """用历史 Trace 将 retrieval miss 归因，gold 只参与离线分析。"""

    cases = {
        item["case_id"]: item
        for item in _jsonl(ROOT / "data/evaluation/evalrag_v0.3.jsonl")
    }
    chunks = {
        item["chunk_id"]: item
        for item in _jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    }
    traces = {item["request_id"]: item for item in _jsonl(RUN / "traces.jsonl")}
    misses = [
        item for item in _jsonl(RUN / "case_results.jsonl")
        if item.get("terminal_stage") == "retrieval_miss"
    ]
    rows = []
    for miss in misses:
        case_id = str(miss["case_id"])
        case = cases[case_id]
        trace = traces[case_id]
        gold = set(case["relevant_chunk_ids"])
        gold_sources = {
            str(chunks[item]["source_type"]) for item in gold if item in chunks
        }
        attempts = [
            item for item in trace.get("attempts", [])
            if item.get("retrieval_decision") is not None
        ]
        first_filter = set(attempts[0].get("source_filter") or ()) if attempts else set()
        candidates = {
            chunk_id
            for attempt in attempts
            for chunk_id in attempt.get("retrieval_decision", {}).get(
                "candidate_chunk_ids", []
            )
        }
        if gold - set(chunks):
            cause = "corpus_or_label_missing"
        elif first_filter and not (first_filter & gold_sources):
            cause = "route_source_filter_mismatch"
        elif case["category"] in {"two_hop", "three_hop"}:
            cause = "graph_path_missing"
        elif not (gold & candidates):
            cause = "candidate_pool_miss"
        else:
            cause = "ranking_or_single_route_insufficient"
        rows.append({
            "case_id": case_id,
            "category": case["category"],
            "query": case["query"],
            "cause": cause,
            "first_source_filter": sorted(first_filter),
            "gold_sources": sorted(gold_sources),
            "gold_in_top20_candidates": bool(gold & candidates),
            "attempt_count": len(attempts),
        })
    payload = {
        "source_run": str(RUN.relative_to(ROOT)),
        "retrieval_miss_count": len(rows),
        "cause_counts": dict(Counter(item["cause"] for item in rows)),
        "gold_absent_from_top20_count": sum(
            not item["gold_in_top20_candidates"] for item in rows
        ),
        "corpus_or_label_missing_count": sum(
            item["cause"] == "corpus_or_label_missing" for item in rows
        ),
        "boundary": "gold labels are used offline only and never enter online selection",
        "cases": rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "failure_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "failure_analysis.md").write_text(
        _report(payload), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"}, ensure_ascii=False, indent=2))
    return 0


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _report(payload: dict[str, object]) -> str:
    lines = [
        "# Historical Retrieval Miss Analysis", "",
        f"- Source run: `{payload['source_run']}`",
        f"- Retrieval misses: {payload['retrieval_miss_count']}",
        f"- Gold absent from Top-20: {payload['gold_absent_from_top20_count']}",
        f"- Corpus/label missing: {payload['corpus_or_label_missing_count']}", "",
        "| Root cause | Cases |", "|---|---:|",
    ]
    for cause, count in sorted(dict(payload["cause_counts"]).items()):
        lines.append(f"| {cause} | {count} |")
    lines.extend(["", "Gold ids were used only after prediction for offline diagnosis.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
