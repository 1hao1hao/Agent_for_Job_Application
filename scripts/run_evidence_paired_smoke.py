"""获得证据外发授权后，一次性运行四条 dev 的有界配对 E2E。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from dotenv import load_dotenv
from run_evidence_oriented_evaluation import GATE, append, build_systems, write_json

from intern_rag.agent import (
    ContextEngine,
    PipelineConfig,
    RagPipeline,
    RagRequest,
    build_model_gateway_from_config,
    load_evidence_config,
)
from intern_rag.agent.generation import LlmClientError
from intern_rag.evaluation import load_chunks_jsonl
from intern_rag.evaluation.failure_funnel import classify_failure_stage
from intern_rag.evaluation.knowledge_dataset import load_knowledge_dataset
from intern_rag.routing import build_active_router_from_registry

OUT = ROOT / "reports/ablations/p1-evidence-final-v03-dev-20261010/authorized-e2e"
PRIVATE = ROOT / "traces/private-evidence-paired-20261010"


def export_public_traces() -> None:
    """原始 Trace -> 白名单运行元数据；不公开问题、正文、回答和个人元属性。"""
    for name in ("evidence_gap", "evidence_oriented"):
        target = OUT / f"{name}_traces.jsonl"
        public = []
        for line in (PRIVATE / f"{name}_traces.jsonl").read_text().splitlines():
            trace = json.loads(line)
            row = {key: trace.get(key) for key in (
                "request_id", "trace_id", "intent", "routed_sources", "latency_ms",
                "error_type", "created_at", "response_status", "prompt_version", "token_usage")}
            row["retrieved_chunks"] = [{key: chunk.get(key) for key in (
                "chunk_id", "rank", "score", "source_type")} for chunk in trace["retrieved_chunks"]]
            row["cited_chunk_ids"] = [c["chunk_id"] for c in trace.get("citations", [])]
            row["context"] = {key: trace.get("context", {}).get(key) for key in (
                "used_chunk_ids", "skipped_chunk_ids", "token_count", "max_chars")}
            row["attempts"] = [{key: attempt.get(key) for key in (
                "attempt", "type", "status", "retrieved_chunk_ids", "latency_ms", "token_usage")}
                for attempt in trace["attempts"]]
            row["redaction"] = "payloads omitted; raw trace retained locally in ignored traces/"
            public.append(json.dumps(row, ensure_ascii=False))
        target.write_text("\n".join(public) + "\n", encoding="utf-8")


class BoundedClient:
    """转发现有 Gateway，限制最多八次生成，保留实际用量与错误类型。"""

    def __init__(self, client: object) -> None:
        self.client = client
        self.calls: list[dict[str, object]] = []
        self.consecutive_failures = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.client, name)

    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        """真实调用 -> 保存用量/异常；两次连续失败或八次调用后不再访问 API。"""
        if len(self.calls) >= 8 or self.consecutive_failures >= 2:
            raise LlmClientError("paired smoke hard limit reached")
        entry: dict[str, object] = {"call": len(self.calls) + 1}
        self.calls.append(entry)
        try:
            result = self.client.generate(prompt, model=model, temperature=temperature)
        except LlmClientError as error:
            entry["error_type"] = type(error).__name__
            self.consecutive_failures += 1
            raise
        else:
            self.consecutive_failures = 0
            entry["usage"] = getattr(self.client, "last_token_usage", None)
            return result


def main() -> None:
    """dev 四类各取第一条 -> 交替运行两策略 -> 保存逐次 Trace 和部分结果。

    不重跑检索消融，不读 test。真实 LLM 最多八次、每次最多十五秒，连续
    两次 provider 失败立即停止。失败调用没有用量时记为未知，不视为零 Token。
    """
    load_dotenv(ROOT / ".env")
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    if (OUT / "case_results.jsonl").exists():
        raise ValueError("paired smoke is single-run; do not overwrite saved predictions")
    cases = [c for c in load_knowledge_dataset(ROOT / "data/evaluation/evalrag_v0.3.jsonl") if c.split == "dev"]
    selected = [next(c for c in cases if c.category == category) for category in (
        "single_source", "cross_source", "two_hop", "unanswerable")]
    cfg = json.loads((ROOT / "configs/model_gateway/gateway_v0.1.json").read_text())
    cfg["gateway"].update(timeout_seconds=15.0, max_attempts_per_provider=1)
    cfg["providers"] = cfg["providers"][:1]
    write_json(OUT / "config.json", {"gateway": cfg, "case_ids": [c.case_id for c in selected],
        "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "authorization": "user explicitly approved dev questions and retrieved evidence export to DeepSeek",
        "generation_limit": 8, "frozen_test": "NOT TESTED"})
    client = BoundedClient(build_model_gateway_from_config(cfg))
    chunks = load_chunks_jsonl(ROOT / "data/processed/chunks/evalrag_v0.3.jsonl")
    router = build_active_router_from_registry(ROOT / "configs/routing/router_registry_v0.1.json")
    systems = build_systems()
    pipelines = {name: RagPipeline(chunks, client, PipelineConfig(
        model="deepseek-v4-flash", router_name="hybrid", evidence=load_evidence_config(GATE),
        context_mode="adaptive", context_strategy="source_balanced", context_token_budget=1800),
        trace_path=PRIVATE / f"{name}_traces.jsonl", router=router, routers={"hybrid": router},
        retriever=systems[name], retrievers={"adaptive": systems[name]}, context_engine=ContextEngine())
        for name in ("evidence_gap", "evidence_oriented")}
    rows = []
    for case in selected:
        for name, pipeline in pipelines.items():
            before = len(client.calls)
            response = pipeline.run(RagRequest(query=case.query, request_id=case.case_id, retriever="adaptive"))
            row = {**classify_failure_stage(case, response, pipeline.last_trace).to_dict(),
                   "strategy": name, "response_status": response.status, "error_type": response.error_type,
                   "generation_calls": len(client.calls) - before}
            rows.append(row)
            append(OUT / "case_results.jsonl", row)
            write_json(OUT / "provider_calls.json", client.calls)
            print(f"{name} {case.case_id}: {response.status} / {response.error_type}", flush=True)
            if client.consecutive_failures >= 2 or len(client.calls) >= 8:
                break
        if client.consecutive_failures >= 2 or len(client.calls) >= 8:
            break
    completed = len(rows) == 8
    export_public_traces()
    write_json(OUT / "summary.json", {"status": "completed" if completed else "partial",
        "verification": "small paired dev smoke only" if completed else "E2E NOT VERIFIED",
        "requested_requests": 8, "completed_requests": len(rows), "generation_calls": len(client.calls),
        "provider_failures": sum("error_type" in c for c in client.calls),
        "usage_available_calls": sum(bool(c.get("usage")) for c in client.calls),
        "known_input_tokens": sum((c.get("usage") or {}).get("input_tokens", 0) for c in client.calls),
        "known_output_tokens": sum((c.get("usage") or {}).get("output_tokens", 0) for c in client.calls),
        "cost": "NOT AVAILABLE: price not configured; failure usage is unknown",
        "stop_reason": "two consecutive provider failures" if client.consecutive_failures >= 2 else (
            "generation limit" if not completed else "completed fixed selection"),
        "production_default_changed": False, "frozen_test": "NOT TESTED"})


if __name__ == "__main__":
    main()
