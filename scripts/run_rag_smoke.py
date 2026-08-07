from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

from intern_rag.agent import (
    OpenAIResponsesClient,
    PipelineConfig,
    RagPipeline,
    RagRequest,
)
from intern_rag.ingestion import load_chunks_from_raw_dir


DEFAULT_MODEL = "gpt-4.1-mini"


def main() -> int:
    """使用真实模型运行正常回答与资料不足两个 smoke query。

    该脚本不是自动化测试，也不产生正式评测指标。缺少 OPENAI_API_KEY 时会
    明确跳过；密钥不会写入输出、Trace 或代码。
    """

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("SKIPPED: 未设置 OPENAI_API_KEY，真实模型 smoke test 未执行。")
        return 0

    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    chunks = load_chunks_from_raw_dir(Path("data/raw"))
    pipeline = RagPipeline(
        chunks=chunks,
        llm_client=OpenAIResponsesClient(
            timeout_seconds=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "30"))
        ),
        config=PipelineConfig(
            model=model,
            temperature=0.0,
            prompt_version="p0-v1",
            context_max_chars=4000,
        ),
        trace_path=Path("traces/rag_smoke.jsonl"),
    )
    requests = [
        RagRequest(
            query="大模型应用研发实习生岗位要求哪些技能？",
            request_id="smoke-answered",
            top_k=3,
        ),
        RagRequest(
            query="现有资料能否证明候选人有量子芯片流片经历？",
            request_id="smoke-insufficient",
            top_k=3,
        ),
    ]

    print(f"Running smoke test with model={model}")
    for request in requests:
        response = pipeline.run(request)
        print(json.dumps(asdict(response), ensure_ascii=False, indent=2))

    print("Trace written to traces/rag_smoke.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
