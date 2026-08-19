from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import build_model_gateway_from_config  # noqa: E402


def main() -> int:
    """有凭证时调用真实 Provider，并保存不含 Prompt/密钥的脱敏结果。"""

    output = ROOT / "reports/smoke/p1-d7-model-gateway"
    output.mkdir(parents=True, exist_ok=True)
    available = [name for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY") if os.environ.get(name)]
    payload: dict[str, object] = {
        "config": "configs/model_gateway/gateway_v0.1.json",
        "available_provider_credentials": available,
        "real_primary_executed": False,
        "real_fallback_executed": False,
    }
    if not available:
        payload["status"] = "skipped_no_credentials"
    else:
        try:
            gateway = build_model_gateway_from_config(ROOT / "configs/model_gateway/gateway_v0.1.json")
            gateway.generate(
                '只输出 {"status":"ok"}，不要添加其他字段。',
                model="gateway",
                temperature=0,
            )
            payload.update({
                "status": "succeeded",
                "real_primary_executed": True,
                "selected_provider": gateway.last_gateway_trace.get("selected_provider"),
                "fallback_used": gateway.last_gateway_trace.get("fallback_used"),
                "attempt_count": gateway.last_gateway_trace.get("attempt_count"),
                "latency_ms": gateway.last_gateway_trace.get("total_latency_ms"),
                "tokens": gateway.last_token_usage,
            })
            payload["real_fallback_executed"] = bool(payload["fallback_used"])
        except Exception as error:
            payload.update({"status": "controlled_failure", "error_type": type(error).__name__})
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] in {"succeeded", "skipped_no_credentials"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
