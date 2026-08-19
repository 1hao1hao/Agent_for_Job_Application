from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter, sleep


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.agent import (  # noqa: E402
    GatewayProvider,
    LlmClientError,
    LlmTimeoutError,
    ModelGateway,
    ModelGatewayConfig,
    ModelGatewayUnavailable,
)


class ScenarioClient:
    """为故障矩阵提供确定性响应，不访问外部模型。"""

    def __init__(self, outcomes: list[object], delay_ms: float = 0.0) -> None:
        self.outcomes = list(outcomes)
        self.delay_ms = delay_ms
        self.last_token_usage = {"input_tokens": 120, "output_tokens": 30}

    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        del prompt, model, temperature
        if self.delay_ms:
            sleep(self.delay_ms / 1000)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)


def main() -> int:
    """运行固定故障场景，保存成功率、fallback、延迟和估算成本。"""

    output = ROOT / "reports/fault_injection/p1-d7-model-gateway-v01"
    output.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "primary_success": (["ok"], ["backup"], 2.0),
        "timeout_fallback": ([LlmTimeoutError("timeout")] * 2, ["backup"], 1.0),
        "rate_limit_retry": ([LlmClientError("HTTP 429"), "ok"], ["backup"], 1.0),
        "provider_5xx_fallback": ([LlmClientError("HTTP 503")] * 2, ["backup"], 1.0),
        "auth_fallback": ([LlmClientError("HTTP 401")], ["backup"], 1.0),
        "all_unavailable": ([LlmClientError("HTTP 400")], [LlmClientError("HTTP 403")], 1.0),
    }
    rows = []
    for name, (primary_outcomes, backup_outcomes, delay_ms) in scenarios.items():
        primary = ScenarioClient(list(primary_outcomes), delay_ms)
        backup = ScenarioClient(list(backup_outcomes), delay_ms)
        gateway = ModelGateway(
            [
                GatewayProvider("primary", primary, "primary-v1", 1.0, 2.0),
                GatewayProvider("backup", backup, "backup-v1", 1.5, 2.5),
            ],
            ModelGatewayConfig(
                max_attempts_per_provider=2,
                backoff_base_seconds=0,
                circuit_failure_threshold=10,
            ),
        )
        started = perf_counter()
        try:
            gateway.generate("sanitized benchmark prompt", model="gateway", temperature=0)
            status = "succeeded"
        except ModelGatewayUnavailable:
            status = "controlled_failure"
        elapsed = (perf_counter() - started) * 1000
        trace = gateway.last_gateway_trace
        rows.append({
            "scenario": name,
            "status": status,
            "fallback_used": bool(trace.get("fallback_used")),
            "selected_provider": trace.get("selected_provider"),
            "attempt_count": trace.get("attempt_count"),
            "latency_ms": elapsed,
            "estimated_cost_usd": trace.get("estimated_cost_usd", 0.0),
            "attempts": trace.get("attempts", []),
        })
    latencies = [float(row["latency_ms"]) for row in rows]
    succeeded = [row for row in rows if row["status"] == "succeeded"]
    summary = {
        "benchmark": "p1-d7-model-gateway-fault-matrix-v0.1",
        "provider_type": "deterministic fake; no network",
        "case_count": len(rows),
        "success_rate": len(succeeded) / len(rows),
        "fallback_rate": sum(bool(row["fallback_used"]) for row in rows) / len(rows),
        "latency_ms": {"p50": _percentile(latencies, 0.50), "p95": _percentile(latencies, 0.95)},
        "estimated_cost_usd": sum(float(row["estimated_cost_usd"]) for row in rows),
        "boundary": "固定故障注入结果，不代表真实 Provider 可用率或线上 SLO。",
    }
    _write_json(output / "summary.json", summary)
    (output / "case_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(summary, rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * ratio + 0.999999) - 1)] if ordered else 0.0


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _report(summary: dict[str, object], rows: list[dict[str, object]]) -> str:
    lines = [
        "# Model Gateway Fault Matrix",
        "",
        "> 本报告使用确定性 Fake Provider，只证明控制流，不代表线上 SLO。",
        "",
        "| Scenario | Status | Selected | Fallback | Attempts | Latency ms |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['status']} | {row['selected_provider']} | "
            f"{int(bool(row['fallback_used']))} | {row['attempt_count']} | {float(row['latency_ms']):.3f} |"
        )
    lines.extend(["", f"- Success rate: {float(summary['success_rate']):.2%}", f"- Fallback rate: {float(summary['fallback_rate']):.2%}"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
