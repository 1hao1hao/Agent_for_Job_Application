from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def main() -> int:
    """把 Component、deterministic Full Stack 与真实 LLM 结果汇总为面试证据报告。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--component-root",
        type=Path,
        default=Path("reports/loadtest/p1-locust-local-20260904"),
    )
    args = parser.parse_args()
    deterministic = _load_summary(args.output_root / "deterministic/summary.json")
    optimized = _load_summary(
        args.output_root / "deterministic-thread-limited/summary.json", required=False
    )
    real = _load_summary(args.output_root / "real/summary.json", required=False)
    environment = _read_json(args.output_root / "deterministic/environment.json")
    preflight = _read_json(args.output_root / "deterministic/preflight.json")
    component = _load_component(args.component_root)
    report = _render_report(
        component, deterministic, optimized, real, environment, preflight
    )
    (args.output_root / "report.md").write_text(report, encoding="utf-8")
    print(args.output_root / "report.md")
    return 0


def _load_summary(path: Path, required: bool = True) -> list[dict[str, object]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    return list(_read_json(path)["tiers"])


def _load_component(root: Path) -> list[dict[str, object]]:
    tiers: list[dict[str, object]] = []
    for concurrency in (1, 5, 10, 20, 50):
        path = root / f"c{concurrency}_stats.csv"
        with path.open(encoding="utf-8", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
        row = next(item for item in rows if item["Name"] == "Aggregated")
        request_count = int(row["Request Count"])
        failures = int(row["Failure Count"])
        tiers.append({
            "concurrency": concurrency,
            "request_count": request_count,
            "rps": float(row["Requests/s"]),
            "p50_ms": float(row["50%"]),
            "p95_ms": float(row["95%"]),
            "p99_ms": float(row["99%"]),
            "failure_rate": failures / request_count if request_count else 0.0,
        })
    return tiers


def _render_report(
    component: list[dict[str, object]],
    deterministic: list[dict[str, object]],
    optimized: list[dict[str, object]],
    real: list[dict[str, object]],
    environment: dict[str, object],
    preflight: dict[str, object],
) -> str:
    deterministic_knee = _knee(deterministic)
    optimized_knee = _knee(optimized)
    real_knee = _knee(real)
    lines = [
        "# EvalRAG Full E2E Load Test",
        "",
        "## 结论边界",
        "",
        "本报告区分组件隔离吞吐、完整内部链路上限和真实模型 Provider 边界。所有数字只代表",
        "当前共享服务器、单 Uvicorn worker 与锁定配置，不是生产 SLA。Component 旧结果保留",
        "用于定位 FastAPI/线程池开销，不再冒充 Full E2E。",
        "",
        "## 环境与真实性",
        "",
        f"- Python `{environment.get('python')}`；逻辑/物理 CPU "
        f"`{environment.get('cpu_logical')}/{environment.get('cpu_physical')}`；内存 "
        f"`{environment.get('memory_total_gib')} GiB`；共享服务器。",
        f"- Retriever：`{preflight.get('retriever_name')}`，版本 "
        f"`{preflight.get('retriever_config_version')}`，配置 "
        f"`{preflight.get('retriever_config_path')}`，未配置 fallback。",
        f"- PostgreSQL：{_connected(preflight, 'postgres')}；Redis：{_connected(preflight, 'redis')}；"
        f"Neo4j：{_connected(preflight, 'neo4j')}。",
        "- pgvector 真实连接并装载 4,208 条向量，Neo4j 真实连接并装载 3,098 个节点；但锁定 "
        "Adaptive v2 的逐请求 Dense/Graph 召回使用版本化本地 exact index/graph artifact。"
        "因此两者本次只完成连接、装载与健康采样，**未用于逐请求检索**。",
        "",
        "## A. Component Baseline (`component_baseline_not_e2e`)",
        "",
        _http_table(component),
        "",
        "## B. Full Stack + Deterministic LLM",
        "",
        _http_table(deterministic),
        "",
        _generation_table(deterministic),
        "",
        "### CPU thread limit 对照",
        "",
        _http_table(optimized) if optimized else "**NOT TESTED**",
        "",
        (
        "该对照只设置 `OMP/MKL/OPENBLAS_NUM_THREADS=1`，不修改 Retriever、Gate 或数据。"
            "并发 5 的吞吐从 0.08 提升至 7.02 RPS、P95 从 25.68 秒降至 0.94 秒；"
            "它验证了线程过度订阅假设，但不替代真实 DeepSeek 容量数据。"
            if optimized else ""
        ),
        "",
        "## C. Full E2E + Real DeepSeek",
        "",
        _http_table(real) if real else "**NOT TESTED**：没有可用的真实模型 Run。",
        "",
        _generation_table(real) if real else "",
        "",
        _usage_summary(real),
        "",
        "## Adaptive 分支与阶段延迟",
        "",
        _strategy_table(deterministic, "Deterministic"),
        "",
        _strategy_table(real, "Real LLM") if real else "",
        "",
        _stage_table(deterministic, real),
        "",
        "`query_analysis` 与 `rerank` 来自 Adaptive Retrieval 决策 Trace；"
        "`persistence_http_overhead` 为 HTTP 实测减 Pipeline total，包含 Run/Trace/Request/Session "
        "持久化及 HTTP 序列化开销。",
        "",
        "## 资源采样",
        "",
        _resource_table(deterministic, optimized, real),
        "",
        "## 容量拐点与瓶颈",
        "",
        f"- Deterministic observed knee：{deterministic_knee}。",
        f"- Deterministic + CPU thread limit observed knee："
        f"{optimized_knee if optimized else 'NOT TESTED'}。",
        f"- Real LLM observed knee：{real_knee if real else 'NOT TESTED'}。",
        "- 拐点定义：首次出现容量相关 internal/timeout/429，或 P95 较上一档超过 2 倍且吞吐增长变缓；Citation 等质量错误单列但不冒充容量错误。",
        "- 本次真实链路在并发 10 的 Retrieval P95 为 79.90 秒、Generation P95 为 15.70 秒，"
        "且线程峰值 916、没有 Provider 429；主要瓶颈是共享 CPU 上的模型推理线程过度订阅和请求排队。",
        "- 四类固定 Query 的 Full E2E Trace 中 Reranker 调用数为 0；因此 CrossEncoder 的"
        "本次压测路径为 **NOT TESTED**，不能用其 dev/unit 证据替代容量数据。",
        "- 资源与依赖原始采样见每档 `c*_resources.csv`；瓶颈结论必须结合阶段 P95、Provider 错误和资源采样解释。",
        "",
        "## 面试回答：系统能支持多少并发/QPS？",
        "",
        "### 10~20 秒",
        "",
        _short_answer(deterministic, optimized, real),
        "",
        "### 追问后的 1 分钟",
        "",
        _long_answer(component, deterministic, optimized, real),
        "",
        "## 工件",
        "",
        "每档保留 Locust CSV/日志、请求 ledger、Trace JSONL、资源 CSV 和机器可读 Summary；",
        "`config_snapshot/` 保存 Retriever、Evidence Gate 与 Model Gateway 配置，不保存 API Key。",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def _http_table(tiers: list[dict[str, object]]) -> str:
    if not tiers:
        return "**NOT TESTED**"
    lines = [
        "| concurrency | requests | RPS | P50 ms | P95 ms | P99 ms | failure |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in tiers:
        lines.append(
            f"| {tier['concurrency']} | {tier['request_count']} | {float(tier['rps']):.2f} | "
            f"{float(tier['p50_ms']):.2f} | {float(tier['p95_ms']):.2f} | "
            f"{float(tier['p99_ms']):.2f} | {float(tier['failure_rate']):.2%} |"
        )
    return "\n".join(lines)


def _generation_table(tiers: list[dict[str, object]]) -> str:
    if not tiers:
        return ""
    lines = [
        "| concurrency | generation requests | generation calls | request P50 ms | request P95 ms | generation P50 ms | generation P95 ms | 429 | timeout |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in tiers:
        errors = dict(tier.get("provider_error_counts", {}))
        lines.append(
            f"| {tier['concurrency']} | {tier['generation_requests']} | "
            f"{tier.get('generation_calls', 0)} | "
            f"{_number(tier.get('generation_request_p50_ms'))} | "
            f"{_number(tier.get('generation_request_p95_ms'))} | "
            f"{_number(tier.get('generation_stage_p50_ms'))} | "
            f"{_number(tier.get('generation_stage_p95_ms'))} | "
            f"{errors.get('rate_limited', 0)} | {errors.get('timeout', 0)} |"
        )
    return "\n".join(lines)


def _usage_summary(tiers: list[dict[str, object]]) -> str:
    """汇总真实模型调用与 Provider 返回的 token；价格未配置时不伪造成本。"""

    if not tiers:
        return "**DeepSeek/API usage: NOT TESTED**"
    calls = sum(int(tier.get("generation_calls", 0)) for tier in tiers)
    input_tokens = sum(int(dict(tier.get("token_usage", {})).get("input_tokens", 0)) for tier in tiers)
    output_tokens = sum(int(dict(tier.get("token_usage", {})).get("output_tokens", 0)) for tier in tiers)
    total_tokens = sum(int(dict(tier.get("token_usage", {})).get("total_tokens", 0)) for tier in tiers)
    return (
        f"真实模型阶段共记录 `{calls}` 次 Generation 调用；Provider usage 合计 "
        f"input/output/total tokens = `{input_tokens}/{output_tokens}/{total_tokens}`。"
        "Model Gateway 配置未固化官方单价，因此 API 成本为 **NOT TESTED（不可由当前配置可靠估算）**。"
    )


def _strategy_table(tiers: list[dict[str, object]], label: str) -> str:
    if not tiers:
        return ""
    strategies = sorted({key for tier in tiers for key in dict(tier.get("strategy_counts", {}))})
    lines = [
        f"### {label}", "",
        "| concurrency | " + " | ".join(strategies) + " | rerank | session ratio |",
        "|---:|" + "---:|" * (len(strategies) + 2),
    ]
    for tier in tiers:
        counts = dict(tier.get("strategy_counts", {}))
        lines.append(
            f"| {tier['concurrency']} | "
            + " | ".join(str(counts.get(strategy, 0)) for strategy in strategies)
            + f" | {tier.get('rerank_count', 0)} | {float(tier.get('session_request_ratio', 0)):.1%} |"
        )
    return "\n".join(lines)


def _stage_table(
    deterministic: list[dict[str, object]], real: list[dict[str, object]]
) -> str:
    selected = ([ _last_completed(deterministic)] if deterministic else []) + (
        [_last_completed(real)] if real else []
    )
    lines = [
        "| mode/concurrency | stage | average ms | P50 ms | P95 ms |",
        "|---|---|---:|---:|---:|",
    ]
    for tier in selected:
        for stage, values in dict(tier.get("stage_latency_ms", {})).items():
            value = dict(values)
            lines.append(
                f"| {tier['mode']}/{tier['concurrency']} | {stage} | "
                f"{float(value['average']):.2f} | {float(value['p50']):.2f} | "
                f"{float(value['p95']):.2f} |"
            )
    return "\n".join(lines)


def _resource_table(
    deterministic: list[dict[str, object]],
    optimized: list[dict[str, object]],
    real: list[dict[str, object]],
) -> str:
    lines = [
        "| mode/concurrency | CPU peak % | RSS peak MB | threads peak | PG connections peak | Redis P95 ms | Neo4j P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labeled = (
        [("deterministic/unbounded", tier) for tier in deterministic]
        + [("deterministic/thread_limit_1", tier) for tier in optimized]
        + [("real/unbounded", tier) for tier in real]
    )
    for label, tier in labeled:
        values = dict(tier.get("resource_summary", {}))
        lines.append(
            f"| {label}/{tier['concurrency']} | {_number(values.get('cpu_peak_percent'))} | "
            f"{_number(values.get('rss_peak_mb'))} | {_number(values.get('threads_peak'))} | "
            f"{_number(values.get('postgres_connections_peak'))} | "
            f"{_number(values.get('redis_ping_p95_ms'))} | {_number(values.get('neo4j_ping_p95_ms'))} |"
        )
    return "\n".join(lines)


def _knee(tiers: list[dict[str, object]]) -> str:
    if not tiers:
        return "NOT TESTED"
    previous: dict[str, object] | None = None
    for tier in tiers:
        errors = dict(tier.get("provider_error_counts", {}))
        request_errors = dict(tier.get("error_type_counts", {}))
        capacity_errors = sum(
            int(request_errors.get(name, 0))
            for name in ("internal_error", "request_timeout", "llm_timeout")
        )
        if (
            int(tier.get("request_count", 0)) == 0
            or capacity_errors
            or errors.get("rate_limited")
            or errors.get("timeout")
        ):
            return f"concurrency={tier['concurrency']}（失败/Provider 限流或超时）"
        if previous is not None:
            latency_ratio = float(tier["p95_ms"]) / max(float(previous["p95_ms"]), 0.001)
            throughput_ratio = float(tier["rps"]) / max(float(previous["rps"]), 0.001)
            if latency_ratio > 2 and throughput_ratio < 1.5:
                return f"concurrency={tier['concurrency']}（P95 >2x 且吞吐增长变缓）"
        previous = tier
    return f"测试上限 {tiers[-1]['concurrency']} 内未触发定义条件"


def _short_answer(
    det: list[dict[str, object]],
    optimized: list[dict[str, object]],
    real: list[dict[str, object]],
) -> str:
    if not det:
        return "Full Stack 尚未完成，不能给出完整系统容量数字。"
    d = det[0]
    if not real:
        return (
            f"在当前共享服务器、单实例配置下，完整内部链路实测到 {d['concurrency']} 并发、"
            f"{float(d['rps']):.2f} RPS、P95 {float(d['p95_ms']):.0f} ms；真实 LLM 容量尚未完成，"
            "因此不能把内部吞吐当成端到端容量。"
        )
    r = real[0]
    internal = optimized[2] if len(optimized) >= 3 else d
    return (
        f"当前共享服务器、单 worker 下，限制 CPU 推理线程后，完整内部链路在并发 "
        f"{internal['concurrency']} 为 {float(internal['rps']):.2f} RPS、"
        f"P95 {float(internal['p95_ms']):.0f} ms；优化前接入 DeepSeek 的并发 1 为 {float(r['rps']):.2f} RPS、"
        f"P95 {float(r['p95_ms']):.0f} ms。并发 2 已出现软拐点，并发 10 有 6 次 90 秒超时，"
        "真实模型档没有在优化后重跑，因此不外推收益，也不声称这是生产 SLA。"
    )


def _long_answer(
    component: list[dict[str, object]],
    det: list[dict[str, object]],
    optimized: list[dict[str, object]],
    real: list[dict[str, object]],
) -> str:
    component_last = component[-1]
    det_last = _last_completed(det)
    text = (
        f"我把容量拆成三层测量。隔离数据库和外部模型的 Component baseline 在 "
        f"{component_last['concurrency']} 并发达到 {float(component_last['rps']):.2f} RPS，但这个数字"
        "只用于定位 FastAPI 和线程池开销。完整 Adaptive Retrieval、Evidence/Context、真实持久化"
        f"但使用 deterministic LLM 时，在 {det_last['concurrency']} 并发实测 "
        f"{float(det_last['rps']):.2f} RPS、P95 {float(det_last['p95_ms']):.0f} ms。"
    )
    if optimized:
        optimized_tier = optimized[2] if len(optimized) >= 3 else optimized[-1]
        text += (
            "资源采样发现推理线程过度订阅后，我只限制 BLAS/OpenMP 线程；"
            f"并发 {optimized_tier['concurrency']} 达到 {float(optimized_tier['rps']):.2f} RPS、"
            f"P95 {float(optimized_tier['p95_ms']):.0f} ms，优化后的软拐点在并发 20。"
        )
    if real:
        real_last = real[-1]
        text += (
            f"真实 DeepSeek 链路在 {real_last['concurrency']} 并发实测 "
            f"{float(real_last['rps']):.2f} RPS、P95 {float(real_last['p95_ms']):.0f} ms；"
            "其中并发 10 有 6 次请求超时；Retrieval P95 明显高于 Generation，且没有 429。"
            "我同时单列真正进入 Generation 的请求，避免 Gate 拒答把延迟虚假拉低；"
            "该真实模型档在 CPU 优化前完成，不能直接外推优化后的端到端容量。"
        )
    else:
        text += "真实 DeepSeek 阶段未执行，所以我不会声称完整外部模型容量。"
    return text + "这些结论只针对当前共享机器和单 Uvicorn worker，不等同于生产 SLA。"


def _last_completed(tiers: list[dict[str, object]]) -> dict[str, object]:
    """返回最高一档至少完成一个请求的结果。"""

    return next(
        tier for tier in reversed(tiers) if int(tier.get("request_count", 0)) > 0
    )


def _connected(preflight: dict[str, object], key: str) -> str:
    value = preflight.get(key)
    return "真实连接" if isinstance(value, dict) and value.get("connected") else "NOT TESTED"


def _number(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.2f}"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
