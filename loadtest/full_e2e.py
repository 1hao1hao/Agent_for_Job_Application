from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
from threading import Event, Thread
from time import perf_counter, sleep
from typing import Iterable
from urllib.request import urlopen


STAGES = (
    "routing", "retrieval", "evidence", "context", "generation", "validation", "total"
)


@dataclass(frozen=True)
class TierSummary:
    """一档并发的 HTTP、Generation、策略与错误汇总。"""

    mode: str
    concurrency: int
    request_count: int
    generation_requests: int
    generation_calls: int
    rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    failure_rate: float
    generation_request_p50_ms: float | None
    generation_request_p95_ms: float | None
    generation_stage_p50_ms: float | None
    generation_stage_p95_ms: float | None
    status_counts: dict[str, int]
    error_type_counts: dict[str, int]
    case_type_counts: dict[str, int]
    strategy_counts: dict[str, int]
    rerank_count: int
    provider_error_counts: dict[str, int]
    token_usage: dict[str, int]
    stage_latency_ms: dict[str, dict[str, float]]
    session_request_ratio: float
    resource_summary: dict[str, float | int | None]


def percentile(values: Iterable[float], quantile: float) -> float:
    """用 nearest-rank 计算可复现百分位；空输入返回 0。"""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[position]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """读取压测 ledger 或 Trace JSONL。"""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collect_traces(host: str, ledger_path: Path, output_path: Path) -> list[dict[str, object]]:
    """根据 Locust 返回的 trace_id 调用正式 Trace API，保存本档请求证据。"""

    traces: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in read_jsonl(ledger_path):
        trace_id = str(record.get("trace_id") or "")
        if not trace_id or trace_id in seen:
            continue
        seen.add(trace_id)
        try:
            with urlopen(f"{host}/v1/traces/{trace_id}", timeout=10) as response:
                trace = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue
        trace["loadtest_case_type"] = record.get("case_type")
        trace["loadtest_http_latency_ms"] = record.get("latency_ms")
        trace["loadtest_session_request"] = record.get("session_request", False)
        traces.append(trace)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for trace in traces:
            output.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return traces


def summarize_tier(
    mode: str,
    concurrency: int,
    ledger: list[dict[str, object]],
    traces: list[dict[str, object]],
    rps: float,
    resource_summary: dict[str, float | int | None] | None = None,
) -> TierSummary:
    """合并 HTTP ledger 与 AgentTrace，区分总请求和真实 Generation 请求。"""

    trace_by_request = {str(item["request_id"]): item for item in traces}
    latencies = [float(item.get("latency_ms", 0.0)) for item in ledger]
    failures = [item for item in ledger if int(item.get("http_status", 0)) != 200]
    generation_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    strategies: Counter[str] = Counter()
    statuses: Counter[str] = Counter(str(item.get("response_status")) for item in ledger)
    provider_errors: Counter[str] = Counter()
    stage_values: dict[str, list[float]] = defaultdict(list)
    rerank_count = 0
    generation_calls = 0
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for record in ledger:
        trace = trace_by_request.get(str(record.get("request_id")))
        if trace is None:
            continue
        latency_map = dict(trace.get("latency_ms", {}))
        for stage in STAGES:
            stage_values[stage].append(float(latency_map.get(stage, 0.0)))
        pipeline_total = float(latency_map.get("total", 0.0))
        http_total = float(record.get("latency_ms", 0.0))
        stage_values["persistence_http_overhead"].append(max(0.0, http_total - pipeline_total))
        decision = dict(dict(trace.get("retrieval", {})).get("decision", {}))
        stage_values["query_analysis"].append(
            float(decision.get("query_analysis_latency_ms", 0.0))
        )
        stage_values["rerank"].append(float(decision.get("rerank_latency_ms", 0.0)))
        strategy = str(decision.get("selected_strategy") or decision.get("strategy") or "unknown")
        strategies[strategy] += 1
        rerank_count += int(bool(decision.get("rerank_invoked", False)))
        attempts = list(trace.get("attempts", []))
        generation_attempts = [
            attempt for attempt in attempts
            if isinstance(attempt, dict)
            and attempt.get("type") in {"initial_generation", "format_repair", "generation"}
        ]
        if generation_attempts:
            generation_pairs.append((record, trace))
        generation_calls += len(generation_attempts)
        for attempt in generation_attempts:
            usage = attempt.get("token_usage")
            if isinstance(usage, dict):
                for key in tokens:
                    value = usage.get(key)
                    if isinstance(value, (int, float)):
                        tokens[key] += int(value)
            gateway = attempt.get("model_gateway")
            if isinstance(gateway, dict):
                for provider_attempt in gateway.get("attempts", []):
                    if isinstance(provider_attempt, dict) and provider_attempt.get("status") != "succeeded":
                        provider_errors[str(provider_attempt.get("reason", "unknown"))] += 1

    generation_http = [float(item[0].get("latency_ms", 0.0)) for item in generation_pairs]
    generation_stage = [
        float(dict(item[1].get("latency_ms", {})).get("generation", 0.0))
        for item in generation_pairs
    ]
    stage_summary = {
        stage: {
            "average": statistics.fmean(values) if values else 0.0,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }
        for stage, values in stage_values.items()
    }
    return TierSummary(
        mode=mode,
        concurrency=concurrency,
        request_count=len(ledger),
        generation_requests=len(generation_pairs),
        generation_calls=generation_calls,
        rps=rps,
        p50_ms=percentile(latencies, 0.50),
        p95_ms=percentile(latencies, 0.95),
        p99_ms=percentile(latencies, 0.99),
        failure_rate=len(failures) / len(ledger) if ledger else 1.0,
        generation_request_p50_ms=percentile(generation_http, 0.50) if generation_http else None,
        generation_request_p95_ms=percentile(generation_http, 0.95) if generation_http else None,
        generation_stage_p50_ms=percentile(generation_stage, 0.50) if generation_stage else None,
        generation_stage_p95_ms=percentile(generation_stage, 0.95) if generation_stage else None,
        status_counts=dict(statuses),
        error_type_counts=dict(
            Counter(
                str(item.get("error_type"))
                for item in ledger
                if item.get("error_type") is not None
            )
        ),
        case_type_counts=dict(Counter(str(item.get("case_type")) for item in ledger)),
        strategy_counts=dict(strategies),
        rerank_count=rerank_count,
        provider_error_counts=dict(provider_errors),
        token_usage=tokens,
        stage_latency_ms=stage_summary,
        session_request_ratio=(
            sum(bool(item.get("session_request")) for item in ledger) / len(ledger)
            if ledger else 0.0
        ),
        resource_summary=resource_summary or {},
    )


def read_locust_rps(path: Path) -> float:
    """从 Locust aggregate 行读取实际吞吐。"""

    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    aggregate = next((row for row in rows if row.get("Name") == "Aggregated"), rows[-1])
    return float(aggregate.get("Requests/s", 0.0))


def summarize_resources(path: Path) -> dict[str, float | int | None]:
    """把每秒资源采样压缩为一档可比较的峰值与依赖状态。"""

    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        return {}

    def numeric(column: str) -> list[float]:
        return [float(row[column]) for row in rows if row.get(column) not in {None, ""}]

    cpu = numeric("cpu_percent")
    rss = numeric("rss_mb")
    threads = numeric("threads")
    pg_connections = numeric("postgres_connections")
    redis_ping = numeric("redis_ping_ms")
    neo4j_ping = numeric("neo4j_ping_ms")
    return {
        "sample_count": len(rows),
        "cpu_peak_percent": max(cpu) if cpu else None,
        "rss_peak_mb": max(rss) if rss else None,
        "threads_peak": int(max(threads)) if threads else None,
        "postgres_connections_peak": int(max(pg_connections)) if pg_connections else None,
        "redis_ping_p95_ms": percentile(redis_ping, 0.95) if redis_ping else None,
        "neo4j_ping_p95_ms": percentile(neo4j_ping, 0.95) if neo4j_ping else None,
    }


class ResourceMonitor:
    """低侵入采样 API 资源及 PostgreSQL/Redis/Neo4j 健康延迟。"""

    def __init__(self, pid: int, output_path: Path, level: str) -> None:
        self.pid = pid
        self.output_path = output_path
        self.level = level
        self.stop_event = Event()
        self.thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)

    def _run(self) -> None:
        import psutil

        process = psutil.Process(self.pid)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow([
                "timestamp", "level", "cpu_percent", "rss_mb", "threads",
                "postgres_connections", "redis_ping_ms", "neo4j_ping_ms",
            ])
            process.cpu_percent(None)
            while not self.stop_event.wait(1.0):
                try:
                    children = process.children(recursive=True)
                    cpu = process.cpu_percent(None) + sum(child.cpu_percent(None) for child in children)
                    rss = process.memory_info().rss + sum(child.memory_info().rss for child in children)
                    threads = process.num_threads() + sum(child.num_threads() for child in children)
                    writer.writerow([
                        perf_counter(), self.level, cpu, rss / 1024 / 1024, threads,
                        _postgres_connections(), _redis_ping_ms(), _neo4j_ping_ms(),
                    ])
                    output.flush()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break


def environment_snapshot() -> dict[str, object]:
    """记录不含凭证的压测机器与运行时信息。"""

    import psutil

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_logical": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_total_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        "shared_host": True,
        "cpu_thread_limit": os.environ.get("EVALRAG_CPU_THREADS", "unbounded"),
        "database_url_configured": bool(os.environ.get("DATABASE_URL")),
        "redis_url_configured": bool(os.environ.get("REDIS_URL")),
        "neo4j_uri_configured": bool(os.environ.get("NEO4J_URI")),
        "deepseek_key_configured": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }


def _postgres_connections() -> int | None:
    try:
        import psycopg
        with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=1) as connection:
            return int(connection.execute("SELECT count(*) FROM pg_stat_activity").fetchone()[0])
    except Exception:
        return None


def _redis_ping_ms() -> float | None:
    try:
        from redis import Redis
        started = perf_counter()
        Redis.from_url(os.environ["REDIS_URL"], socket_timeout=1).ping()
        return (perf_counter() - started) * 1000
    except Exception:
        return None


def _neo4j_ping_ms() -> float | None:
    try:
        from neo4j import GraphDatabase
        started = perf_counter()
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        )
        driver.verify_connectivity()
        driver.close()
        return (perf_counter() - started) * 1000
    except Exception:
        return None


def dump_summary(summary: TierSummary, path: Path) -> None:
    """保存一档机器可读 Summary。"""

    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
