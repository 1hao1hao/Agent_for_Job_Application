from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import site
import subprocess
import sys
from time import sleep, time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loadtest.full_e2e import (  # noqa: E402
    ResourceMonitor,
    collect_traces,
    dump_summary,
    environment_snapshot,
    read_jsonl,
    read_locust_rps,
    summarize_resources,
    summarize_tier,
)


def main() -> int:
    """通过正式 Runtime 运行 deterministic/real 两层 Full E2E 压测。

    输入是已启动的 PostgreSQL、Redis、Neo4j、锁定 Retriever 工件与环境变量；脚本
    先 fail-fast 校验依赖，再启动 `create_runtime_app()`。每档运行 Locust、采样资源、
    通过 Trace API 回收阶段数据并生成 Summary；失败率、超时或 429 到达容量边界时停止升压。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deterministic", "real"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", default="")
    parser.add_argument("--duration", default="")
    parser.add_argument("--host", default="http://127.0.0.1:8020")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--spawn-rate", type=float, default=10.0)
    parser.add_argument(
        "--cpu-threads", type=int, default=0,
        help="limit BLAS/OpenMP inference threads per process; 0 keeps current defaults",
    )
    args = parser.parse_args()
    # 校园服务器注入了全局 HTTP 代理；本机正式 API 必须绕过代理，否则预检会假超时。
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    concurrencies = _parse_concurrencies(
        args.concurrencies
        or ("1,2,5,10,20" if args.mode == "real" else "1,5,10,20,50")
    )
    duration = args.duration or ("90s" if args.mode == "real" else "45s")
    if args.cpu_threads < 0:
        raise ValueError("cpu-threads must be non-negative")
    if args.cpu_threads:
        os.environ["EVALRAG_CPU_THREADS"] = str(args.cpu_threads)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(args.cpu_threads)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.output.mkdir(parents=True, exist_ok=True)

    preflight = _preflight(args.mode)
    (args.output / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "environment.json").write_text(
        json.dumps(environment_snapshot(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _snapshot_configs(args.output)

    env = os.environ.copy()
    env.update({
        # 当前 Conda 自带非 namespace 的 zope 包；把 user site 放前面才能加载 Locust/gevent。
        "PYTHONPATH": f"{site.getusersitepackages()}:{SRC}:{ROOT}",
        "EVALRAG_PROJECT_ROOT": str(ROOT),
        "EVALRAG_LLM_BACKEND": "deepseek" if args.mode == "real" else "fake",
        "EVALRAG_RETRIEVER_CONFIG": str(
            ROOT / "configs/retrieval/adaptive_v2_v0.3.json"
        ),
        "EVALRAG_MODEL": (
            "deepseek-v4-flash" if args.mode == "real" else "deterministic-demo"
        ),
    })
    env.pop("EVALRAG_RETRIEVER_FALLBACK_CONFIG", None)
    server_log = (args.output / "server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "intern_rag.serving.runtime:create_runtime_app", "--factory",
            "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1",
        ],
        cwd=ROOT,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_for_health(args.host, server)
        sessions = _prepare_sessions(args.host, count=5)
        env["EVALRAG_LOADTEST_SESSIONS"] = json.dumps(sessions)
        summaries: list[dict[str, object]] = []
        for concurrency in concurrencies:
            prefix = args.output / f"c{concurrency}"
            ledger = args.output / f"c{concurrency}_ledger.jsonl"
            if ledger.exists():
                ledger.unlink()
            tier_env = {**env, "EVALRAG_LOADTEST_LEDGER": str(ledger)}
            monitor = ResourceMonitor(
                server.pid, args.output / f"c{concurrency}_resources.csv", str(concurrency)
            )
            monitor.start()
            command = [
                sys.executable, "-c",
                (
                    "import site,zope,runpy;"
                    "zope.__path__.append(site.getusersitepackages() + '/zope');"
                    "runpy.run_module('locust', run_name='__main__')"
                ),
                "-f", "loadtest/locustfile.py",
                "--headless", "--only-summary", "--host", args.host,
                "--users", str(concurrency),
                "--spawn-rate", str(min(args.spawn_rate, concurrency)),
                "--run-time", duration,
                "--csv", str(prefix),
            ]
            with (args.output / f"c{concurrency}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, cwd=ROOT, env=tier_env, stdout=log,
                    stderr=subprocess.STDOUT, check=False,
                )
            monitor.stop()
            traces = collect_traces(
                args.host, ledger, args.output / f"c{concurrency}_traces.jsonl"
            )
            summary = summarize_tier(
                args.mode,
                concurrency,
                read_jsonl(ledger),
                traces,
                read_locust_rps(Path(f"{prefix}_stats.csv")),
                summarize_resources(args.output / f"c{concurrency}_resources.csv"),
            )
            dump_summary(summary, args.output / f"c{concurrency}_summary.json")
            summaries.append(json.loads(json.dumps(summary.__dict__)))
            # Locust 遇到业务级 502（如 citation_invalid）也返回 1；容量停止条件由
            # 结构化错误类型判断，避免把模型质量问题误当成并发边界。
            if completed.returncode not in {0, 1} or _capacity_boundary(summary.__dict__):
                break
        summary_payload = {"mode": args.mode, "tiers": summaries}
        (args.output / "summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        deepseek = dict(preflight["deepseek"])
        deepseek["called"] = args.mode == "real" and any(
            int(item.get("generation_calls", 0)) > 0 for item in summaries
        )
        deepseek["total_generation_calls"] = (
            sum(int(item.get("generation_calls", 0)) for item in summaries)
            if args.mode == "real" else 0
        )
        preflight["deepseek"] = deepseek
        (args.output / "preflight.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
        server_log.close()
    print(f"full E2E artifacts: {args.output}")
    return 0


def _preflight(mode: str) -> dict[str, object]:
    """验证正式依赖与锁定工件，任何缺失均中止且不产生 Full E2E 名义结果。"""

    required_env = ("DATABASE_URL", "REDIS_URL", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
    missing = [key for key in required_env if not os.environ.get(key, "").strip()]
    if mode == "real" and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        missing.append("DEEPSEEK_API_KEY")
    if missing:
        raise RuntimeError(f"missing required environment variables: {', '.join(missing)}")
    paths = {
        "retriever_config": ROOT / "configs/retrieval/adaptive_v2_v0.3.json",
        "bm25_index": ROOT / "data/processed/indexes/evalrag_v0.3/bm25-v1/index.json",
        "dense_index": ROOT / "data/processed/indexes/evalrag_v0.3/bge-small-zh-v1.5/index.json",
        "graph_index": ROOT / "data/processed/graphs/evalrag_v0.3/job-skill-experience-v0.2.json",
    }
    missing_paths = [str(path) for path in paths.values() if not path.exists()]
    if missing_paths:
        raise RuntimeError(f"missing locked artifacts: {missing_paths}")

    import psycopg
    from redis import Redis
    from neo4j import GraphDatabase

    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=5) as connection:
        postgres_version = str(connection.execute("SHOW server_version").fetchone()[0])
        vector_enabled = bool(
            connection.execute("SELECT 1 FROM pg_extension WHERE extname='vector'").fetchone()
        )
        vector_rows = int(
            connection.execute(
                "SELECT count(*) FROM rag_chunk_embeddings "
                "WHERE dataset_version='evalrag_v0.3'"
            ).fetchone()[0]
        )
    redis = Redis.from_url(os.environ["REDIS_URL"], socket_timeout=5)
    redis_version = str(redis.info("server")["redis_version"])
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    driver.verify_connectivity()
    with driver.session() as session:
        neo4j_version = str(session.run(
            "CALL dbms.components() YIELD versions RETURN versions[0] AS version"
        ).single()["version"])
        neo4j_nodes = int(session.run(
            "MATCH (n:EvalRAGNode {dataset_version: 'evalrag_v0.3'}) RETURN count(n) AS count"
        ).single()["count"])
    driver.close()
    config = json.loads(paths["retriever_config"].read_text(encoding="utf-8"))
    return {
        "status": "passed",
        "mode": mode,
        "retriever_name": config["retriever_name"],
        "retriever_config_version": config["config_version"],
        "retriever_config_path": str(paths["retriever_config"].relative_to(ROOT)),
        "fallback_disabled": True,
        "postgres": {"connected": True, "version": postgres_version},
        "pgvector": {
            "connected": vector_enabled,
            "row_count": vector_rows,
            "used_by_locked_retriever": False,
            "reason": "locked Adaptive v2 uses versioned local exact dense index",
        },
        "redis": {"connected": True, "version": redis_version},
        "neo4j": {
            "connected": True,
            "version": neo4j_version,
            "node_count": neo4j_nodes,
            "used_by_locked_retriever": False,
            "reason": "locked Adaptive v2 loads the versioned graph artifact into memory",
        },
        "deepseek": {
            "configured": mode == "real",
            "called": False,
            "total_generation_calls": 0,
        },
    }


def _wait_for_health(host: str, server: subprocess.Popen[object]) -> None:
    for _ in range(180):
        if server.poll() is not None:
            raise RuntimeError("formal API exited during startup; inspect server.log")
        try:
            with urlopen(f"{host}/health", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                return
        except Exception:
            pass
        sleep(1)
    raise TimeoutError("formal API did not become healthy")


def _prepare_sessions(host: str, count: int) -> list[dict[str, str]]:
    """通过正式 HTTP 接口建立压测 Session/Profile，后续约 30% Query 带会话边界。"""

    scopes: list[dict[str, str]] = []
    suffix = str(int(time()))
    for index in range(count):
        user_id = f"loadtest-{suffix}-{index}"
        session = _json_request(
            f"{host}/v1/users/{user_id}/sessions", "POST", {"title": "Full E2E load test"}
        )
        _json_request(
            f"{host}/v1/users/{user_id}/profile",
            "PUT",
            {
                "facts": [
                    {"key": "目标岗位", "value": "大模型应用研发实习生", "source": "loadtest", "confirmed": True}
                ]
            },
        )
        scopes.append({"user_id": user_id, "session_id": str(session["session_id"])})
    return scopes


def _json_request(url: str, method: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url, method=method, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} while preparing load-test state") from error


def _capacity_boundary(summary: dict[str, object]) -> bool:
    errors = dict(summary.get("provider_error_counts", {}))
    request_errors = dict(summary.get("error_type_counts", {}))
    capacity_errors = sum(
        int(request_errors.get(name, 0))
        for name in ("internal_error", "request_timeout", "llm_timeout")
    )
    return (
        int(summary.get("request_count", 0)) == 0
        or capacity_errors > 0
        or int(errors.get("rate_limited", 0)) > 0
        or int(errors.get("timeout", 0)) > 0
    )


def _snapshot_configs(output: Path) -> None:
    snapshot = output / "config_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in (
        ROOT / "configs/retrieval/adaptive_v2_v0.3.json",
        ROOT / "configs/evidence/gate_calibrated_v0.3.json",
        ROOT / "configs/model_gateway/gateway_v0.1.json",
    ):
        shutil.copy2(path, snapshot / path.name)


def _parse_concurrencies(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError("concurrencies must contain positive integers")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
