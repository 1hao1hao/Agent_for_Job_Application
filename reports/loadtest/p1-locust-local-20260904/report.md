# EvalRAG 本地 Locust 压测报告

## 结论

这是单台共享校园服务器上的工程压测，不是生产 SLA。Deterministic 模式在 1/5/10/20/50
并发下共完成 18,988 次 `/v1/query` 请求，HTTP/业务状态失败数均为 0。吞吐在测试范围内持续
增长，但从 20 增加到 50 并发时，RPS 只从 183.18 增至 320.16，而 P95 从 21 ms 增至
120 ms，明显的尾延迟拐点位于 20 到 50 并发之间。

## 环境与口径

- 时间：2026-09-04，服务器与 Locust 同机。
- CPU：2 x Intel Xeon Gold 6458Q，共 64 个在线核心；该节点为共享环境。
- 内存：503 GiB，总可用内存在开始前约 364 GiB。
- Python 3.11.7，Locust 2.46.4。
- 服务：单 Uvicorn worker；本机回环 HTTP；默认 `ThreadPoolExecutor` 实测 32 workers。
- 每档 30 秒，用户等待 50-150 ms；Query 均匀覆盖 exact fact、semantic、multi-source、
  relation reasoning。
- 检索固定为 BM25 v0.3。真实执行 FastAPI、`asyncio.to_thread`、QueryService、Rule Router、
  BM25、Evidence Gate、Context、Generator parser、Citation Validator 和 Trace 构造。
- 外部 LLM 替换为 deterministic client；PostgreSQL/Redis 替换为线程安全计数 adapter，Trace
  不落盘。因此结果不包含数据库连接与磁盘 Trace I/O，不能代表 Docker Compose 完整服务栈。

## Mode A：Deterministic Pipeline

| concurrency | requests | RPS | average | P50 | P95 | P99 | failure rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 269 | 9.27 | 5.35 ms | 5 ms | 8 ms | 10 ms | 0.00% |
| 5 | 1,382 | 47.59 | 4.49 ms | 4 ms | 9 ms | 14 ms | 0.00% |
| 10 | 2,733 | 94.16 | 5.22 ms | 5 ms | 12 ms | 19 ms | 0.00% |
| 20 | 5,316 | 183.18 | 8.26 ms | 6 ms | 21 ms | 32 ms | 0.00% |
| 50 | 9,288 | 320.16 | 55.78 ms | 55 ms | 120 ms | 140 ms | 0.00% |

服务进程 130 个采样点的平均/峰值 CPU 为 17.12%/50.00%，平均/峰值 RSS 为
106.93/112.62 MiB。CPU 数值是单进程 `ps` 口径，且跨越启动和全部阶梯，只用于辅助解释，
不能当作整机利用率。

## Mode B：真实 LLM 小流量验证

真实 DeepSeek 配置运行 1 并发、15 秒，共 10 次请求，0 失败，平均 1,220.46 ms，P50 8 ms，
P95/P99 约 6,500 ms，RPS 0.77。四类问题中部分被 Evidence Gate 在生成前快速拒答，进入模型
生成的路径落在约 1.5-6.5 秒区间。因此 10 条样本的 P95 很不稳定，只能证明真实模型路径可用，
不能作为容量结论；本轮也没有采集 provider token/cost。

## Observed Bottlenecks

1. 1-20 并发近似线性增长；50 并发超过默认 32 个线程后，RPS 增长变慢且 P95/P99 约放大
   5.7/4.4 倍。这与 `asyncio.to_thread` 排队相符，但当前实验不能排除 Python GIL、同步访问日志
   和 BM25 本地计算的共同影响。
2. 服务器 CPU 未接近整机饱和，说明本轮首先碰到的是单进程执行/排队边界，而非 64 核整机算力。
3. 真实模型路径的秒级耗时远大于 deterministic 本地链路的毫秒级耗时；真实 E2E 首要变量是
   LLM 网络/服务延迟。
4. PostgreSQL、Dense、Graph 和 CrossEncoder 未纳入本轮 HTTP 压测，不能据此断言它们不是
   瓶颈。

## Recommended Improvements

- 在 API 层显式设置 Pipeline 并发上限与过载返回，避免请求无限进入默认线程池排队。
- 对单/多 Uvicorn worker 做同口径对照；注意每个 worker 会复制本地索引内存。
- 对 PostgreSQL 使用连接池，并单独复测完整 Compose 链路，量化当前短连接成本。
- Dense/CrossEncoder 等 CPU-heavy 阶段使用独立进程或受控推理服务，避免阻塞同一 Python 进程。
- 真实 LLM 通过 Model Gateway 的 semaphore、rate limit 和 timeout 控制外部容量，而不是按本地
  320 RPS 放行。

## 复现

```bash
python -m pip install -r requirements-loadtest.txt
OUTPUT_DIR=reports/loadtest/local-run \
  DURATION=30s \
  LOCUST_PYTHON=python \
  scripts/run_local_loadtest.sh
```

原始数据为同目录 `c*_stats.csv`、`c*_stats_history.csv`、`c*_failures.csv` 和
`server_resources.csv`；真实 LLM 原始结果位于
`../p1-locust-real-llm-20260904/`。
