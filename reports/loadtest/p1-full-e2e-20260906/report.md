# EvalRAG Full E2E Load Test

## 结论边界

本报告区分组件隔离吞吐、完整内部链路上限和真实模型 Provider 边界。所有数字只代表
当前共享服务器、单 Uvicorn worker 与锁定配置，不是生产 SLA。Component 旧结果保留
用于定位 FastAPI/线程池开销，不再冒充 Full E2E。

## 环境与真实性

- Python `3.11.7`；逻辑/物理 CPU `64/64`；内存 `503.36 GiB`；共享服务器。
- Retriever：`graph_adaptive`，版本 `adaptive-retrieval-v2.0`，配置 `configs/retrieval/adaptive_v2_v0.3.json`，未配置 fallback。
- PostgreSQL：真实连接；Redis：真实连接；Neo4j：真实连接。
- pgvector 真实连接并装载 4,208 条向量，Neo4j 真实连接并装载 3,098 个节点；但锁定 Adaptive v2 的逐请求 Dense/Graph 召回使用版本化本地 exact index/graph artifact。因此两者本次只完成连接、装载与健康采样，**未用于逐请求检索**。

## A. Component Baseline (`component_baseline_not_e2e`)

| concurrency | requests | RPS | P50 ms | P95 ms | P99 ms | failure |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 269 | 9.27 | 5.00 | 8.00 | 10.00 | 0.00% |
| 5 | 1382 | 47.59 | 4.00 | 9.00 | 14.00 | 0.00% |
| 10 | 2733 | 94.16 | 5.00 | 12.00 | 19.00 | 0.00% |
| 20 | 5316 | 183.18 | 6.00 | 21.00 | 32.00 | 0.00% |
| 50 | 9288 | 320.16 | 55.00 | 120.00 | 140.00 | 0.00% |

## B. Full Stack + Deterministic LLM

| concurrency | requests | RPS | P50 ms | P95 ms | P99 ms | failure |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 0.36 | 2388.96 | 4727.21 | 4727.21 | 0.00% |
| 5 | 2 | 0.08 | 25172.50 | 25676.27 | 25676.27 | 0.00% |
| 10 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00% |

| concurrency | generation requests | generation calls | request P50 ms | request P95 ms | generation P50 ms | generation P95 ms | 429 | timeout |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 7 | 1893.41 | 4063.56 | 0.12 | 0.24 | 0 | 0 |
| 5 | 2 | 2 | 25172.50 | 25676.27 | 0.11 | 0.13 | 0 | 0 |
| 10 | 0 | 0 | N/A | N/A | N/A | N/A | 0 | 0 |

### CPU thread limit 对照

| concurrency | requests | RPS | P50 ms | P95 ms | P99 ms | failure |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84 | 2.86 | 253.31 | 345.77 | 652.14 | 0.00% |
| 5 | 207 | 7.02 | 604.50 | 941.78 | 1038.46 | 0.00% |
| 10 | 210 | 7.07 | 1297.68 | 1866.40 | 2088.10 | 0.00% |
| 20 | 191 | 6.48 | 2725.19 | 4271.38 | 4675.70 | 0.00% |
| 50 | 173 | 5.91 | 6918.68 | 9432.80 | 10558.42 | 0.00% |

该对照只设置 `OMP/MKL/OPENBLAS_NUM_THREADS=1`，不修改 Retriever、Gate 或数据。并发 5 的吞吐从 0.08 提升至 7.02 RPS、P95 从 25.68 秒降至 0.94 秒；它验证了线程过度订阅假设，但不替代真实 DeepSeek 容量数据。

## C. Full E2E + Real DeepSeek

| concurrency | requests | RPS | P50 ms | P95 ms | P99 ms | failure |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35 | 0.20 | 5351.56 | 7441.12 | 10176.50 | 8.57% |
| 2 | 33 | 0.19 | 11158.00 | 17006.96 | 22311.28 | 6.06% |
| 5 | 26 | 0.14 | 30098.08 | 51902.92 | 53615.47 | 3.85% |
| 10 | 25 | 0.13 | 61423.77 | 90318.39 | 95206.62 | 28.00% |

| concurrency | generation requests | generation calls | request P50 ms | request P95 ms | generation P50 ms | generation P95 ms | 429 | timeout |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21 | 21 | 4584.20 | 7441.12 | 2337.98 | 2886.54 | 0 | 0 |
| 2 | 21 | 21 | 8311.91 | 14519.33 | 2167.72 | 3395.69 | 0 | 0 |
| 5 | 17 | 17 | 26794.43 | 53615.47 | 2586.39 | 4175.20 | 0 | 0 |
| 10 | 13 | 13 | 55315.87 | 61423.77 | 4297.16 | 15703.72 | 0 | 0 |

真实模型阶段共记录 `72` 次 Generation 调用；Provider usage 合计 input/output/total tokens = `106220/14101/120321`。Model Gateway 配置未固化官方单价，因此 API 成本为 **NOT TESTED（不可由当前配置可靠估算）**。

## Adaptive 分支与阶段延迟

### Deterministic

| concurrency | bm25 | graph_hybrid | hybrid | rerank | session ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 4 | 0 | 30.0% |
| 5 | 0 | 0 | 2 | 0 | 0.0% |
| 10 | 0 | 0 | 0 | 0 | 0.0% |

### Real LLM

| concurrency | bm25 | graph_hybrid | hybrid | rerank | session ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 14 | 12 | 0 | 34.3% |
| 2 | 9 | 12 | 12 | 0 | 36.4% |
| 5 | 9 | 9 | 8 | 0 | 34.6% |
| 10 | 5 | 6 | 8 | 0 | 32.0% |

| mode/concurrency | stage | average ms | P50 ms | P95 ms |
|---|---|---:|---:|---:|
| deterministic/5 | routing | 0.02 | 0.02 | 0.02 |
| deterministic/5 | retrieval | 23333.31 | 23043.49 | 23623.12 |
| deterministic/5 | evidence | 0.05 | 0.03 | 0.06 |
| deterministic/5 | context | 0.75 | 0.63 | 0.88 |
| deterministic/5 | generation | 0.12 | 0.11 | 0.13 |
| deterministic/5 | validation | 0.02 | 0.01 | 0.03 |
| deterministic/5 | total | 23334.48 | 23044.84 | 23624.12 |
| deterministic/5 | persistence_http_overhead | 2089.91 | 1548.38 | 2631.43 |
| deterministic/5 | query_analysis | 0.03 | 0.02 | 0.03 |
| deterministic/5 | rerank | 0.00 | 0.00 | 0.00 |
| real/10 | routing | 0.03 | 0.03 | 0.05 |
| real/10 | retrieval | 43700.89 | 46010.44 | 79900.77 |
| real/10 | evidence | 0.06 | 0.04 | 0.11 |
| real/10 | context | 24.37 | 0.68 | 192.32 |
| real/10 | generation | 3718.63 | 4003.20 | 15703.72 |
| real/10 | validation | 0.03 | 0.02 | 0.15 |
| real/10 | total | 47448.13 | 50399.52 | 79901.22 |
| real/10 | persistence_http_overhead | 8276.73 | 8814.60 | 17118.81 |
| real/10 | query_analysis | 0.03 | 0.03 | 0.05 |
| real/10 | rerank | 0.00 | 0.00 | 0.00 |

`query_analysis` 与 `rerank` 来自 Adaptive Retrieval 决策 Trace；`persistence_http_overhead` 为 HTTP 实测减 Pipeline total，包含 Run/Trace/Request/Session 持久化及 HTTP 序列化开销。

## 资源采样

| mode/concurrency | CPU peak % | RSS peak MB | threads peak | PG connections peak | Redis P95 ms | Neo4j P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| deterministic/unbounded/1 | 814.50 | 1055.54 | 80.00 | 7.00 | 92.39 | 101.55 |
| deterministic/unbounded/5 | 811.20 | 1472.41 | 400.00 | 7.00 | 104.90 | 104.55 |
| deterministic/unbounded/10 | 802.80 | 1697.49 | 976.00 | 6.00 | 187.67 | 111.82 |
| deterministic/thread_limit_1/1 | 75.50 | 1061.98 | 9.00 | 6.00 | 3.02 | 6.21 |
| deterministic/thread_limit_1/5 | 192.00 | 1186.19 | 13.00 | 8.00 | 2.80 | 3.48 |
| deterministic/thread_limit_1/10 | 207.90 | 1357.05 | 18.00 | 9.00 | 1.93 | 3.85 |
| deterministic/thread_limit_1/20 | 231.00 | 1710.83 | 28.00 | 11.00 | 1.73 | 5.27 |
| deterministic/thread_limit_1/50 | 220.90 | 2114.01 | 40.00 | 15.00 | 1.29 | 9.77 |
| real/unbounded/1 | 846.90 | 1308.42 | 81.00 | 7.00 | 75.14 | 83.94 |
| real/unbounded/2 | 816.20 | 1631.71 | 146.00 | 6.00 | 83.93 | 97.15 |
| real/unbounded/5 | 828.50 | 2160.49 | 340.00 | 7.00 | 90.43 | 103.13 |
| real/unbounded/10 | 820.60 | 3329.60 | 916.00 | 8.00 | 96.68 | 206.83 |

## 容量拐点与瓶颈

- Deterministic observed knee：concurrency=5（P95 >2x 且吞吐增长变缓）。
- Deterministic + CPU thread limit observed knee：concurrency=20（P95 >2x 且吞吐增长变缓）。
- Real LLM observed knee：concurrency=2（P95 >2x 且吞吐增长变缓）。
- 拐点定义：首次出现容量相关 internal/timeout/429，或 P95 较上一档超过 2 倍且吞吐增长变缓；Citation 等质量错误单列但不冒充容量错误。
- 本次真实链路在并发 10 的 Retrieval P95 为 79.90 秒、Generation P95 为 15.70 秒，且线程峰值 916、没有 Provider 429；主要瓶颈是共享 CPU 上的模型推理线程过度订阅和请求排队。
- 四类固定 Query 的 Full E2E Trace 中 Reranker 调用数为 0；因此 CrossEncoder 的本次压测路径为 **NOT TESTED**，不能用其 dev/unit 证据替代容量数据。
- 资源与依赖原始采样见每档 `c*_resources.csv`；瓶颈结论必须结合阶段 P95、Provider 错误和资源采样解释。

## 面试回答：系统能支持多少并发/QPS？

### 10~20 秒

当前共享服务器、单 worker 下，限制 CPU 推理线程后，完整内部链路在并发 10 为 7.07 RPS、P95 1866 ms；优化前接入 DeepSeek 的并发 1 为 0.20 RPS、P95 7441 ms。并发 2 已出现软拐点，并发 10 有 6 次 90 秒超时，真实模型档没有在优化后重跑，因此不外推收益，也不声称这是生产 SLA。

### 追问后的 1 分钟

我把容量拆成三层测量。隔离数据库和外部模型的 Component baseline 在 50 并发达到 320.16 RPS，但这个数字只用于定位 FastAPI 和线程池开销。完整 Adaptive Retrieval、Evidence/Context、真实持久化但使用 deterministic LLM 时，在 5 并发实测 0.08 RPS、P95 25676 ms。资源采样发现推理线程过度订阅后，我只限制 BLAS/OpenMP 线程；并发 10 达到 7.07 RPS、P95 1866 ms，优化后的软拐点在并发 20。真实 DeepSeek 链路在 10 并发实测 0.13 RPS、P95 90318 ms；其中并发 10 有 6 次请求超时；Retrieval P95 明显高于 Generation，且没有 429。我同时单列真正进入 Generation 的请求，避免 Gate 拒答把延迟虚假拉低；该真实模型档在 CPU 优化前完成，不能直接外推优化后的端到端容量。这些结论只针对当前共享机器和单 Uvicorn worker，不等同于生产 SLA。

## 工件

每档保留 Locust CSV/日志、请求 ledger、Trace JSONL、资源 CSV 和机器可读 Summary；
`config_snapshot/` 保存 Retriever、Evidence Gate 与 Model Gateway 配置，不保存 API Key。
