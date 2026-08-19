# EvalRAG 三条运行链路

本文只解释“数据怎样流动”。模块为什么存在、算法和指标统一查阅
[项目地图](project_map_zh.md)，避免在多个文档重复维护。

## 一、在线回答链路

```text
RagRequest
-> AgentRuntime
-> Router
-> Adaptive Retriever
-> Evidence Gate
-> Context Engine
-> Generator / Model Gateway
-> Citation Validator
-> RagResponse
```

### 1. 请求与运行时

用户的 `query`、会话信息和配置组成 `RagRequest`。AgentRuntime 为这次请求创建唯一
`run_id/trace_id`，后续路由、检索、重试和模型调用都作为它的子 Span，不会每个阶段写一条
互不关联的 Trace。

### 2. Router

Router 接收 Query，输出 `RouteDecision(intent, routed_sources, confidence, reason)`。
规则方法擅长精确术语，语义方法负责同义表达，Hybrid 根据置信度融合；已确认反馈只在离线更新
prototype 并通过 shadow/dev gate 后发布，单次在线请求不会直接修改 Router。

### 3. Adaptive Retriever

Query Analyzer 判断问题更依赖精确词、语义还是关系路径。Retriever 从 BM25、Dense、RRF 和
Graph + Vector 中选择策略，统一输出按 rank 排序的 `list[RetrievalResult]`。复杂且低置信的
Query 才触发 CrossEncoder，对已经召回的候选重排，不能找回第一阶段完全漏掉的 Chunk。

### 4. Evidence Gate

Gate 检查：

- Route 是否为 `unknown`；
- 检索结果数量和最高分是否达到配置阈值；
- 结果是否覆盖路由要求的 source。

输出有三种：

- `sufficient`：证据足够，进入 Context Engine。
- `retryable`：当前有一定证据，但可能被 source filter 限制；去掉过滤做一次全库检索。
- `insufficient`：空 Query、unknown route、无有效结果或重试后仍不足，明确拒答。

扩源不是重复原检索，而是放宽 Router 给 Retriever 的来源范围；最多一次，避免无限循环。

### 5. Context Engine

Context Engine 把 system instruction、当前 Query、确认的 Profile、最近历史、长期语义 Memory
和检索 Evidence 放入统一 token budget。它先去重、按优先级选择，再保留完整证据；预算不足时
记录 skipped IDs 和压缩/裁剪原因，不在一个 Chunk 中间粗暴截断。

### 6. Generator 与 Gateway

Generator 把 Context 包装成结构化 Prompt，要求模型只依据证据输出
`answer/cited_chunk_ids/sufficient/reason`。Model Gateway 处理 timeout、瞬时错误有界退避、
并发上限、熔断和 Provider fallback。模型 JSON 非法时会用修复 Prompt 再生成一次；仍失败则
返回受控 `llm_format_error`。

### 7. Citation Validator 与响应

Validator 检查 cited chunk ID 是否来自本轮 Context、是否重复，以及回答状态与 citations 是否
一致。合法结果转换为 `RagResponse`；非法引用不会被静默删除后继续回答，而是记录错误并进入
Trace。这里验证“引用是否合法”，事实是否真的被证据支持由离线 Claim-Level Grounding 审核。

## 二、离线评测链路

```text
Corpus + EvaluationCase + RunConfig
-> 系统实际运行
-> Prediction + Trace
-> Metrics / Semantic Audit
-> Failures / Report
-> Regression / CI Gate
```

1. `EvaluationCase` 保存 Query、expected intent/sources/relevant chunk IDs、answerable 和
   expected points；这些是评测标签，不会传给在线 Pipeline 偷看。
2. `RunConfig` 固定 dataset、split、Router/Retriever/模型版本和阈值。
3. Runner 真实调用系统得到 prediction 和 Trace，不手写 predicted 字段。
4. 检索阶段计算 Router Accuracy、Recall@k、MRR、NDCG 和延迟；回答阶段计算引用、拒答、
   要点覆盖和事实支持性。
5. `case_results` 保存逐题结果，`failures` 只列不满足协议的题，`summary` 是机器可读汇总，
   `report` 是带配置、表格和失败分析的人类可读报告。
6. 确认的失败先进入 open regression；修复并验证后转为 fixed regression。CI 每次自动运行
   fixed case 和 reference 指标，防止旧问题复发。
7. dev 可反复比较；frozen test 只在配置固定后运行。指标代码出错时可重算保存的 prediction，
   但不能反复重新调用随机模型并挑最好的一次。

## 三、服务与异步评测链路

```text
在线 Query:
Client -> FastAPI -> AgentRuntime -> PostgreSQL -> RagResponse + trace_id

批量评测:
Client -> FastAPI -> PostgreSQL queued Job -> Redis Queue
       -> Evaluation Worker -> report files -> PostgreSQL final status
```

- **FastAPI**：把 Query、Trace 查询和 Evaluation Job 变成稳定 HTTP 接口。
- **PostgreSQL**：保存请求、Trace、任务状态、Run 摘要和报告路径，是状态真相来源。
- **Redis**：队列只存待执行 job ID，并缓存最近会话；不保存最终任务结果。
- **Worker**：独立消费队列，执行耗时评测，把状态从 `queued` 推进到
  `running -> succeeded/failed`。
- **Docker Compose**：统一启动 API、Worker、PostgreSQL 和 Redis，并配置健康检查和持久化卷。

同一个 idempotency key 重复提交评测时，PostgreSQL 返回原 Job，不会重复入队。Worker 的受控
重试只处理配置允许的瞬时失败，有次数上限；业务错误、鉴权错误或重试耗尽会落为 `failed`，
同时保留 error type 供排查。
