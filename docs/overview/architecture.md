# EvalRAG 详细架构

本文件描述当前 P1 的模块契约、职责边界和失败状态。第一次阅读请先看
[项目地图](project_map_zh.md)；需要看整体连接时查 [架构图](architecture_diagram.md)；
数字统一以 [最终实验报告](../evaluation/final_experiment_report.md) 为准。

## 架构目标

EvalRAG 是围绕中文多源知识推理构建的 RAG Agent Harness。它不把“能调用 LLM”当作完成，
而要求每次请求可追踪、每次改动可评测、已确认失败可回归、证据不足时可拒答。

系统分为四条链路：

```text
Knowledge Build: raw/public data -> Document -> Chunk -> indexes / graph
Online Runtime:  request -> route -> adaptive retrieve -> gate -> context -> generate -> validate
Evaluation:      cases + config -> predictions -> metrics -> failures -> regression / CI gate
Serving:         FastAPI -> Runtime / Worker -> PostgreSQL / Redis / pgvector / Neo4j / reports
```

求职资料是当前业务 profile，核心 Retriever、Runtime、Trace 和 Evaluation 不硬编码求职回答模板。

## 1. 知识与索引

### 统一证据契约

`Document` 保存原文与统一 metadata；`Chunk` 是检索、图关系、Context 和 Citation 共同消费的
最小证据单元。Chunk 完整继承 source、路径、时效、owner scope 和 provenance，并增加稳定 ID、
序号和切分信息。

```text
Importer -> Document -> boundary-aware chunking -> Chunk
```

稳定 `chunk_id` 由来源路径、source type、文本内容和 chunk 序号等稳定输入生成；相同快照重复构建
得到相同 ID，使 `relevant_chunk_ids`、Citation 和 Regression 不会随运行随机漂移。

### Corpus v0.3

知识构建保存三类工件：

- `manifest`：原始材料的来源、revision、许可、采集时间、公开/脱敏和审核状态。
- `versioned chunks`：固定版本的证据快照，供索引与评测共同引用。
- `stats`：文档/Chunk 数、长度分布、空文档、重复 hash、近重复组和来源占比。

SHA-256 用于完全重复检测；SimHash 用于模板化近重复候选；provenance 用于回答“材料从哪里来、
何时获取、能否公开、是否脱敏”。它们属于离线 Corpus 构建，不参与在线 Query 检索打分。

### 索引与图

```text
Versioned Chunks
  -> BM25 document statistics
  -> BGE Dense vectors -> file exact / pgvector exact or HNSW
  -> deterministic entity/relation extraction -> Graph JSON / Neo4j
```

图包含 Job、Skill、Project、Experience、Technology、Company 等节点，以及 `requires`、
`demonstrates`、`uses`、`belongs_to`、`related_to` 等有向关系。节点和边必须保留支撑它们的
`chunk_ids`；图只用于寻路，最终进入 Prompt 和 Citation 的仍是原始 Chunk。

## 2. 在线 Runtime

### 请求与响应

`RagRequest` 包含 query、请求配置以及可选 `user_id/session_id`；`RagResponse` 返回 answer、
合法 citations、状态、trace_id、路由来源、延迟和受控错误。服务、CLI 和 Evaluation Worker
复用同一 Runtime，不平行实现三套 Pipeline。

### Router

统一 `Router` 接口输出：

```text
RouteDecision(intent, routed_sources, confidence, reason, details)
```

- Rule：关键词命中，低延迟且可解释。
- Semantic：BGE Query 与版本化 intent prototypes 的相似度和 margin。
- Hybrid：一致时采用结果；只允许高分、高 margin 的 Semantic 覆盖弱 Rule。
- Feedback：只消费已确认、通过 shadow/dev gate 的短意图锚点；在线请求不直接学习。

Router 负责缩小知识来源，不负责决定 BM25/Dense/Graph；后者由 Query Analyzer 处理。

### Query Analyzer 与 Adaptive Retriever

```text
query + routed_sources
-> QueryFeatures
-> EvidenceRequirement
-> strategy selection
-> candidates
-> retrieval confidence
-> optional CrossEncoder
-> list[RetrievalResult] + RetrievalDecision
```

`QueryFeatures` 只保留四个核心证据信号：精确词面、语义、多来源、实体关系；实体类型和
`is_unanswerable_route` 是辅助上下文。`EvidenceRequirement` 描述需要 lexical、semantic、
graph 或 multi-source 能力，策略映射为：

| 证据需求 | Retriever |
|---|---|
| 精确事实 | BM25 |
| 语义解释/改写 | Dense |
| 多源综合或不确定 | BM25 + Dense RRF Hybrid |
| 实体关系推理 | Graph + Vector |
| Graph 不可用 | 受控降级到 Hybrid |

首次检索后，数量、首位 margin、required source coverage 和双路一致性组合成置信度。
`rerank_policy=low_confidence` 时只对低置信候选运行一次 CrossEncoder，并用加权 RRF 保留原排序；
它不能找回未进入候选集的 Chunk。策略、特征、证据需求、置信度、重排原因和模型版本进入 Trace。

### Graph + Vector

Graph Retriever 先做实体链接和关系识别，再执行有 hop、节点数和 timeout 上限的双向 BFS，收集
节点/边引用的 Chunk。Vector 分支为 Sparse + BGE Dense 的 RRF Hybrid；两路候选去重后再用
外层 RRF 融合，并保存 vector rank、graph rank、path、edge IDs 和 path validity。

Graph 没有链接到实体时返回空路，由 Vector 候选兜底。Graph-only 不是默认策略，因为它提供关系
路径但文本覆盖不足。

### Evidence Gate 与两类重试

Evidence Gate 检查 route、结果数量、最高分和 required-source coverage，输出：

```text
sufficient   -> 构建 Context 并生成
retryable    -> 去掉 source filter，全库检索一次
insufficient -> 明确拒答
```

扩源重试只解决“Router 过滤过窄但全库可能有证据”，最多一次。模型 JSON 格式错误由 Generator
使用修复 Prompt 再生成一次；这两类重试原因不同、计数独立，均不能无限循环。

### ContextPolicy 与 ContextEngine

```text
SessionMemoryService
  History: Redis -> miss/error -> PostgreSQL
  Profile/Summary: PostgreSQL
  Memory: query embedding -> pgvector user-scoped retrieval
-> ContextInputs
-> ContextSignalExtractor
-> ContextPolicy -> ContextPlan
-> ContextEngine -> ManagedContext
```

ContextSignalExtractor 计算 History Token Pressure、指代/省略与 BGE 语义连续性组成的 Follow-up
分数，以及长期 Memory 的最高相关度。ContextPolicy 只决定是否使用 Profile/Summary、Recent
History 条数和 Memory top-k；ContextEngine 在统一 token budget 下执行优先级、跨层去重、完整
Evidence 选择和裁剪。system 与当前 query 放不下时受控失败，不静默删除。

Profile 只允许显式确认写入；Memory 有 user scope、来源、重要性、版本、TTL 和 active 状态。
Trace 只保存 segment/memory ID、预算和选择原因，不复制敏感 Profile 原文。

### Generator、Gateway 与 Validator

Generator 要求模型只依据本轮 Context 返回：

```json
{
  "answer": "...",
  "cited_chunk_ids": ["chunk-id"],
  "sufficient": true,
  "reason": ""
}
```

Model Gateway 在统一 `LlmClient` 契约外处理 timeout、429/5xx 有界退避、并发 semaphore、熔断和
Provider fallback；鉴权错误不盲目重试。API key 只从环境变量读取，不进入配置、Trace 或报告。

Citation Validator 校验 cited ID 是否来自本轮 Context、是否重复，以及 `sufficient` 与 citations
组合是否合法。它只验证结构与证据引用范围，不判断自然语言事实是否真的被证据支持。

## 3. Run、Span 与恢复

AgentRuntime 为一次请求创建一个 root Run；routing、query analysis、retrieval、rerank、evidence、
context、generation、validation 和 retry 是子 Span。Trace 记录配置 fingerprint、attempt、ID 引用、
latency、token、错误和决策原因，不保存密钥、未脱敏正文或完整原始 Prompt。

- Checkpoint 保存已完成阶段、工件引用和 side-effect key。
- Resume 只恢复 fingerprint 一致且工件仍存在的同一 Run；漂移时拒绝旧状态并安全重跑。
- Replay 对保存的 request/config/artifact 执行 Fake 全链或阶段重放；不可确定复现的外部模型输出
  返回 unavailable，不伪装一致。

Trace 写入失败与业务结果隔离：能返回回答时不因观测后端异常而丢失业务响应，但会保留受控错误。

## 4. Evaluation Harness

`EvaluationCase` 保存 query、category、split、expected sources/intent、relevant Chunk IDs、answerable
和 expected points；这些标签不会传给在线 Pipeline。`RunConfig` 固定 dataset、split、组件版本、
阈值、模型、Prompt 和预算。

一次真实运行保存：

```text
run_config.json
case_results.jsonl
failures.jsonl
summary.json
report.md
traces / grader verdicts（按任务需要）
```

指标分层：

- Routing：Router Accuracy / Source Exact。
- Retrieval：Recall@3/5、MRR、NDCG@5、P50/P95。
- Reliability：Citation Validity、Abstention Accuracy、End-to-End Success。
- Answer Audit：Semantic Key-Point Coverage、Claim-Level Grounding。
- Context：Follow-up Success、Prompt Token、History Redundancy、Memory Recall。
- Runtime：total/stage latency、tokens、估算成本、failure type。

Lexical Key-Point scorer 保留为确定性 baseline；LLM grader 保存逐 point/claim 的 verdict、reason、
answer evidence、citation 和 cited evidence。`unknown/unavailable` 不能默认改为 covered/supported。

### Failure 与 Regression

`failures.jsonl` 保存预测不满足标签或可靠性协议的 Case，不只是脚本异常。确认失败后先进入
`open regression`；修复、完整 dev 验证且无隐藏退化后转为 `fixed regression`。CI 自动断言 fixed
Case，并比较 reference 指标阈值；open Case 不伪装成 pass，也不计入 fixed pass rate。

```text
failure -> inspect Trace -> classify root cause -> scoped fix
        -> rerun complete dev -> fixed RegressionCase -> CI Gate
```

dev 可反复比较；frozen test 只在配置固定后运行。指标代码出错时可以用同一批已保存 predictions
重算，但不能重新调用随机模型并挑选更好的 Run。

## 5. Serving 与持久化

FastAPI 暴露：

```text
GET  /health
POST /v1/query
GET  /v1/traces/{trace_id}
POST /v1/evaluation-jobs
GET  /v1/evaluation-jobs/{job_id}
```

在线 Query 同步调用 Runtime；批量 Evaluation 只快速创建 Job，不阻塞 HTTP：

```text
EvaluationJobRequest
-> PostgreSQL idempotent queued Job
-> Redis Queue(job_id)
-> Worker: queued -> running -> succeeded | failed
-> report volume + PostgreSQL summary/report_path
```

- PostgreSQL：请求、Run/Trace、Job、Profile/Summary/Memory 元数据的真相来源。
- Redis：短期 Job 队列和最近 Session History 缓存，不保存最终任务状态。
- pgvector：持久化 Dense Index 与 user-scoped Semantic Memory。
- Neo4j：版本化知识图、Chunk 引用与 provenance。
- 文件卷：完整 report、case results、failures 和大体积工件。

Docker Compose 编排 API、Worker、PostgreSQL/pgvector、Redis 和 Neo4j。GitHub Actions 分为服务链
集成、PR Evaluation Gate 和手动完整持久化消融；本地无 Docker 时不能用 adapter 单测冒充容器
重启恢复验证。

## 6. 当前证据边界

- `evalrag_v0.3` 是当前 Corpus/Benchmark；v0.2 只保留为 P0 历史实验。
- v0.3 标签为 corpus-grounded AI-assisted，不是线上分布或独立人工金标准。
- Graph+Vector 提升 frozen 检索指标，但 CPU P95 增加；Adaptive selector 和 On-demand Reranker
  尚未取得质量/延迟 Pareto 最优。
- Context 60 组/300 turns 是不调用 LLM 的确定性 dev benchmark，没有 untouched multi-turn test。
- Citation Validity 不等于事实支持度；检索 Recall/MRR 不等于最终答案准确率。
- v0.3 frozen E2E 暴露过度拒答，因此不能用局部检索提升声称端到端质量已经提升。

详细数字、run ID 和失败 Case 见 [最终实验报告](../evaluation/final_experiment_report.md)。
