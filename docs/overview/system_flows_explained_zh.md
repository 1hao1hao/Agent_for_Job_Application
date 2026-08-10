# EvalRAG 三条链路快速复习

> 本文只帮助快速恢复项目主线，不要求按文件顺序通读代码。阅读时始终问三个问题：
> 上一步传来什么、本步解决什么、下一步拿到什么。

## 先区分三条链路

```text
在线回答链路：一个 Query 如何得到有证据的回答或明确拒答
离线评测链路：一批固定问题如何衡量系统质量并定位失败
P1 服务链路：外部如何通过 HTTP 调用回答，并异步提交批量评测
```

在线回答链负责“运行系统”，离线评测链负责“检查系统”，P1 服务链负责“让外部稳定地
调用这两种能力”。

## 一、在线回答链路

### 完整数据流

```text
用户 Query
  -> RagRequest
  -> Router
  -> RouteDecision(intent, routed_sources, confidence, reason)
  -> Retriever
  -> list[RetrievalResult](chunk, score, rank, reason)
  -> Evidence Gate
  -> EvidenceDecision(sufficient | retryable | insufficient)
  -> Context Builder
  -> BuiltContext(text, items, used/skipped chunk ids, budget)
  -> Generator
  -> GenerationResult(answer, cited_chunk_ids, sufficient, reason)
  -> Citation Validator
  -> ValidationResult(valid, citations, issues)
  -> RagResponse(answered | insufficient_evidence | error)

各阶段结果、attempt、耗时、token 和错误
  -> 一条请求级 AgentTrace
```

### 每一步为什么存在

| 阶段 | 接收什么 | 做什么 | 返回什么 |
|---|---|---|---|
| `RagRequest` | Query 和运行参数 | 封装一次请求，不统一切分 Query | 稳定请求契约 |
| Router | 原始 Query | 判断意图和需要搜索的 source types | `RouteDecision` |
| Retriever | Query、Chunks、source filter | 对候选 Chunk 打分并排序 | `list[RetrievalResult]` |
| Evidence Gate | RouteDecision、检索结果 | 检查证据是否值得进入生成 | `EvidenceDecision` |
| Context Builder | 排序结果、字符预算 | 按 rank 拼接完整证据 | `BuiltContext` |
| Generator | Query、BuiltContext | 构造 Prompt，调用 LLM 并解析 JSON | `GenerationResult` |
| Validator | GenerationResult、BuiltContext | 校验 citation id 和字段组合 | `ValidationResult` |
| Pipeline | 前述依赖与配置 | 串联所有阶段并形成最终状态 | `RagResponse` + Trace |

### Router 和 Retriever 不要混淆

- Router 选择“去哪些来源找”，不返回最终证据。
- Retriever 在这些来源中选择“哪些 Chunk 最相关”。
- Router Hybrid 不使用 RRF：只有 Rule 较弱且 Semantic 足够明确时，Semantic 才覆盖
  Rule；其他情况保留 Rule。
- Retriever Hybrid 才使用 RRF，把词面检索和 Dense 检索的**排名**融合。

Query 也不是先统一切分一次：Rule 检查关键词，Keyword/BM25 做 tokenization，Semantic/
Dense 则编码完整 Query。

### Evidence Gate 与两类重试

Evidence Gate 检查四件事：Router 是否可回答、结果数量、最高检索分数、所需来源是否
全部覆盖。

```text
sufficient  -> 证据达到门槛，进入 Context Builder
retryable   -> 证据弱且尚未重试，去掉 source filter 后全库检索一次
insufficient -> 无法路由，或扩源后仍不足，直接拒答
```

Pipeline 还有一次独立的格式修复：LLM 返回非法 JSON、缺少字段或字段类型错误时，使用
相同 Prompt 再生成一次。当前没有本地猜测修复；第二次仍失败就返回
`llm_format_error`。Citation 不合法也不会继续重试，而是返回受控错误。

链路中有两个 `sufficient`：Evidence Gate 的状态是系统规则判断“可以尝试生成”；
`GenerationResult.sufficient` 是模型对“Context 是否足以回答”的声明，仍要经过 Validator。

## 二、离线评测链路

离线评测由“准备固定证据”和“运行固定题目”两部分组成。

### 1. Corpus 准备

```text
五类 raw 文档
  -> Ingestion：Document -> Chunk
  -> Corpus Builder
  -> corpus manifest + versioned chunks + stats
```

#### Manifest 是什么

Manifest 是**语料清单**，不是按运行批次分类的结果。它一行记录一份原始文档：

```text
document_id、source_type、source_path、采集日期、公开状态、是否脱敏、
content_hash、内容来源、审核状态
```

它回答“这次语料由哪些文档组成、来自哪里、有没有重复、能否公开”。实验批次由后面的
`run_id` 和 `RunConfig` 区分。

#### 为什么 Chunk 要版本化

切分函数没变，也仍然需要版本化，因为原始文档、metadata、`max_chars` 或 Chunk ID 都
可能改变。评测标签直接引用 `relevant_chunk_ids`；如果每次临时重新切分，旧标签和旧
报告可能再也找不到原来的证据。

```text
evalrag_v0.2.jsonl = 当时那批原文 + metadata + 切分配置产生的固定 Chunk 快照
```

版本化的目的不是证明切分算法经常变化，而是保证旧实验可以复现和比较。

#### Stats 是什么

Stats 是语料质量概览，包括文档数、各 source 数量、Chunk 数、长度 min/mean/P50/P95、
空文档、重复 content hash 和已审核文档数。它用于在运行评测前发现“空文件、重复凑数、
Chunk 过长或某类来源缺失”等数据问题，不参与在线回答。

### 2. EvaluationCase 与 RunConfig

`EvaluationCase` 是题目和人工期望，不含系统预测：

```text
query、category、split、expected_intent、expected_sources、
relevant_chunk_ids、answerable、expected_points
```

`RunConfig` 固定本次实验使用的 dataset version、`dev/test`、Router/Retriever strategy、
top-k、具体阈值和配置版本。这里的 version 是数据、Prompt 或策略配置的版本，不是下面
某一个算法自动生成的版本号。

`dev` 可以理解为**开发调参集**，但不是模型训练集：项目不会用它做梯度训练，而是反复
比较策略、阈值和失败案例。`test` 是最终检查集，在配置确定前冻结，不能看完结果后继续
针对它调参。

### 3. Runner 实际执行

```text
EvaluationCase + versioned Chunks + RunConfig
  -> Evaluation Runner
  -> 对每条 Query 实际调用 Router / Retriever / Pipeline
  -> system predictions + Case Results + AgentTrace
```

`predicted` 字段必须来自系统运行，不能手写。不同评测层计算不同内容：

其中，检索消融 Runner 只执行 Router 和 Retriever，保存逐 Case 预测与阶段延迟；端到端
Runner 才执行完整 Pipeline 并保存 `AgentTrace`。它们共享同一份 Case 标签，但回答指标
只由端到端 Run 计算。

| 评测层 | 指标 | 真正衡量什么 |
|---|---|---|
| Router | Router Accuracy | intent 和 routed sources 是否都与标签一致 |
| Retrieval | Recall@3/5 | 前 3/5 名覆盖了多少 relevant Chunk |
| Retrieval | MRR | 第一条 relevant Chunk 是否排得靠前 |
| Grounding | Citation Validity | 模型返回的 citation id 是否存在于本轮 Context；不等于引用支持答案 |
| Answer | Key-Point Coverage | 回答覆盖了多少 expected points |
| Refusal | Abstention Accuracy | 不可回答 Case 中，系统正确返回拒答的比例 |
| Grounding | Unsupported Answer Rate | 已审核 answered Case 中，存在无证据事实主张的比例 |
| Overall | End-to-End Success | 路由、召回、引用、要点、事实支持或正确拒答是否共同满足 |
| Efficiency | P50/P95 latency、tokens、cost | 典型耗时、尾部耗时和模型调用代价 |

Abstention Accuracy 不是“检索失败占比”：

```text
分子 = 标注为不可回答，并且系统返回 insufficient_evidence 的 Case
分母 = 所有标注为不可回答的 Case
```

Semantic Audit 会复用已经保存的答案，不重新调用 Generator：LLM Grader 判断同义表达
是否覆盖 expected point，并把回答拆成 factual claims，逐条检查 cited Context 是否支持。

### 4. 输出工件有什么区别

| 工件 | 内容 | 用途 |
|---|---|---|
| `case_results.jsonl` | 每条 Case 的标签、预测和指标 | 查看全部细节，重新计算指标 |
| `traces.jsonl` | 每条请求各阶段真实过程 | 定位失败发生在哪一步 |
| `failures.jsonl` | 未满足当前评测标准的 Case 子集 | 集中排查问题，不代表评测程序崩溃 |
| `summary.json` | 本次 Run 的聚合指标、数量和延迟 | 程序读取、跨 Run 对比 |
| Markdown report | 指标表、差异 Case、原因和结论 | 供项目使用者阅读 |

`failures` 通常表示**系统预测不符合标签或质量门槛**，例如路由错误、Recall@5 不完整、
可回答题被拒答、要点覆盖不足或 unsupported claim。评测程序本身异常属于 run error / grader
unavailable，不应伪装成普通质量失败。

`summary` 是单次 Run 的机器可读汇总；`report` 是在一个或多个 summary、case 和 failure
基础上写成的人类可读分析，所以两者不是重复文件。

### 5. Confirmed Failure 与 Regression

不是每个低分 Case 都立即成为 Regression Case。先根据 Case Result 和 Trace 确认：

```text
评测失败
  -> Trace 定位 root cause
  -> 明确“系统应该怎样表现”
  -> confirmed failure
  -> 修复并在完整 dev 上检查副作用
  -> fixed RegressionCase
```

- `open`：问题已确认但还没修复，只记录，不计入通过率。
- `fixed`：问题已经修复，包含可以执行的输入、期望字段和断言。
- Automated Regression：以后每次修改 Router/Retriever/Pipeline 后，测试程序自动执行所有
  fixed cases；一旦旧问题复现，测试立即失败并显示 case id 和 failure type。

因此，Evaluation 负责发现和衡量问题；Regression 负责守住已经确认的修复成果。

## 三、P1 服务与异步评测链路

P1 没有改变核心 RAG 算法，而是在它外面增加稳定 HTTP 契约、任务状态和可复现运行环境。

### 1. 在线 Query 服务

```text
POST /v1/query
  -> FastAPI / Pydantic 校验 RagRequestBody
  -> 转成领域 RagRequest
  -> PipelineQueryService.execute()
  -> RagPipeline.run()
  -> RagResponse + AgentTrace
  -> PostgreSQL 保存请求索引和 Trace
  -> HTTP RagResponseBody

GET /v1/traces/{trace_id}
  -> PostgreSQL
  -> 完整 AgentTrace
```

FastAPI 只负责参数校验、timeout 和 HTTP 错误码映射，不重新实现 Router 或 Pipeline。
正常回答、可靠拒答和系统错误继续由 `RagResponse.status` 表达。

### 2. 为什么 Evaluation 要异步执行

一次完整评测可能运行几十到上百条 Query，耗时远长于一次在线回答。如果在 HTTP 请求中
同步执行，连接会长期占用，也无法可靠查询进度或在进程重启后恢复。因此创建接口只登记
任务并快速返回 `job_id`。

### 3. 异步 Evaluation Job 状态流

```text
POST /v1/evaluation-jobs
  -> 校验 EvaluationJobRequest(dataset_version, split, run_config)
  -> PostgreSQL create_job(status=queued)
  -> Redis 只入队 job_id
  -> HTTP 202 + EvaluationJob

EvaluationWorker.run_once()
  -> Redis dequeue(job_id)
  -> PostgreSQL 原子 queued -> running
  -> Evaluation Executor 调用现有 Runner
  -> 文件系统保存完整 Run Artifacts
  -> PostgreSQL 保存 Run 摘要和 report_path
  -> running -> succeeded

受控异常或 timeout
  -> running -> failed(error_type, error_message)
  -> 显式 retry 且未超过预算时 -> queued
```

### 4. PostgreSQL、Redis 和文件系统为什么各做一件事

| 组件 | 保存什么 | 为什么这样设计 |
|---|---|---|
| PostgreSQL | Job 状态、attempt、请求/Trace、Run config/summary、report path | 是持久化真相来源，API/Worker 重启后仍可查询 |
| Redis | 等待 Worker 消费的 job id | 负责快速队列协调，不承担最终状态 |
| 文件系统 volume | case results、failures、traces 和完整报告 | 大体积工件不塞进任务状态表 |
| Docker Compose | API、Worker、PostgreSQL、Redis 和 volume | 一条命令复现真实协作环境 |

`Idempotency-Key` 防止客户端重复提交同一评测而创建多个 Job。Worker 启动时还会把中断
遗留的 `running` Job 恢复为 `queued`；重试次数仍由 PostgreSQL 中的预算限制，不会无限
循环。

## 四、三条链路如何连起来

```text
在线 Pipeline 产生 RagResponse + Trace
                 |
                 v
离线 Runner 用固定 EvaluationCase 批量调用 Pipeline，并生成指标与 failures
                 |
                 v
confirmed failure 固化为 RegressionCase，约束下一次代码修改

FastAPI 暴露单次 Pipeline；Redis Worker 异步触发离线 Runner；
PostgreSQL 保存可查询状态；文件系统保存可复查工件。
```

三条链路可以概括为：

> EvalRAG 的在线链路负责有证据地回答或拒答，离线 Harness 用固定标签、预测、Trace 和
> 指标定位质量问题，Regression 保证已修复问题不复现；P1 再通过 FastAPI、PostgreSQL、
> Redis Worker 和 Docker Compose 把单次问答与批量评测变成可调用、可恢复的服务。
