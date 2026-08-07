# EvalRAG Architecture

## 架构目标

EvalRAG 是一个围绕 RAG 运行与质量评估设计的轻量 Agent Harness。P0 使用原生 Python 和少量必要依赖，重点不是框架数量，而是：

- 模块契约清楚。
- 运行过程可追踪。
- 失败可以归因。
- 实验能够复现。
- 修改能够回归验证。

新人第一次理解项目时先读 `docs/project_map_zh.md`。本文件用于补充详细职责、
约束和设计取舍，不要求按文件顺序一次读完。

## 系统边界

系统分为四条链路：

```text
1. Data Pipeline
   raw documents -> normalize -> chunk -> index

2. Online RAG Pipeline
   query -> route -> retrieve -> evidence gate
         -> context -> generate -> validate -> answer / retry / abstain
         -> trace

3. Evaluation Harness
   frozen labels + run config -> execute pipeline -> case results
                              -> metrics -> report -> failures

4. Serving
   FastAPI -> request validation -> pipeline -> response / error
```

求职资料只属于 Data 与 Evaluation 的一个 profile，核心 pipeline 不应直接硬编码“简历回答模板”。

## 当前状态

当前已实现：

```text
raw files -> Document -> Chunk
query -> rule router -> keyword retrieval -> context -> structured generation
      -> citation validation -> RagResponse
routing/retrieval/context/generation/validation -> AgentTrace JSONL
versioned reviewed labels -> real router/retriever predictions
                          -> Recall@3 / Recall@5 / MRR / Router Accuracy
                          -> standard run artifacts
```

P0 目标：

```text
raw files
  -> normalized chunks
  -> keyword index / dense index

RagRequest
  -> Router
  -> Configurable Retriever
  -> Context Builder
  -> Structured LLM Generator
  -> Citation Validator
  -> Evidence Checker
       -> GENERATE
       -> BROADEN_SOURCE_AND_RETRY_ONCE
       -> ABSTAIN
  -> RagResponse
  -> AgentTrace

EvaluationDataset + RunConfig
  -> EvaluationRunner
  -> Metrics + CaseResults + FailureTaxonomy
  -> Versioned Report
  -> Executable Regression
```

## 离线数据链路

查询时不重新切分全部文档。数据新增或变化后执行离线处理：

1. importer 将 manual、JSON 或 CSV 数据转成 `Document`。
2. chunking 生成稳定 `Chunk.id` 并继承 metadata。
3. exporter 保存版本化 chunks。
4. keyword retriever 直接消费 chunks。
5. dense index builder 预计算 embedding 并保存索引。

### 当前 source types

- `jd`
- `resume`
- `interview`
- `project_logs`
- `user_profile`

### 岗位时效性 metadata

```text
source_type
job_id
company
job_title
city
source_platform
source_url
first_seen_at
last_seen_at
status: active | expired | unknown
version
source_priority
content_hash
```

Chunk 必须完整继承 Document metadata。`expired`、`last_seen_at` 和 `source_priority` 后续可用于 freshness filter，但 P0 不实现爬虫或复杂数据库。

## 在线 Pipeline

### RagRequest / RagResponse

目标请求：

```text
RagRequest
  request_id: str
  query: str
  top_k: int
  retriever: keyword | dense | hybrid
```

目标响应：

```text
RagResponse
  request_id: str
  trace_id: str
  answer: str
  citations: list[Citation]
  routed_sources: list[str]
  status: answered | insufficient_evidence | error
  latency_ms: float
  error_type: str | None
```

先检查能否复用现有 `AnswerResult`、`Citation` 和 `AgentTrace`，不平行定义重复结构。

### Source Router

职责：

- 根据 query 预测 intent。
- 返回优先 source types。
- 保留 matched rules 或置信信息，供 Trace 与评测使用。

当前实现是规则 Router。P0 先保留为可解释 baseline，不急于换 LLM Router。

Router 失败不通过“全库搜索后看答案不错”来掩盖，必须单独计算 Router Accuracy。

### Retriever

统一目标接口：

```python
class Retriever(Protocol):
    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        ...
```

三种 P0 配置：

1. `keyword`：现有关键词重叠 baseline。
2. `dense`：P0-D3 正式配置使用 512 维 `BAAI/bge-small-zh-v1.5` 中文
   embedding，并固定 Hugging Face commit；字符 TF-IDF + LSA 保留为无模型下载
   fallback。
3. `hybrid`：Keyword 与 Dense 使用 RRF 融合。

RRF：

```text
rrf_score(document) = sum(1 / (rrf_k + rank_i))
```

Hybrid 结果应保留：

- keyword rank/score。
- dense rank/score。
- fused score。
- retrieval reason。

不得删除 Keyword 实现，它是实验对照组。

### Reranker

`RerankRetriever` 先让基础 Retriever 召回 top-N，再通过统一 `RerankScorer`
接口对 Query-Chunk pair 重新打分。它只能改变候选顺序，不能找回召回阶段完全
遗漏的 Chunk。自动化测试使用 `FakeRerankScorer`；真实 adapter 为固定 model 与
revision 的 `CrossEncoderRerankScorer`。

P0-D5 因运行环境未完成 CrossEncoder 权重下载，正式 dev 对照使用确定性的中文
token overlap scorer。该 candidate 的 Recall@3/5、MRR 均退化且 P95 增加，因此
最终配置关闭 Reranker。这个负向结果保留在报告中，不能声称神经 Reranker 带来提升。

P0-D3 的索引保存在 `data/processed/indexes/<dataset_version>/`，记录 dataset、
embedding name/version、dimensions 和 chunk count。查询阶段只编码 Query，不能
重新拟合语料或重复编码全库。Retriever 由配置工厂注入 Pipeline 和 Evaluation，
两者不针对具体策略编写分支。

### Context Builder

P0-D1-T1 已实现 `ContextItem`、`BuiltContext` 和 `build_context()`。

输入：

- query。
- ranked retrieval results。
- context budget。

输出：

- 带 chunk id、source type、title 和 text 的上下文。
- used chunk ids。
- skipped chunk ids。
- truncation reason。

规则：

- 保持 retrieval rank。
- 不生成新事实。
- 不丢失 chunk id。
- 预算不足时只在 chunk 边界截断，P0 不截断单个 chunk 中间。
- 使用严格 rank 前缀；某个 chunk 放不下后，不跳过它去填充更低排名结果。
- Context Builder 的选择必须写入 Trace。

### Structured LLM Generator

模型输入：

- query。
- context。
- prompt version。

模型目标输出：

```json
{
  "answer": "回答内容",
  "cited_chunk_ids": ["chunk_001"],
  "sufficient": true,
  "reason": ""
}
```

约束：

- 只能根据上下文回答。
- 关键结论必须引用 chunk id。
- 证据不足时返回 `sufficient=false`。
- 不允许引用上下文中不存在的 id。

Generator 只负责模型调用与结构解析，不负责判断引用是否真实存在。

自动化测试使用 Fake LLM；真实模型仅用于显式 smoke test 和离线评测 run。
当前真实 adapter 包括基于标准库的 `OpenAIResponsesClient`，以及基于 OpenAI
兼容 Chat Completions 的 `DeepSeekChatClient`。DeepSeek adapter 开启 JSON Mode、
关闭思考模式，并记录 input/output/cache token；API key 只读取
`DEEPSEEK_API_KEY`，不进入配置、Trace 或报告。

### Citation Validator

LLM 输出是不可信外部输入。第一版执行确定性校验：

- JSON 和字段是否合法。
- cited id 是否存在于本轮 context。
- cited id 是否重复。
- `sufficient=true` 时 citations 是否为空。
- `sufficient=false` 时是否仍返回 citations。

Citation Validity 只说明引用 id 合法，不等于引用语义支持结论。语义支持由人工标注或独立 grader 评估。

### P0-D4 门控与有限重试 Pipeline

当前 `RagPipeline.run()` 可按配置选择 Rule、Semantic 或 Hybrid Router，并
串联 Retriever、Evidence Gate、Context Builder、Structured Generator 与
Citation Validator。一次请求只写一条 JSONL Trace，内部重试保存在
`attempts`，最终返回 `answered`、`insufficient_evidence` 或 `error`。

### Evidence Checker

状态：

```text
sufficient
retryable
insufficient
```

第一版可解释规则：

```text
unknown route
  -> insufficient（正常不可回答，不调用模型）

retrieval results empty
  -> retryable；重试耗尽后 retrieval_miss

top score / rank evidence below calibrated threshold
  -> retryable

intent/source policy requires multiple sources but one required source has no evidence
  -> retryable

valid evidence available
  -> sufficient
```

阈值只允许在 dev 集校准，冻结 test 集不得用于调参。

### 有限状态与重试

```text
ROUTE
  -> RETRIEVE
  -> CHECK_EVIDENCE
       -> sufficient: GENERATE
       -> retryable and retry_count == 0:
            BROADEN_SOURCES -> RETRIEVE
       -> insufficient:
            ABSTAIN
  -> VALIDATE_GENERATION
       -> format error and retry_count == 0:
            REGENERATE
       -> invalid citation:
            CONTROLLED_ERROR
       -> valid:
            ANSWER
```

约束：

- source 扩展最多一次。
- 格式修复最多一次。
- 每次重试记录 reason、latency 和 token usage；客户端不返回 usage 时明确记为
  `not_reported_by_client`，不使用字符数伪装 token。
- 不允许无限循环。

### Router V2

- Rule Router 保留关键词命中理由，适合精确且低延迟的显式表达。
- Semantic Router 使用固定 revision 的中文 embedding 比较 Query 与意图原型；
  模型和阈值只在 dev 选择。
- Hybrid Router 在 Rule 与 Semantic 一致时直接采用结果，只允许高分且有足够
  margin 的语义结果覆盖弱规则，并在 `RouteDecision.details` 保存两路判断。
- Pipeline 只依赖统一 `Router` Protocol，不包含实验策略分支。

## Trace 设计

一次 Query 对应一条完整 Trace，重试作为同一 Trace 内的 attempt 记录。

目标字段：

```text
trace_id
request_id
query
dataset_case_id
run_id
git_commit
intent
routed_sources
retrieval_config
retrieval_attempts
context_chunk_ids
prompt_version
model_config
generation_attempts
citations
evidence_decision
response_status
latency_by_stage
token_usage
estimated_cost
error_type
error_message
created_at
```

Trace 的目的不是记录越多越好，而是能回答：

- 失败发生在哪个阶段？
- 使用了什么配置？
- 为什么重试或拒答？
- 该请求属于哪个评测 run？
- 修改后同一 case 是否变好？

## Evaluation Harness

### EvaluationDataset

每条 case 至少包含：

```text
case_id
query
category
split: dev | test
expected_intent
expected_sources
relevant_chunk_ids
answerable
expected_points
```

### RunConfig

```text
run_id
dataset_version
git_commit
retriever_name
top_k
embedding_model
rrf_k
llm_model
temperature
prompt_version
context_budget
evidence_thresholds
```

### EvaluationRun

一次运行产出：

```text
summary.json
case_results.jsonl
failures.jsonl
run_config.json
latency.json
```

必须从真实 Router、Retriever 和 Pipeline 生成 predictions，不再从 fixture 读取手写 predicted values作为正式报告。

### 指标分层

Routing：

- Router Accuracy。

Retrieval：

- Recall@3。
- Recall@5。
- MRR。
- P50/P95 retrieval latency。

Grounding 与 Safety：

- Citation Validity。
- Key-Point Coverage。
- Abstention Accuracy。
- Unsupported Answer Rate。

P0-D5 的 Key-Point Coverage 使用规范化子串匹配，只作为 lexical baseline；它会
漏判同义表达。P0-D6 已通过可注入 Grader Protocol 增加语义要点评分，并保留
逐 point 的 verdict、reason 和 evidence span。Unsupported Answer 不由指标汇总
函数自动推断，而应先把回答拆成 factual claims，再逐条对照 cited Context 保存
`supported | unsupported | unknown`、证据位置和理由；grader 失败或证据模糊时必须
记为 unknown/unavailable，不能默认 supported。

P0-D6 只读取 P0-D5 保存的 Case Results 和 Trace：

```text
expected_points + answer
  -> KeyPointGrader
  -> PointJudgment[]

answer + cited chunk text
  -> GroundingGrader
  -> ClaimJudgment[]
  -> unsupported_answer: true | false | unknown
```

自动化测试注入 Fake grader；正式离线审核使用 DeepSeek JSON Mode。Prompt、模型、
token、成本、延迟和逐调用错误均落盘。Judge 与 Generator 使用同一模型家族，因此
报告必须声明自评偏差，不能把模型 verdict 冒充独立人工 ground truth。

End-to-End：

- Success Rate。
- P50/P95 total latency。
- token usage。
- estimated cost。

Regression：

- Regression Pass Rate。

固定公式、分割方式与运行规则见 `docs/evaluation_protocol.md`。

## Failure Taxonomy

统一错误类型：

- `router_wrong`
- `retrieval_miss`
- `context_insufficient`
- `context_truncated`
- `llm_timeout`
- `llm_format_error`
- `citation_invalid`
- `unsupported_answer`
- `should_abstain`
- `unexpected_abstention`
- `service_error`
- `unknown_error`

`insufficient_evidence` 是合法业务状态；只有预期可回答却因系统失败没有回答时，才记为 failure。

## Regression

Regression Case 至少包含：

```text
case_id
query
failure_type
expected_behavior
original_run_id
original_trace_id
fixed_in_commit
added_at
```

状态：

- `open`：已确认但尚未修复，不计入 pass rate 分母。
- `fixed`：已经修复，必须进入自动化回归。

闭环：

```text
evaluation failure
  -> inspect trace
  -> classify
  -> reproduce on dev
  -> implement one scoped change
  -> rerun dev
  -> run frozen test
  -> add fixed regression case
```

P0-D5 已实现版本化 `RegressionCase` loader/runner：`fixed` case 必须被自动化断言，
`open` case 只作为待修复清单，不进入 pass rate。当前 fixed 1/1 通过、open 3 条。

## P0-D5 Frozen Configuration

最终配置及输入文件 hash 固定在 `configs/final/p0_v0.2.json`。Frozen test 只执行
声明的四种检索配置一次，最终选择 Hybrid 并关闭 Reranker。端到端工件使用
deterministic extractive generator，因此可验证 citation id、拒答和 Harness 数据流，
但不能替代真实 LLM 的答案支持性、token 或成本评测。

随后在相同冻结 Router/Retriever/Evidence 配置上运行 `deepseek-v4-flash`：真实
Pipeline 直接复用 `RagPipeline`，每条 Case 生成 `RagResponse` 和请求级 Trace，
汇总 token、价格快照成本和阶段延迟。模型返回的 `sufficient` 与 citations 不被
直接信任，仍由确定性 Validator 决定最终状态。

## Serving

P0 在质量闭环之后增加最小 FastAPI：

```text
POST /v1/query
GET  /health
```

服务层负责：

- 请求校验。
- request/trace id。
- timeout。
- 统一错误响应。
- 调用 pipeline。

服务层不负责：

- 重新实现 Router/Retriever。
- 在 Web 请求中运行完整批量评测。
- 用内存变量伪装持久化队列。

批量评测首先使用 CLI。只有出现真实异步需求和可测量收益时，P1 才增加 Redis Queue 与 worker。

## 运行工件目录

```text
configs/
  retrieval/
  generation/
  evaluation/

data/
  raw/
  processed/chunks/
  indexes/
  evaluation/

reports/
  runs/<run_id>/
  comparisons/
  failure_cases/

traces/
  local/
  sanitized_examples/
```

目录按任务按需创建。运行工件必须能够关联 `run_id`、dataset version 和 Git commit。

## 配置、安全与隐私

- API key 只从环境变量读取。
- 不在源码、配置、Trace、报告和对话中打印密钥。
- 真实简历、手机号、邮箱和账号必须脱敏。
- 外部文档记录来源、采集日期和可公开状态。
- 自动化测试不访问网络或付费 API。
- 真实模型 run 显式记录模型、参数、token 和成本。

## P0 之后

只有 P0 指标和失败闭环完成后，才考虑：

- reranker。
- LLM Router。
- Redis Evaluation Worker。
- Trace 可视化。
- LangGraph 状态机重构。
- 更通用的数据 profile。

不做多 Agent、MCP 或 Kubernetes 作为 P0 装饰。
