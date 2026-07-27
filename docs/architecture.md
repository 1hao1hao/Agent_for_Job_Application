# Architecture

## 设计目标

第一版架构以简单、可测、可解释为优先级。系统先用原生 Python 完成核心 RAG/Agent 闭环，保留清晰模块边界，后续再决定是否迁移到 LangGraph、LlamaIndex 或数据库服务。

核心原则：

- 先跑通端到端，再优化效果。
- 每个模块只承担一个清晰责任。
- 所有关键中间结果都能被 trace 记录。
- 失败样例必须能沉淀为测试，而不是只停留在口头复盘。
- 数据结构优先用标准库 dataclass、TypedDict 或简单 dict，避免过早引入重框架。

## 第一版组件

```text
raw data
  -> ingestion
  -> chunks
  -> retrieval
  -> rerank
  -> router / agent
  -> answer + citations
  -> trace
  -> evaluation / regression tests
```

## 模块职责

### ingestion

负责把多源文本转成统一 chunk。

输入：

- `data/raw/jd/`
- `data/raw/resume/`
- `data/raw/interview/`
- `data/raw/project_logs/`
- `data/raw/user_profile/`

输出：

- chunk id
- source type
- source path
- title
- text
- metadata

第一版只支持本地 `.txt` 或 `.md` 文件即可。

### 时效性感知检索

岗位 JD 是高频变化数据，同一职位可能出现下架、更新、重复发布或多平台信息不一致。第一版不实现爬虫、自动刷新或复杂数据库，但 ingestion 需要从数据结构上支持时效性建模。

`Document` 和 `Chunk` 的 metadata 统一包含：

```text
source_type: str
job_id: str
company: str
job_title: str
city: str
source_platform: str
source_url: str
first_seen_at: str
last_seen_at: str
status: active | expired | unknown
version: int
source_priority: int
content_hash: str
```

设计约定：

- `source_type` 仍由 `data/raw/` 下的一级目录推断，避免旧数据必须手动填写。
- `.md` 文件可以用简单 front matter 提供岗位 metadata；缺失字段使用默认值。
- chunk 切分后必须完整继承 document metadata，再追加 `chunk_index`、`source_file_name` 和 `char_count`。
- retrieval 阶段后续可以基于 `status`、`last_seen_at`、`version` 和 `source_priority` 做 freshness filter 或排序加权。
- `expired` JD 默认不应作为最终回答的强证据，但可以用于解释岗位变化或历史对比。
- 多平台重复 JD 后续可以用 `company + job_title + city + source_url/version` 做轻量去重或版本聚合。

### importer adapter

岗位数据采集不作为第一版核心依赖。招聘官网可能存在动态渲染、反爬、页面结构变化和合规限制，因此当前系统把“采集”拆成 adapter，把核心输入统一为 `JobPosting`。

当前实现：

- `manual`: 人工整理为标准字段后导入。
- `csv`: 从 CSV 文件导入一批岗位。
- `json`: 从 JSON 文件导入一批岗位。

future work：

- `crawler`: 官网或招聘平台采集 adapter，只负责产出同样的 `JobPosting` 结构。
- `api`: 如果平台提供合规 API，可新增 API adapter。
- `dedupe`: 根据 `job_id`、`content_hash`、`source_url`、`version` 做去重和版本合并。

`JobPosting` 字段：

```text
job_id: str
company: str
job_title: str
city: str
source_platform: str
source_url: str
description: str
requirements: str
first_seen_at: str
last_seen_at: str
status: active | expired | unknown
version: int
content_hash: str
```

`JobPosting -> Document` 转换规则：

- `description` 和 `requirements` 组成 `Document.text`。
- 岗位生命周期字段进入 `Document.metadata`。
- `source_type` 固定为 `jd`。
- 后续 chunk 切分时继续完整继承 metadata。

### retrieval

负责根据 query 找到候选 chunk。

第一版可以从最简单策略开始：

- 关键词匹配或轻量 TF-IDF。
- 返回 top k chunks。
- 保留 score 和 rank。

后续可替换为：

- embedding vector search。
- hybrid search。
- cross-encoder rerank。

### routing

负责识别用户意图，并决定优先检索哪些知识源。

初始 intent 可包含：

- `analyze_jd`: 分析岗位 JD。
- `match_resume`: 简历和 JD 匹配。
- `interview_prepare`: 面试题准备。
- `project_explain`: 项目经历讲解。
- `application_plan`: 投递计划和用户画像建议。
- `unknown`: 无法判断。

第一版可用规则实现，后续再换成模型分类或 LLM router。

### agent

负责把 router、retrieval、rerank、citation 和 answer generation 串起来。

第一版不做复杂多 Agent，只实现一个清晰的单轮流程：

1. 接收 query。
2. 判断 intent。
3. 选择知识源。
4. 检索 chunks。
5. rerank。
6. 基于证据生成回答。
7. 输出 citations。
8. 写入 trace。

### tracing

负责记录可观测 Agent Trace。

每次请求至少记录：

- request id
- query
- intent
- routed sources
- retrieved chunks
- rerank results
- selected citations
- tool calls
- final answer
- latency by stage
- error type
- created at

trace 先输出为本地 JSONL，方便后续分析和回放。

### evaluation

负责把系统输出转成指标。

第一版优先支持：

- Recall@k
- Citation Accuracy
- Router Accuracy
- Tool Success Rate
- Hallucination Rate 的人工标注入口或简化计算

评测数据放在 `tests/fixtures/` 或 `tests/regression/`，先用小样本跑通闭环。

## 数据对象草案

### Chunk

```text
id: str
source_type: str
source_path: str
title: str
text: str
metadata: dict
```

### DocumentMetadata

```text
source_type: str
job_id: str
company: str
job_title: str
city: str
source_platform: str
source_url: str
first_seen_at: str
last_seen_at: str
status: active | expired | unknown
version: int
source_priority: int
content_hash: str
```

### JobPosting

```text
job_id: str
company: str
job_title: str
city: str
source_platform: str
source_url: str
description: str
requirements: str
first_seen_at: str
last_seen_at: str
status: active | expired | unknown
version: int
content_hash: str
```

### Document

```text
source_path: str
title: str
text: str
metadata: DocumentMetadata
```

### RetrievalResult

```text
chunk_id: str
score: float
rank: int
reason: str | None
```

### AgentTrace

```text
request_id: str
query: str
intent: str
routed_sources: list[str]
retrieved_chunks: list[dict]
rerank_results: list[dict]
tool_calls: list[dict]
citations: list[dict]
latency_ms: dict[str, float]
error_type: str | None
answer: str
```

这些结构只是第一版约定，真正实现时可以根据测试逐步调整。

## 端到端流程

1. 用户输入一个实习求职相关 query。
2. router 识别 intent 和候选知识源。
3. retrieval 从对应 source type 中召回 top k chunks。
4. rerank 对候选结果重新排序。
5. agent 基于 top chunks 生成回答，并显式附 citations。
6. tracing 写入完整 trace。
7. evaluation 可读取 trace 和标注数据计算指标。
8. 失败样例进入 `tests/regression/`。

## 错误类型

第一版 trace 中建议统一错误类型，便于后续统计：

- `none`
- `ingestion_error`
- `retrieval_miss`
- `router_error`
- `rerank_error`
- `tool_error`
- `citation_error`
- `hallucination`
- `unknown_error`

## 后续演进方向

第二版可考虑：

- 用 LangGraph 表达 Agent 状态机。
- 用 LlamaIndex 管理索引、节点和检索器。
- 引入 embedding、vector db 和 hybrid search。
- 引入自动化评测报告。
- 增加 Web UI 展示 trace 和指标。
- 把失败样例自动转成 regression tests。
