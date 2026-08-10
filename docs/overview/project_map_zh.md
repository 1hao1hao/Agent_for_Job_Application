# EvalRAG 项目地图

> 这是 EvalRAG 的项目导航。先理解本页，再按需阅读
> `architecture.md` 和具体代码，不要求一次看懂所有实现细节。

## 一句话定义

EvalRAG 是一个可观测、可评测、可回归的多源 RAG Agent Harness：
它不仅回答问题，还能说明证据从哪里来、失败发生在哪一步、修改后指标是否改善。

## 谁会使用

当前演示领域是中文实习求职资料，典型用户是正在准备求职的候选人。

典型问题：

- 单来源：“这个岗位要求哪些技术？”
- 多来源：“结合岗位要求和我的简历，分析匹配点与缺口。”
- 项目讲解：“为什么 Context Builder 需要预算控制？”
- 不可回答：“这个岗位的具体薪资是多少？”语料没有证据时应拒答。

EvalRAG 的架构不绑定求职领域。替换 Corpus 和 Evaluation Dataset 后，也可以用于
企业知识库、客服文档或技术手册。

## 两条主链

项目包含两条互相配合、但不要混在一起理解的链路。

在线回答、离线评测和 P1 服务化链路的逐步解释见
[三条链路快速复习](system_flows_explained_zh.md)。本页只保留导航和核心数据结构。

### 在线回答链路

```text
RagRequest
  -> Router
  -> Retriever
  -> Evidence Gate
  -> Context Builder
  -> Generator
  -> Citation Validator
  -> RagResponse

每个阶段的状态
  -> AgentTrace
```

这条链解决：

> 一次用户问题如何得到有证据的回答或明确拒答？

### 离线评测链路

```text
Corpus + EvaluationCase + RunConfig
  -> Evaluation Runner
  -> 系统实际预测
  -> Metrics
  -> Case Results + Failures + Summary
  -> Regression Case
```

这条链解决：

> 系统哪里做得不好、一次修改是否真的提升、旧问题是否重新出现？

## 一次完整请求

下面只关注每个箭头传递的数据长什么样。

```text
用户问题字符串
  -> RagRequest
     query / request_id / top_k / retriever

RagRequest.query
  -> selected_router(query)
  -> RouteDecision
     intent / routed_sources / strategy / confidence / reason

query + list[Chunk] + routed_sources
  -> Retriever.__call__()
  -> list[RetrievalResult]
     Chunk + score + rank + reason

list[RetrievalResult]
  -> Evidence Gate（P0-D4）
  -> EvidenceDecision
     sufficient / retryable / insufficient + reason

query + ranked list[RetrievalResult] + context budget
  -> build_context()
  -> BuiltContext
     ContextItem 列表 + 模型可读 text + used/skipped ids

query + BuiltContext + prompt version
  -> generate_answer()
  -> GenerationResult
     answer / cited_chunk_ids / sufficient / reason

GenerationResult + BuiltContext
  -> validate_generation()
  -> ValidationResult
     valid / citations / issues

ValidationResult + Pipeline 状态
  -> RagResponse
     answered / insufficient_evidence / error

所有阶段数据与耗时
  -> AgentTrace
  -> JSONL
```

`EvidenceDecision` 已在 P0-D4 实现，保存 status、reason、来源覆盖、最高分和
重试计数；门槛按 Retriever 分开配置。

## 同一证据的不同面孔

这些结构不是互相无关的数据，而是同一份证据在链路中逐步增加信息。

| 阶段 | 数据结构 | 比上一阶段多了什么 | 给谁使用 |
|---|---|---|---|
| 数据导入 | `Document` | 统一正文与文档 metadata | Chunking |
| 数据切分 | `Chunk` | chunk id、来源、局部正文 | Retriever |
| 检索排序 | `RetrievalResult` | score、rank、匹配 reason | Evidence Gate、Context |
| 上下文组织 | `ContextItem` | 模型输入需要的稳定证据字段 | Generator、Validator |
| 上下文结果 | `BuiltContext` | 格式化文本、used/skipped ids、预算统计 | Generator、Trace |
| 模型输出 | `GenerationResult` | answer、模型声明的 citation ids 和充分性 | Validator |
| 合法引用 | `Citation` | 已确认存在于本轮 Context 的 id 和 source path | `RagResponse` |

最需要记住的一句话：

> `Chunk` 是原始证据；`RetrievalResult` 是排过名的证据；
> `ContextItem` 是准备交给模型的证据；`Citation` 是回答最终引用的证据。

## 核心模块地图

| 模块 | 为什么需要 / 解决什么 | 核心做法 | 输入 -> 输出 |
|---|---|---|---|
| Ingestion | 消除五类文档格式差异 | 读取 metadata、切分并继承来源信息 | raw files -> `Document` / `Chunk` |
| Corpus | 固定一次实验到底用了哪些证据 | 保存文档清单、版本化 Chunk 和质量统计 | `data/raw/` -> manifest / chunks / stats |
| Router | 避免每个问题都搜索全部来源 | Rule、Semantic 或二者决策融合 | query -> `RouteDecision` |
| Retriever | 从允许来源中找出最相关证据 | Keyword、BM25、Dense、RRF Hybrid | query + Chunks -> ranked `RetrievalResult` |
| Evidence Gate | 防止弱证据直接进入生成 | 检查 route、数量、最高分和来源覆盖 | route + results -> 生成 / 扩源 / 拒答 |
| Context Builder | 控制模型输入长度并保留完整证据 | 按 rank 在字符预算内拼接，不截半个 Chunk | ranked results -> `BuiltContext` |
| Generator | 把证据变成结构化回答 | 构造 Prompt、调用 LLM、解析 JSON | query + Context -> `GenerationResult` |
| Citation Validator | 阻止不存在、重复或组合矛盾的引用 | 对照本轮 Context 做确定性校验 | generation + Context -> `ValidationResult` |
| Pipeline | 让模块形成一次完整请求 | 串联主链，并控制扩源和格式重试各一次 | `RagRequest` -> `RagResponse` |
| Trace | 让失败可以定位到具体阶段 | 请求级保存各阶段结果、attempt、耗时和错误 | pipeline states -> `AgentTrace` |
| Evaluation | 判断修改是否真的改善 | 固定 Case/Config，保存预测并计算分模块与端到端指标 | cases + config -> Run Artifacts |
| Semantic Audit | 判断同义要点和事实主张是否有证据 | LLM 逐 point / claim 审核并保留 verdict | saved answers + labels/context -> audit results |
| Regression | 防止已修复问题重新出现 | confirmed failure 转成可执行固定断言 | fixed cases -> pass/fail |
| Serving / Worker | 提供 HTTP 调用和非阻塞批量评测 | FastAPI + PostgreSQL 状态 + Redis 队列 + Worker | HTTP request/job -> response/report |

## 当前完成度

### 已完成

- v0.1 烟雾集：30 份五类中文语料、30 个自然 Chunk。
- 60 条 EvaluationCase：40 dev / 20 frozen test。
- Ingestion、规则 Router、Keyword Retriever。
- Context Builder、结构化 Generator、Citation Validator、单轮 Pipeline。
- 请求级 AgentTrace。
- Evaluation Runner 和正式 Keyword dev baseline。
- 142 个自动化测试覆盖离线模块、完整 Pipeline、DeepSeek adapter、live runner、
  Semantic/Grounding grader、FastAPI、PostgreSQL/Redis Worker 与固定 Demo 导出；
  测试通过只证明代码行为稳定，不代表答案准确率。

当前 Keyword dev baseline：

| 指标 | 结果 |
|---|---:|
| Router Accuracy | 22.50% |
| Recall@3 | 39.72% |
| Recall@5 | 53.61% |
| MRR | 45.89% |

这些是后续优化的起点，不是最终效果。

### P0 状态

P0-D1 至 P0-D7 已全部完成。README、最终实验报告、架构图和三个固定 Demo 均已
落盘；后续任务进入 P1 Backlog。

### P0-D3 已完成

- v0.2：100 文档、310 Chunk、120 Query，80 dev / 40 frozen test。
- Keyword、Dense、Hybrid 已在相同 dev 集完成消融。
- BGE + RRF Hybrid 的 Recall@3/Recall@5 高于 Keyword，但 MRR 略降且 CPU 延迟
  明显增加，详细证据见 `reports/ablations/p0-d3-v02-dev-20260801/`。

### P0-D4 已完成

- Rule/Semantic/Hybrid Router 在相同 v0.2 dev 上 Accuracy 分别为 91.25%、
  87.50%、96.25%；Hybrid CPU P95 约 2600 ms，质量提升伴随明显延迟。
- Evidence Gate、source 扩展一次、格式修复一次和 attempt Trace 已接入 Pipeline。
- 删除过宽的 `公司` 规则词后，Rule Accuracy 从 88.75% 提升至 91.25%，并把
  已确认失败固化为可执行 regression test。

### P0-D5 已完成

- 实现统一 Reranker/Scorer 接口、Fake scorer 和 CrossEncoder adapter；dev 上的
  token-overlap candidate 造成 Recall 与 MRR 退化，因此冻结为关闭。
- 在 40 条 frozen test 上一次性比较 Keyword、Dense、Hybrid 和已声明 candidate；
  最终 Hybrid Recall@3 68.33%、Recall@5 74.44%、MRR 66.78%。
- 补齐 Citation、要点覆盖、拒答和端到端指标；deterministic extractive baseline
  的 Citation Validity 100%、Abstention Accuracy 90%、End-to-End Success 67.50%。
- fixed regression 1/1 通过，3 条 open case 明确排除在 pass rate 分母之外。
- 原 P0-D5 extractive baseline 不调用 LLM；随后已补跑真实模型。Unsupported Answer
  Rate 的模型辅助审核仍不是独立人工语义核验结果。

### P0-D5 真实 LLM 补全

- 接入 `deepseek-v4-flash` 非思考 JSON Mode，密钥只从环境变量读取。
- 在不修改 frozen 标签、Router、Retriever 和 Evidence 配置的前提下，完整运行
  80 dev / 40 test：frozen Citation Validity 100%、Key-Point Coverage 68.89%、
  Abstention Accuracy 100%、End-to-End Success 67.50%、P95 4136.71 ms。
- frozen 实际调用 25 次，共 39,147 tokens，按运行时价格快照估算约 $0.0063。
- Validator 拦截了 1 条 `sufficient=false` 却携带 citations 的真实模型输出；主要
  失败仍是 Evidence Gate 来源覆盖过严造成 unexpected abstention。
- 当前 UAR 来自模型辅助审核流程，但落盘结果缺少逐 claim/evidence 判定，暂视为
  provisional；P0-D6 将补齐可复查的 Grounding verdict，不能把当前 0% 当成独立
  人工审核或最终幻觉率。

### P0-D6 已完成

- 在不重跑 P0-D5 predictions 的前提下实现 Semantic Key-Point 与 Claim-Level
  Grounding 审核，每个 point/claim 保存 verdict、reason、引用和 evidence span。
- frozen Semantic Key-Point Coverage 从 lexical 68.89% 提升到 74.44%；这表示
  grader 能识别部分同义表达，不等于答案准确率提升。
- dev 找到 1 条 unsupported claim；frozen 23 条 answered 中 20 条可判断、3 条
  unknown。unknown 不按 supported 处理，因此严格 E2E 显示 unavailable。
- 正式工件：`reports/final/p0-d6-semantic-grounding-v0.2/`。

### P0-D7 已完成

- 根 README 先呈现问题、正式指标、两条主链和复现命令，再说明技术实现。
- `docs/evaluation/final_experiment_report.md` 汇总数据、检索、Router、可靠性、成本和失败闭环。
- `examples/fixed_demos/` 提供单来源、多来源和拒答三个真实 frozen 工件的脱敏重放。
- 公开文档和固定 Demo 均可回溯到代码、配置、Trace 与报告路径。

## MVP 完成标准

P0 达到以下条件才算可复现 MVP：

1. v0.2 至少包含 100 份文档、300 个自然 Chunk 和 120 条 Query，并有同主题干扰项。
2. Keyword、Dense、Hybrid 在同一数据集上有正式对比。
3. Rule、Semantic、Hybrid Router 在同一数据集上有正式对比。
4. 系统能完成单来源、多来源回答和证据不足拒答。
5. Citation、拒答、端到端成功率和 P50/P95 有正式指标。
6. 至少一个真实失败完成“Trace 定位 -> 修复 -> 重评 -> Regression”。
7. frozen test 只在最终配置确定后运行。
8. 三个固定 CLI Demo 可复现，对外指标都能定位到仓库报告。

P1-D1 已完成标准 BM25、FastAPI、PostgreSQL、Redis Worker 和 Docker Compose。下一步
先完成 P1-D2 Agent Runtime 生命周期解耦和 P1-D3 多轮 Context/Memory 预算消融；它们
分别补强 Harness 编排和长上下文治理。向量数据库仍只在 profiling 证明必要时实施，
Skill Registry、LLM Router、Trace 可视化和 LangGraph 只保留为有实验问题时的对照项。

## 核心技术点

### 1. Hybrid Retrieval 与可比较实验

关键问题：

- Keyword 与 Dense 分别擅长什么。
- 为什么用 RRF 融合 rank，而不是直接相加两种不可比分数。
- Recall@k、MRR 和延迟如何证明改动是否有效。

### 2. 证据约束与受控拒答

关键问题：

- Context、GenerationResult、Citation 和 ValidationResult 的关系。
- 为什么模型声明的 citation 和 sufficient 都不可信。
- Evidence Gate 如何在生成前决定继续、重试或拒答。

### 3. Trace、Evaluation 与 Regression 闭环

关键问题：

- 一条 Trace 如何定位 failure stage。
- dev、frozen test 和 Regression 各自解决什么问题。
- 为什么保留失败 case 比只报告平均指标更有工程价值。
