# EvalRAG 项目地图

> 这是一份面向项目作者的导航文档。先理解本页，再按需阅读
> `architecture.md` 和具体代码，不要求一次看懂所有实现细节。

## 一句话定义

EvalRAG 是一个可观测、可评测、可回归的多源 RAG Agent Harness：
它不仅回答问题，还能说明证据从哪里来、失败发生在哪一步、修改后指标是否改善。

## 谁会使用

当前演示领域是中文实习求职资料，典型用户是正在准备求职的候选人。

典型问题：

- 单来源：“这个岗位要求哪些技术？”
- 多来源：“结合岗位要求和我的简历，分析匹配点与缺口。”
- 项目讲解：“如何向面试官解释 Context Builder？”
- 不可回答：“这个岗位的具体薪资是多少？”语料没有证据时应拒答。

EvalRAG 的架构不绑定求职领域。替换 Corpus 和 Evaluation Dataset 后，也可以用于
企业知识库、客服文档或技术手册。

## 两条主链

项目包含两条互相配合、但不要混在一起理解的链路。

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
  -> route_query()
  -> RouteDecision
     intent / routed_sources / matched_keywords

query + list[Chunk] + routed_sources
  -> Retriever.retrieve()
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

| 模块 | 为什么需要 | 输入 | 输出 | 当前状态 |
|---|---|---|---|---|
| Ingestion | 把五类原始材料统一成检索可用证据 | `.md/.txt/JSON/CSV` | `Document`、`Chunk` | 已完成 |
| Corpus | 固定评测使用哪批文档和 Chunk | `data/raw/` | manifest、versioned chunks、stats | 已完成 |
| Router | 缩小搜索范围并记录用户意图 | query | `RouteDecision` | Rule/Semantic/Hybrid 已完成 |
| Keyword Retriever | 提供可解释的词面检索 baseline | query、Chunks、source filter | `list[RetrievalResult]` | 已完成 |
| Dense Retriever | 召回同义表达和语义相近证据 | query、向量索引 | `list[RetrievalResult]` | 已完成 |
| Hybrid Retriever | 结合精确词面和语义召回 | Keyword/Dense ranks | fused `RetrievalResult` | 已完成 |
| Evidence Gate | 生成前判断证据是否足够 | route、retrieval results | 生成、重试或拒答决定 | 已完成 |
| Context Builder | 在预算内组织完整证据 | ranked results、budget | `BuiltContext` | 已完成 |
| Generator | 让模型只依据 Context 生成结构化答案 | query、BuiltContext | `GenerationResult` | 已完成 |
| Citation Validator | 防止模型引用不存在或重复的 id | GenerationResult、BuiltContext | `ValidationResult` | 已完成 |
| Pipeline | 串起在线链路并统一失败状态 | `RagRequest`、依赖和配置 | `RagResponse` | Evidence/格式单次重试已完成 |
| Trace | 保存一次请求每个阶段发生了什么 | 阶段结果、耗时、错误 | `AgentTrace` JSONL | 已完成 |
| Evaluation | 用固定标签和配置计算真实指标 | cases、chunks、config | run artifacts | frozen retrieval 与 live LLM E2E 已完成 |
| Regression | 防止已经修好的失败再次出现 | confirmed failure case | 自动化断言 | fixed/open 已完成 |
| Semantic Grader | 评估同义要点与事实证据支持性 | answer、expected points、cited Context | point/claim verdicts | P0-D6 已完成 |
| Serving | 提供薄外部调用入口 | CLI/HTTP request | `RagResponse` | 移入 P1，不是核心亮点 |

## 当前完成度

### 已完成

- v0.1 烟雾集：30 份五类中文语料、30 个自然 Chunk。
- 60 条 EvaluationCase：40 dev / 20 frozen test。
- Ingestion、规则 Router、Keyword Retriever。
- Context Builder、结构化 Generator、Citation Validator、单轮 Pipeline。
- 请求级 AgentTrace。
- Evaluation Runner 和正式 Keyword dev baseline。
- 123 个自动化测试覆盖离线模块、完整 Pipeline、DeepSeek adapter、live runner、
  Semantic/Grounding grader 与固定 Demo 导出；
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

P0-D1 至 P0-D7 已全部完成。README、最终实验报告、架构图、三个固定 Demo、简历
证据表和三分钟面试讲稿均已落盘；后续任务进入 P1 Backlog，不再把 P0 包装工作
拆成新的开发任务。

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
  Rate 的 Codex 审核仍不是独立人工语义核验结果。

### P0-D5 真实 LLM 补全

- 接入 `deepseek-v4-flash` 非思考 JSON Mode，密钥只从环境变量读取。
- 在不修改 frozen 标签、Router、Retriever 和 Evidence 配置的前提下，完整运行
  80 dev / 40 test：frozen Citation Validity 100%、Key-Point Coverage 68.89%、
  Abstention Accuracy 100%、End-to-End Success 67.50%、P95 4136.71 ms。
- frozen 实际调用 25 次，共 39,147 tokens，按运行时价格快照估算约 $0.0063。
- Validator 拦截了 1 条 `sufficient=false` 却携带 citations 的真实模型输出；主要
  失败仍是 Evidence Gate 来源覆盖过严造成 unexpected abstention。
- 当前 UAR 来自 Codex 审核流程，但落盘结果缺少逐 claim/evidence 判定，暂视为
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
- `docs/final_experiment_report.md` 汇总数据、检索、Router、可靠性、成本和失败闭环。
- `examples/fixed_demos/` 提供单来源、多来源和拒答三个真实 frozen 工件的脱敏重放。
- `docs/resume_evidence.md` 与 `docs/interview_guide_zh.md` 连接每条简历数字、代码路径、
  报告路径和三分钟讲解主线。

## MVP 完成标准

P0 达到以下条件才算简历可用 MVP：

1. v0.2 至少包含 100 份文档、300 个自然 Chunk 和 120 条 Query，并有同主题干扰项。
2. Keyword、Dense、Hybrid 在同一数据集上有正式对比。
3. Rule、Semantic、Hybrid Router 在同一数据集上有正式对比。
4. 系统能完成单来源、多来源回答和证据不足拒答。
5. Citation、拒答、端到端成功率和 P50/P95 有正式指标。
6. 至少一个真实失败完成“Trace 定位 -> 修复 -> 重评 -> Regression”。
7. frozen test 只在最终配置确定后运行。
8. 三个固定 CLI Demo 可复现，简历数字都能定位到仓库报告。

P1 才考虑向量数据库规模化、生成式 LLM Router、FastAPI/Docker 完整服务化、异步评测
队列、Trace 可视化和 LangGraph 重构。这些不是当前简历版本的前置条件。

## 最终简历介绍

以下表述只使用最终报告中已经落盘的结果：

> **EvalRAG：可观测、可评测的多源知识 RAG Agent Harness｜Python**
>
> 基于原生 Python 搭建多源 RAG Agent Harness，统一 Router、Keyword/Dense/
> RRF Hybrid Retrieval、语义混合 Router、预算感知 Context、结构化生成、
> Citation Validation 与请求级 Trace；构建包含同主题干扰项的版本化中文多源
> 评测集。
>
> 建立 Router Accuracy、Recall@k、MRR、Citation Validity、Abstention
> Accuracy 与端到端延迟评测；在 40 条 frozen test 上对比三种检索策略，
> RRF Hybrid 相比 Keyword 将 Recall@5 从 67.22% 提升至 74.44%，MRR 从
> 60.83% 提升至 66.78%，并记录检索 P95 从 99.55 ms 增至 802.50 ms 的代价。
>
> 设计证据门控和有限重试，在弱证据、非法 JSON 和非法引用场景中执行重试、
> 拒答或受控错误；将真实失败 Trace 固化为 Regression Case，验证检索、路由
> 与 Prompt 修改不会重新引入旧问题。

## 三分钟讲解顺序

### 第 1 分钟：问题与系统

“很多 RAG Demo 只能看到最后答案，答错后不知道是路由、检索、上下文还是模型
出了问题。我做 EvalRAG，是为了把一次请求变成可回放的 Pipeline，并让每次改动
都能在固定数据集上量化。”

接着用一句话说主链：

```text
Router -> Retriever -> Evidence Gate -> Context -> Generator
-> Citation Validator -> Response，同时写 Trace
```

### 第 2 分钟：数据与实验

“我先用 v0.1 烟雾集跑通 Harness，再扩展带同主题干扰项的 v0.2。固定 dev/test
后比较 Keyword、Dense 和 RRF Hybrid，同时对比 Rule、Semantic 和 Hybrid
Router；指标使用 Router Accuracy、Recall@k、MRR 和延迟，不把检索指标说成
答案准确率。”

### 第 3 分钟：可靠性与失败闭环

“模型输出的 JSON、citation id 和 sufficient 都不直接信任，而是由确定性
Validator 和 Evidence Gate 决定回答、重试或拒答。每次请求写一条 Trace；
我从真实失败中定位根因，修复后重跑 dev，并把 case 固化为 Regression。”

最后展示一个数字提升和一个失败案例，不继续罗列技术栈。

## 最值得深入掌握的三个技术点

### 1. Hybrid Retrieval 与可比较实验

需要讲清：

- Keyword 与 Dense 分别擅长什么。
- 为什么用 RRF 融合 rank，而不是直接相加两种不可比分数。
- Recall@k、MRR 和延迟如何证明改动是否有效。

### 2. 证据约束与受控拒答

需要讲清：

- Context、GenerationResult、Citation 和 ValidationResult 的关系。
- 为什么模型声明的 citation 和 sufficient 都不可信。
- Evidence Gate 如何在生成前决定继续、重试或拒答。

### 3. Trace、Evaluation 与 Regression 闭环

需要讲清：

- 一条 Trace 如何定位 failure stage。
- dev、frozen test 和 Regression 各自解决什么问题。
- 为什么保留失败 case 比只报告平均指标更有工程价值。

## 以后如何理解一个任务

每个新任务只按三层阅读：

1. **系统层**：回到本页，指出它在两条主链中的位置和存在理由。
2. **模块层**：只确认输入结构、核心处理、输出结构和上下游。
3. **代码层**：只读入口函数、核心数据结构和一条关键测试。

不要在第一次阅读时追求：

- 看完文件里的所有私有辅助函数。
- 记住每个 Python 语法细节。
- 理解与本任务无关的配置和异常分支。
- 同时补齐 Python、RAG、部署和全部指标。

先能复述数据流，再进入代码；先理解核心路径，再处理细节。
