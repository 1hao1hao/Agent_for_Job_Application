# EvalRAG 项目地图

本文是理解项目的唯一主地图。先读主链和模块表，再按需进入代码；实验数字以链接的 Run Artifact 为准。

## 一句话定义

EvalRAG 是一个面向中文求职知识推理的图增强、自适应 RAG Agent Harness：它把岗位 JD、
技术面经、项目资料、简历和用户画像统一为可追溯证据，经路由、混合/图检索、证据门控、
上下文编排和引用约束生成回答，并用 Trace、Benchmark、Regression 与 CI Gate 约束每次迭代。

## 四条主链

### 1. 离线知识构建

```text
公开/脱敏资料
  -> Importer / Ingestion
  -> Document
  -> 标题、段落与句子边界感知切分
  -> Chunk
  -> SHA-256 / SimHash 去重 + provenance 审核
  -> manifest + versioned chunks + stats
  -> BM25 / BGE Dense Index / Job-Skill-Experience Graph
  -> 文件索引或 pgvector / Neo4j adapter
```

- `manifest`：每份原始材料的来源、采集时间、许可、公开/脱敏和审核状态清单。
- `versioned chunks`：固定版本的 Chunk 快照，让索引、评测标签和实验结果能互相对应。
- `stats`：文档数、Chunk 数、长度分布、空文档、重复 hash 和来源占比等质量统计。

### 2. 在线回答

```text
RagRequest
  -> AgentRuntime 创建 root run / Trace
  -> Feedback Hybrid Router
  -> Query Analyzer + Adaptive Retriever
       BM25 / Dense / RRF / Graph + Vector
       低置信复杂 Query -> 按需 CrossEncoder
  -> Evidence Gate
       sufficient  -> Context Engine
       retryable   -> 去掉 source filter，全库检索一次
       insufficient -> 拒答
  -> Context Engine
       system + query + profile + history + memory + evidence
       在 token budget 内去重、排序和完整证据选择
  -> Generator
  -> Model Gateway -> DeepSeek / 备用 OpenAI-compatible Provider
  -> JSON 解析；格式错误最多修复生成一次
  -> Citation Validator
  -> RagResponse
  -> Trace spans + checkpoint / replay 工件
```

两类重试解决不同问题：

- **扩源重试**解决“路由过滤过窄导致可能漏证据”，最多一次，并不是重复执行同一检索。
- **格式修复重试**解决“模型答案 JSON 不符合契约”，最多一次，不改变检索证据。

### 3. 离线评测与回归

```text
Corpus + EvaluationCase + RunConfig
  -> Router / Retriever / Pipeline 真实运行
  -> predictions + traces + latency/tokens
  -> Retrieval / Answer / Grounding Metrics
  -> case_results + failures + summary + report
  -> Trace 定位根因
  -> open / fixed RegressionCase
  -> 自动回归 + CI Evaluation Gate
  -> 最终配置一次性运行 frozen test
```

- `dev` 用于反复比较策略和调参；`frozen test` 只在配置冻结后运行，避免看答案调系统。
- `failure` 是系统预测不符合标签或可靠性规则的 Case，不是评测脚本崩溃。
- `open regression` 记录待修问题；`fixed regression` 自动断言已修问题不能复发。

### 4. 服务与异步评测

```text
POST /v1/query
  -> FastAPI -> AgentRuntime / RagPipeline
  -> RagResponse + trace_id
  -> PostgreSQL 保存请求、Run 和 Trace

POST /v1/evaluation-jobs
  -> PostgreSQL 幂等创建 queued Job
  -> Redis Queue 只传递 job_id
  -> Evaluation Worker: queued -> running -> succeeded / failed
  -> 文件卷保存完整报告
  -> PostgreSQL 保存最终状态、摘要和 report_path
```

PostgreSQL 是任务状态的真相来源；Redis 负责短期队列和最近会话缓存；文件系统保存体积较大的
实验工件。Docker Compose 统一启动 API、Worker、PostgreSQL 和 Redis。

## 主链数据结构

| 数据结构 | 它代表什么 | 主要生产者 -> 消费者 |
|---|---|---|
| `Document` | 原文、路径和统一 metadata 的文档 | Importer -> Chunker |
| `Chunk` | 可检索、可引用的最小证据单元 | Chunker -> Index / Retriever |
| `RagRequest` | 一次查询及运行配置 | API/CLI -> Runtime |
| `RouteDecision` | intent、目标 sources、置信度和匹配依据 | Router -> Retriever / Gate |
| `RetrievalResult` | Chunk、score、rank 和策略解释 | Retriever -> Gate / Context |
| `EvidenceDecision` | sufficient / retryable / insufficient 及原因 | Gate -> Pipeline |
| `ManagedContext` / `BuiltContext` | 在预算内选出的模型输入证据和记忆 | Context Engine -> Generator |
| `GenerationResult` | 模型解析后的 answer、引用 ID、sufficient 和 reason | Generator -> Validator |
| `Citation` | 合法 chunk_id 与 source_path | Validator -> RagResponse |
| `AgentTrace` / `Span` | 一次请求及各阶段尝试、耗时、token 和错误 | Runtime/Pipeline -> Trace Store |
| `EvaluationCase` | Query 与 expected intent/source/chunk/points/answerable 标签 | Dataset -> Runner |
| `RunConfig` / Run Artifacts | 固定策略、版本和逐 Case 结果 | Runner -> Report / CI Gate |

## 核心模块与算法

| 模块 / 方法 | 为什么引入 | 如何设计，为什么这样设计 | 实际解决的问题 / 效果 |
|---|---|---|---|
| Document / Chunk Schema | 五类资料格式不同，下游不应分别适配 | 统一正文、source_type、路径、时效和 provenance；Chunk 完整继承 metadata | 检索、引用、图关系和评测都消费同一证据结构 |
| 边界感知 Chunking | 仅按固定长度会切断句子，仅按句号会产生过短 Chunk | 先按标题/空行识别段落，再按句子边界拆长段，最后合并到预算，不截断正常短句 | 保留语义完整性；同集消融结果保存在 `reports/ablations/p1-chunking-v02-dev-20260812/` |
| Manifest / 版本化 Corpus | 数据变化会使实验无法复现 | 固定 dataset version，保存来源、revision、许可、hash、脱敏和审核状态 | v0.3 将 669 份输入清洗为 658 份文档、4208 个 Chunk |
| Rule Router | 需要低成本、可解释 baseline 和精确术语规则 | Query token 命中 intent keywords，稳定冲突排序 | 能解释 `matched_keywords`，但难覆盖同义改写 |
| Semantic Router | 关键词无法识别同义表达 | BGE 编码 Query 与 intent prototypes；用最低相似度和第一/第二差值拒绝模糊路由 | 找回语义表达，同时暴露 prototype 污染风险 |
| Hybrid + Feedback Router | 单一路由策略各有盲区，已确认错误应可积累 | Rule 与 Semantic 按置信度融合；反馈离线提取短意图锚点，经 shadow/dev gate 后发布版本 | v0.2 dev 从 Hybrid 96.25% 到反馈版 100%；v0.3 新分布上 Feedback 与 Rule 的 Source Exact 同为 43.75%，说明旧锚点不能直接迁移 |
| BM25 | token overlap 没有词频、逆文档频率和长度归一 | 标准 BM25 公式，离线统计文档频率与平均长度，接口与其他 Retriever 一致 | 提供可解释稀疏检索基线；精确术语快，但同义召回弱 |
| Dense Retrieval | BM25 依赖词面重叠 | `BAAI/bge-small-zh-v1.5` 离线编码 Chunk，查询时只编码 Query并计算余弦相似度 | 找回“参与智能问答开发”等语义改写，代价是更高 CPU 延迟 |
| RRF Hybrid | BM25 与 Dense 原始分数不可直接相加 | Reciprocal Rank Fusion 只融合名次，去重后稳定排序 | v0.2 frozen 相比 Keyword，Recall@3 55.56% -> 68.33%，MRR 60.83% -> 66.78% |
| Query Analyzer / Adaptive Retrieval | 所有 Query 都跑最重策略会浪费延迟 | 检索前从 Query 和 Router sources 提取 7 个可解释特征选择 BM25、Dense、Hybrid 或 Graph+Vector；检索后再计算候选置信度，决定是否 CrossEncoder 重排 | 将“首次检索选路”与“低置信补救”分成两阶段；但 v0.3 dev 因规则漏触发关系 Query，Adaptive Recall@5/MRR 为 48.33%/42.21%，低于固定 Graph+Vector 的 58.33%/52.10% |
| Job-Skill-Experience Graph | 向量相似不等于能连接“岗位要求-项目技能-个人经历” | 抽取 Job、Skill、Project、Experience、Technology、Company 节点和有向关系，全部回指 Chunk | v0.3 构建 3098 节点、2741 边，支持可解释多跳证据 |
| Graph + Vector Retrieval | Graph-only 容易漏文本，Vector-only 缺关系路径 | 实体链接和有界多跳召回图证据，再用 RRF 与向量 Chunk 融合 | 80 条 frozen 上相对 BM25：Recall@5 46.67% -> 63.33%，MRR 35.19% -> 57.58%；P95 15.50 -> 1209.40 ms |
| CrossEncoder Rerank Policy | 召回改善后仍可能存在前排噪声，但全量重排成本高 | 对同一 BM25+Dense+RRF 候选分别运行 never / always / low-confidence，用同一 MiniLM revision 控制变量 | v0.3 dev：always 将 MRR 44.31% -> 49.22%但 P95 1252 -> 2799 ms；按需调用率 18.12%、MRR 45.18%、P95 2175 ms，无 Pareto 最优 |
| Evidence Gate | 检索有结果不等于证据足够生成 | 检查 route、结果数量、最高分和 required-source coverage；只允许一次扩源重试 | 将生成、重试、拒答变为可解释决策，避免弱证据直接进入模型 |
| Source-Balanced Context | 纯 rank 贪心容易被单一来源占满预算 | 在紧预算下优先保证 required sources，再按 rank 补充，且不截断单个 Chunk | 1200 字符预算下完整来源覆盖率 30.19% -> 54.72%，相关证据召回下降 0.94 pp |
| Context Engine / 分层记忆 | 多轮 Prompt 会膨胀并重复读历史 | 对 system、query、profile、recent history、semantic memory、evidence 统一 token 预算和去重；PG 持久化、Redis 缓存 | 60 组场景中 semantic memory 平均 66.22 tokens、重复历史读取 3 -> 0；该结果不是回答准确率 |
| Generator JSON Contract | 自由文本难以校验引用和拒答状态 | Prompt 约束只依据 Context，模型返回 answer/cited_chunk_ids/sufficient/reason；解析失败受控重试一次 | 生成结果可被程序验证，而不是把模型输出直接交给用户 |
| Model Gateway | 外部模型有 timeout、429、5xx 和供应商故障 | Provider Protocol + 有界退避、并发 semaphore、熔断和 fallback，鉴权错误不盲重试 | 6 类 Fake 故障注入验证控制流，真实 DeepSeek primary smoke 通过；备用 Provider 未做真实 fallback |
| Citation Validator | 模型可能返回不存在或重复的证据 ID | 校验 ID 存在性、去重和 sufficient/citation 组合，合法后才构造 Citation | 非法引用不能进入最终回答；Citation Validity 不等于事实支持度 |
| AgentRuntime / Checkpoint / Replay | HTTP、CLI、Worker 各自编排会产生行为漂移，中断后也难恢复 | 统一 Runtime 创建 root run 和 spans；保存配置 fingerprint 与阶段 checkpoint；Fake replay 重放固定输入 | 三入口共享生命周期，能恢复或拒绝误用旧状态，并复现实验控制流 |
| Trace / Regression / CI Gate | 指标下降只看均值难定位，已修问题可能复发 | 逐阶段记录 route、候选、attempt、reason、latency/token；失败分为 open/fixed case，CI 检查 reference 阈值 | 4 条 fixed regression 全部通过，失败能定位到具体 Case 和阶段 |
| Semantic Key-Point / Claim Grounding | 字符串包含会漏判同义表达，“引用合法”也不代表事实受支持 | LLM grader 分别判断每个 expected point 和每条 factual claim，保存 verdict、evidence span、reason 与版本 | 结论可回查到要点、断言和证据；unknown 不被伪装成 supported |
| FastAPI + PostgreSQL + Redis Worker | Query 和长耗时评测不能只靠脚本同步运行 | FastAPI 暴露稳定契约；PostgreSQL 保存状态；Redis 仅传 job_id；Worker 独立执行并落盘报告 | 支持 Query/Trace 查询和幂等异步评测，服务重启后任务状态仍可追踪 |

## Hybrid + Feedback Router 详解

### 1. Hybrid Router 本身怎么决策

```text
query
  -> Rule Router     -> rule RouteDecision
  -> Semantic Router -> semantic RouteDecision + score + margin
  -> HybridRouter
  -> final RouteDecision
```

Hybrid Router 不是 RRF，而是有优先级的决策规则：

1. Rule 和 Semantic 选中同一 intent：使用 Rule，reason 为 `rule_semantic_agree`。
2. 两路冲突，但 Semantic 分数 `>= 0.55`、第一/第二 intent 差值 `>= 0.04`，
   且 Rule 只命中不超过 2 个关键词或返回 unknown：Semantic 覆盖弱 Rule。
3. 其他冲突：保留 Rule，避免低置信语义结果覆盖可解释的精确规则。

Trace 会保留 `rule_intent / semantic_intent / semantic_score / semantic_margin /
rule_keyword_count`，所以可以看出最终为什么选 Rule 或 Semantic。

### 2. Feedback 是什么

Feedback 是一条**已确认的误路由记录**，保存在版本化 JSONL，主要包含：

```text
query
original_intent / original_sources
corrected_intent / corrected_sources
failure_type
router_version
source=evaluation | user_confirmed
```

它不会在用户请求期间直接修改 Router。这样可以避免一条错误反馈立即污染线上行为。

### 3. “离线提取短意图锚点”是什么

评测 Query 中有这样的失败样例：

```text
岗位分析：怎样把口语问题改成可检索表达？
项目怎么讲：怎样识别已过期的招聘信息？
```

`_feedback_prototype()` 优先取中英文冒号前 `2~16` 个字的前缀，得到：

```text
“岗位分析” -> analyze_jd + [jd]
“项目怎么讲” -> project_explain + [project_logs, resume]
```

这个短前缀就是**意图锚点**。不直接把整个长 Query 当 prototype，是因为长句中的
“模型、检索、岗位”等共享主题可能被错误学到某一 intent，导致相邻 Query 误判。

候选 Router 会做两件事：

- 把短锚点追加到正确 intent 的 Semantic prototypes。
- `FeedbackRouter` 对“完全等于锚点”或“以 `锚点：` 开头”的 Query 直接使用已确认修正；
  其他 Query 仍委托给原 Hybrid Router。

P1-D6 shadow 脚本同时验证了“prototype 追加 + 锚点直接修正”；当前服务从
Registry 构建 active Router 时，生效的是 `FeedbackRouter` 锚点直接修正层，未在请求期间
动态更新 Semantic prototypes。

### 4. “shadow/dev gate 后发布”是什么

```text
confirmed feedback JSONL
  -> 构建 candidate Router
  -> baseline Hybrid 和 candidate 在同一 dev 上运行
  -> compare_router_versions()
  -> gate passed
  -> RouterVersionRegistry.publish()
  -> active_version 指向新版本
```

`shadow` 表示候选版本只做对照运行，先不接管在线请求。Gate 检查：

| Gate | 通过条件 |
|---|---|
| Accuracy | 候选版不低于 baseline |
| Unknown Precision / Recall | 未知意图识别不退化 |
| P95 | 不超过 baseline 的配置比例，当前为 1.25 倍 |
| No Case Regression | 不允许原先正确的任意 dev Case 被改错 |

只有全部通过，候选版才会写入 Router Registry 并成为 `active_version`；否则保留原版本。
Registry 保留 parent version、feedback dataset 和 report path，因此可以审计和回滚。

P1-D6 在 v0.2 dev 上将 Hybrid Accuracy 从 96.25% 提高到 100%；但同一批旧锚点
在 v0.3 新 Query 分布上没有超过 Rule，说明这是有边界的离线反馈修正，不是可泛化的
“在线自学习 Router”。

## Job-Skill-Experience Graph 详解

### 1. 图的输入和输出

```text
evalrag_v0.3 list[Chunk]
  + configs/graph/job_skill_v0.2.json 实体/别名词典
  -> DeterministicGraphExtractor.extract()
  -> 每个 Chunk 的 GraphExtraction(nodes, edges)
  -> build_knowledge_graph()
  -> 去重、合并 chunk_ids/provenance、稳定排序
  -> KnowledgeGraph JSON
```

当前不用 LLM 抽取实体和关系，而是使用 metadata、标题和版本化别名词典做确定性抽取。
这样容易复现和追溯，但只能识别词典覆盖的 Skill/Technology。

v0.3 图工件是 `job-skill-experience-v0.2.json`，包含 3098 个节点、2741 条边，
节点和边均能回到原始 Chunk、URL/文档 ID 和时间 provenance。

### 2. 节点是怎么得到的

每个 Chunk 先建立一个**文档实体节点**：

| Chunk source | 文档节点类型 | 节点名来源 |
|---|---|---|
| `jd` | `job` | metadata 中的 `job_title`，缺失时用 Chunk 首行/标题 |
| `resume` | `experience` | Chunk 首行标题 |
| `project_logs` | `project` | Chunk 首行标题 |
| `user_profile` | `experience` | Chunk 首行标题 |
| `interview` | `interview_question` | Chunk 首行标题 |

然后补充实体：

- JD metadata 中存在 `company`：创建 `company` 节点。
- JD metadata 中存在 `city`：创建 `location` 节点。
- Chunk 文本命中词典中的规范名或别名：创建 `skill` 或 `technology` 节点。

例如词典把 `RAG / 知识库问答 / retrieval augmented generation` 统一到
`skill:检索增强生成`，把 `BGE / sentence-transformers / Hugging Face` 等文本命中到相应
Technology。节点 ID 由 `node_type + 规范化名称` 做 SHA-1 短哈希生成；同名同类型实体会合并，
并汇总所有支持它的 `chunk_ids`。

### 3. 边是怎么得到的

| 来源/条件 | 边 | 含义 |
|---|---|---|
| JD + company | `company -posts-> job`、`job -belongs_to-> company` | 公司发布岗位 |
| JD + city | `job -located_in-> location` | 岗位工作地 |
| JD 命中 Skill/Technology | `job -requires-> concept` | 岗位要求技能或技术 |
| Resume/User Profile 命中概念 | `experience -demonstrates-> concept` | 经历证明某项能力 |
| Project Log 命中 Skill | `project -demonstrates-> skill` | 项目证明某项能力 |
| Project Log 命中 Technology | `project -uses-> technology` | 项目使用某技术 |
| Interview 命中概念 | `interview_question -asks_about-> concept` | 面试题考察某概念 |
| 同一 Chunk 同时命中 Skill 和 Technology | `skill -related_to-> technology` | 文本证据中两者相关 |

每条边也保存支撑这个关系的 `chunk_ids` 和 provenance。边 ID 由
`edge_type + source_node_id + target_node_id` 做稳定 SHA-1 短哈希；多个 Chunk 产生同一关系时，
边不重复创建，而是合并证据 ID。

### 4. 图实际长什么样

```text
示例公司 (company)
  -posts-> 大模型应用开发实习生 (job)
              -requires-> 检索增强生成 (skill)
              -located_in-> 广州 (location)

EvalRAG 项目 (project)
  -demonstrates-> 检索增强生成 (skill)
                      -related_to-> Transformers/BGE (technology)
```

关键点是：图节点不是最终给 LLM 的证据。节点和边只是用于寻路的结构，
其 `chunk_ids` 指向的原始 Chunk 才会进入 Context 和 Citation。

## Graph + Vector Retrieval 详解

### 1. Graph Retrieval 怎么召回 Chunk

```text
query
  -> QueryDecomposer.decompose()
  -> 实体链接 + 关系标记 + 子目标
  -> GraphRetriever
  -> 从命中实体开始有界 BFS
  -> 收集途经节点和边的 chunk_ids
  -> ranked list[RetrievalResult]
```

1. **实体链接**：用图中节点名和 aliases 在 Query 中做子串匹配。例如 Query 中的 `RAG`
   链接到 `skill:检索增强生成`。
2. **关系识别**：“要求/岗位”对应 `requires`，“证明/经历”对应 `demonstrates`，
   “项目/使用”对应 `uses`等。
3. **有界遍历**：从命中节点开始 BFS，当前最多 3 hops、160 节点、100 ms，防止图遍历失控。
   数据中的边保留方向，但检索邻接表允许双向走，因此能沿 `job -> skill <- project`
   找到与岗位技能相关的项目。
4. **Chunk 召回**：访问一个节点时收集它的 `chunk_ids`，分数为 `1/(1+hops)`；
   经过一条边时也收集边的 `chunk_ids`，分数约为 `0.8/(1+hops)`，若边类型命中
   Query 关系再加 `0.2`。
5. **去重排序**：同一 Chunk 被多条路径找到时保留最佳路径，再按图分数降序、hops 升序和
   chunk ID 稳定排序。

返回的 `RetrievalResult.details` 保留 `graph_path / graph_edge_ids / graph_hops /
path_valid`，因此可以说明某个 Chunk 是通过哪条关系路径召回的。如果 Query 没有链接任何图实体，
Graph Retriever 返回空结果，由上层保留 Vector 候选。

### 2. Vector Retrieval 是不是 Dense

**从概念上说**，Vector Retrieval 通常指 Dense Retrieval。

**但在当前 `GraphVectorRetriever` 代码里**，`vector_retriever` 是可注入的统一 Retriever；
factory 实际传入的是 `HybridRetriever`，即：

```text
稀疏路（BM25，或旧配置的 Keyword）
  + BGE Dense Retrieval
  -> 第一层 RRF
  -> vector_results
```

所以当前项目中的“Graph + Vector”更准确地说是：

```text
Graph Retrieval + (Sparse + Dense Hybrid Retrieval)
```

P1-D9 的 `graph_vector_v0.3` 明确配置 `adaptive_lexical="bm25"`，所以该实验的 Vector 分支是
BM25 + BGE Dense + RRF。旧的 P1-D7 frozen 配置未显式指定 `adaptive_lexical`，保留了
Keyword + Dense 的兼容默认；不应把两组实验的稀疏分支混为同一配置。

### 3. Graph 和 Vector 如何融合

`GraphVectorRetriever` 先把候选数扩到 `top_k * 4`，分别运行 Vector 和 Graph，
对相同 `chunk_id` 去重，再使用外层 RRF：

```text
graph_vector_score
  = 1 / (60 + vector_rank)   # 如果被 Vector 召回
  + 1 / (60 + graph_rank)    # 如果被 Graph 召回
```

- 同时被两路召回的 Chunk 通常更靠前。
- 只被其中一路召回的 Chunk 也会保留，避免 Graph 或 Vector 某一路漏召回就直接丢证据。
- Graph 没有结果时直接 fallback 到 Vector 排名。
- 融合结果保留 `vector_rank / graph_rank / graph_path / fused_score`，供 Trace 和失败分析。

因为 Vector 分支内部已经做了一次 Sparse + Dense RRF，Graph + Vector 是**两层 rank 融合**：

```text
BM25 + Dense -> 内层 RRF -> Vector Hybrid
Graph + Vector Hybrid -> 外层 RRF -> final RetrievalResult
```

### 4. 一个完整召回例子

Query：“哪个项目能证明我符合 RAG 岗位要求？”

```text
Query 命中 RAG -> 链接到 skill:检索增强生成

Graph 路：
  大模型岗位 -requires-> RAG <-demonstrates- EvalRAG 项目
  -> 召回岗位 Chunk、requires 边的 JD Chunk、项目 Chunk

Vector 路：
  BM25 命中“RAG/岗位/项目”
  + BGE 召回“检索增强、知识库问答”等语义候选
  -> 内层 RRF

外层 RRF：
  同时被图路径和文本检索找到的 EvalRAG 项目 Chunk 获得更高融合分
  -> 进入 top-k Context
```

这就是图的实际用处：Dense 只能说“两段文本语义相似”，图可以明确给出
“岗位要求 RAG，该项目证明 RAG”的关系路径，同时最终仍返回可引用的原始 Chunk。

## Query Analyzer + Adaptive Retriever 详解

### 1. 它不是一次决策，而是两阶段决策

```text
Query + Router.routed_sources
  -> QueryAnalyzer.analyze()
  -> QueryFeatures
  -> QueryAnalyzer.choose_strategy()
  -> 选择首次检索策略
  -> 产生 candidates
  -> _retrieval_confidence()
  -> 高置信：直接返回
  -> 低置信：根据 rerank policy 决定是否 CrossEncoder 重排
  -> list[RetrievalResult] + RetrievalDecision Trace
```

- **检索前**：Query Analyzer 只看 Query 文本和 Router 给出的 `source_types`，选择首次检索策略。
- **检索后**：Adaptive Retriever 才能看到候选排名和分数，计算置信度并决定是否重排。
- 因此“低置信”**不是** Query Analyzer 特征，而是首次检索后的结果质量信号。

### 2. 可选的检索与重排算法

| 策略 | 实际做什么 | 适合的 Query |
|---|---|---|
| `bm25` | 根据词频、逆文档频率和文档长度计算词面相关性 | 短 Query、岗位名、技术名和精确术语 |
| `dense` | 用 `BAAI/bge-small-zh-v1.5` 编码 Query，与离线 Chunk 向量计算余弦相似度 | 同义改写、词面不重合的语义问题 |
| `hybrid` | 分别运行稀疏检索和 Dense，再用 RRF 融合 rank | 多来源、含精确术语但也需要语义召回的 Query |
| `graph_hybrid` | 运行有界 Graph Retrieval，再用 RRF 与 Vector Hybrid 候选融合 | 需要连接“岗位-技能-项目/经历”的跨文档关系 Query |
| CrossEncoder Rerank | 将 `Query + 候选 Chunk` 成对编码并重新评分，再用加权 RRF 融合原排名与重排名 | 首次候选置信度低且希望改善前排排序时 |

`hybrid` 的稀疏路可配置：`adaptive_lexical="bm25"` 时是 BM25 + Dense；未配置时
工厂保留旧的 Keyword + Dense 默认行为。P1-D9 的 Reranker 控制实验明确使用
BM25 + Dense + RRF；P1-D7 已冻结的旧配置没有因本轮文档更新而改写。

### 3. QueryFeatures 的数值从哪里来

| 特征 | 计算方法 | 作用 |
|---|---|---|
| `char_count` | `query.strip().lower()` 后的字符数 | `<= 8` 且没有精确术语时，视为短词面 Query |
| `source_count` | Router 输出的 `routed_sources` 数量 | 两种及以上来源时视为多来源 Query |
| `has_exact_term` | 命中 `BM25/RRF/RAG/Trace/...` 技术词表，或匹配英文/数字技术名正则 | 保留稀疏检索的精确匹配优势 |
| `has_semantic_rewrite` | 命中“换句话、同义、通俗、口语、改写”等标记词 | 表示词面重合可能较弱，倾向 Dense |
| `is_multi_source` | Router 来源数 `>= 2`，或命中“结合、对比、匹配、综合、个人经历” | 同时保留词面和语义召回路 |
| `is_cross_document` | 命中“哪些项目、能否证明、对应起来、关系路径”等词，或同时出现“岗位”与“项目/经历/简历” | Graph 可用时触发 Graph + Vector |
| `is_unanswerable_route` | Router 明确返回空 `source_types=set()` | 直接返回空检索结果，不调用实际 Retriever |

这些都是**确定性规则特征**，不是 LLM 或学习模型预测出来的“通用语义复杂度”。
好处是便宜、可解释和易于写 Trace；局限是 marker 没覆盖新表达时会选错策略。

### 4. 首次检索策略的优先级

`choose_strategy()` 从上往下判断，命中后立即停止：

| 优先级 | 条件 | 策略 | 例子 |
|---:|---|---|---|
| 1 | Router 明确无可搜来源 | 不执行检索 | “给我未公开的公司薪资名单” |
| 2 | Graph 可用且 `is_cross_document=true` | `graph_hybrid` | “哪个项目能证明我符合这个岗位的 RAG 要求？” |
| 3 | `is_multi_source=true` | `hybrid` | “结合 JD 和简历分析匹配度” |
| 4 | 明确语义改写且没有精确术语 | `dense` | “换句话说明如何减少召回偏科” |
| 5 | Query 不超过 8 字且没有精确术语 | `bm25` | “岗位职责” |
| 6 | 命中精确技术术语 | `hybrid` | “RRF 是什么？” |
| 7 | 上述均未命中 | `hybrid` | 普通但无法明确判定偏词面还是偏语义的 Query |

这里的“自适应”是**一个可解释的规则策略选择器**，不是训练出来的 Policy Model。

### 5. 检索置信度怎么算

首次检索后，`_retrieval_confidence()` 把四项信号加权到 `0~1`：

```text
confidence = 0.25 * quantity
           + 0.20 * margin
           + 0.25 * source_coverage
           + 0.30 * agreement
```

| 信号 | 怎么算 | 直观含义 |
|---|---|---|
| `quantity` | `min(1, 候选数 / top_k)` | 候选数量是否足够 |
| `margin` | `(第1名分数 - 第2名分数) / 第1名分数`，限制在 0~1 | 第一名是否明显优于第二名 |
| `source_coverage` | top-k 已覆盖 Router sources 数 / Router 要求数 | 多来源问题的必需来源是否齐全 |
| `agreement` | Hybrid 第一名同时由稀疏和 Dense 召回时为 1，否则为 0；Graph Hybrid 同理检查 Graph/Vector | 两条独立召回路是否相互印证 |

当前 v0.3 配置门槛为 `0.65`。例如候选够 5 条，但前两名几乎同分、缺一个
Router 要求的来源，且首位只被单路召回，置信度就会低于门槛。

### 6. 什么时候重排

| `rerank_policy` | 行为 |
|---|---|
| `never` | 不调用 CrossEncoder，用于 baseline/消融 |
| `always` | 只要有候选就重排，用于评估质量上限和延迟代价 |
| `low_confidence` | `confidence < threshold` 时重排，是按需模式 |

Graph + Vector 结果当前不再进入 CrossEncoder；只对 BM25、Dense 或 Hybrid 候选执行
重排。CrossEncoder 也不直接覆盖原排名，而是用下式做保守融合：

```text
fused_score = 2 / (60 + original_rank)
            + 1 / (60 + rerank_rank)
```

原排名权重为 2，重排名权重为 1，目的是让 CrossEncoder 改善前排，同时降低它将已验证
召回排序完全推翻的风险。最终的 `strategy / confidence / rerank_invoked / reason /
candidate_count / model version` 都会写入 `RetrievalDecision` 和 Trace。

### 7. 当前实验告诉我们什么

- 固定 Graph + Vector 在 v0.3 dev 的 Recall@5/MRR 为 58.33%/52.10%，当前 Adaptive Graph
  为 48.33%/42.21%；根因是手工 cross-document markers 没覆盖所有关系型问法。
- Always Rerank 把 MRR 从 44.31% 提高到 49.22%，但 P95 从 1252 ms 增加到
  2799 ms；按需重排调用率 18.12%，但没有同时取得最优质量和延迟。
- 所以这个模块已经实现了可配、可解释和可消融的自适应控制，但当前规则选择器
  还不是最终质量最优策略。

## 评测指标速查

| 指标 | 回答的问题 | 计算口径 | 阶段 |
|---|---|---|---|
| Router Accuracy | 路由意图是否正确 | 预测 intent 正确数 / 有 intent 标签的 Case 数 | Router |
| Recall@k | 前 k 个结果是否召回标注证据 | 每 Case 的 relevant IDs 命中比例再取平均 | Retrieval |
| MRR | 第一条相关证据是否靠前 | 第一条 relevant 结果排名倒数的平均值 | Retrieval |
| NDCG@5(折扣累计增益) | 多条相关证据的前排排序质量如何 | DCG@5 / 理想 DCG@5 | Retrieval / Rerank |
| Citation Validity | 引用 ID 是否真实且允许使用 | 合法引用数 / 模型返回引用数 | Validation |
| Abstention Accuracy | 可答和不可答时是否做对“回答/拒答”决策 | 决策与人工 answerable 标签一致的 Case 数 / 全部 Case 数 | Pipeline |
| Key-Point Coverage | 回答覆盖了多少期望要点 | covered points / expected points；语义版保留 unknown | Answer Audit |
| Claim-Level Grounding | 每条事实主张是否被引用证据支持 | 对 claim 输出 supported / unsupported / unknown 和证据片段 | Grounding Audit |
| End-to-End Success | 路由、证据、回答、引用和拒答是否整体满足规则 | 满足协议全部条件的 Case 数 / 全部 Case 数 | Evaluation Harness |
| P50 / P95 | 典型和尾部延迟如何 | 延迟分布的 50/95 分位数 | 各阶段与总链路 |

Abstention 与重试不是一回事：重试是得到最终决策前的内部动作；Abstention Accuracy 只比较最终
`answered/insufficient` 与人工标注的 `answerable=true/false` 是否一致。

## 关键取舍

1. **Graph + Vector 不是免费提升**：frozen 检索质量显著提高，但 CPU P95 增加约 1.19 秒；
   因此保留 Adaptive 选路，而不是所有 Query 强制走图。
2. **Reranker 的结论依赖候选和触发策略**：v0.3 同集控制实验中，Always 提高前排
   排序但显著增加 P95，按需策略只得到小幅 MRR 改善并伴随 Recall@5 退化；不宣称
   存在同时最优质量和延迟的策略。
3. **当前端到端仍有明确缺口**：v0.3 frozen E2E 为 8 answered / 69 insufficient / 3 error，
   主要是旧 Router/Evidence Gate 对新关系问题过度拒答；不能把检索提升夸大成答案准确率提升。

## 最短阅读顺序

1. 本文四条主链和模块表。
2. [系统链路解释](system_flows_explained_zh.md)：用一次请求和一次评测理解数据如何流动。
3. [最终实验报告](../evaluation/final_experiment_report.md)：核对指标、负结果和证据路径。
4. [口头介绍](../development/interview_guide_zh.md)：练习 30 秒和 5 分钟版本。
