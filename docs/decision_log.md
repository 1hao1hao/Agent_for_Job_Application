# Decision Log

## 2026-06-13: Day 1 项目骨架先行

决定：

- 第一阶段先建立项目骨架和文档，不实现业务代码。
- 后续每个任务都按“任务、背景、目标、输入、输出、要求、不要做”的模板推进。
- 每个新增或重构模块都必须能说明模块输入、输出、核心函数、失败排查方式和面试讲法。

原因：

- 项目目标是可评测多源 RAG/Agent 系统，不是普通聊天机器人。
- 先固定边界和验收方式，可以避免过早引入框架或写出难以解释的 demo。
- 面试项目需要能被追问，因此模块设计要从一开始就保留可讲述性和可排查性。

影响：

- `docs/task_board.md` 会作为每周任务和完成状态的主入口。
- `docs/coding_rules.md` 会约束每次实现后的总结格式。
- Day 2 开始才进入业务代码，实现范围从 ingestion 和 Chunk schema 开始。

## 2026-06-13: Day 2 使用原生 Python 实现 ingestion

决定：

- 用 `dataclass` 定义第一版 `Chunk`，暂不引入 Pydantic。
- 只支持本地 `.md` 和 `.txt` 文件。
- source type 从 `data/raw/` 下的一级目录推断。
- 空文件返回空 chunk 列表，而不是抛异常。
- chunk id 使用 source type、文件路径、chunk index 和文本内容生成确定性哈希。

原因：

- 第一版目标是跑通从多源文本到统一 chunk 的最小闭环。
- 使用标准库能降低环境成本，也方便面试时解释实现细节。
- 确定性 chunk id 有利于后续 retrieval、citation 和 regression test 对齐。

影响：

- 后续 retrieval 模块可以直接消费 `list[Chunk]`。
- 如果未来引入 PDF、HTML 或数据库，需要扩展 ingestion 的读取层，但不应改变 `Chunk` 的核心字段。
- 当前 chunk 切分是基础字符预算加段落边界，不代表最终检索效果最优。

## 2026-06-16: Day 3 使用关键词重叠实现基础检索

决定：

- 第一版 retrieval 使用标准库实现关键词检索，不引入 BM25 或向量库。
- 英文和数字按连续词提取，中文按单字和相邻双字提取。
- 检索分数使用 query token 命中比例。
- 支持 `source_types` 过滤，便于后续 router 控制检索范围。

原因：

- 当前目标是跑通可评测检索接口，而不是追求最终召回效果。
- 简单关键词检索容易解释，也方便用单元测试验证排序、top-k 和 source filter。
- 中文实习求职数据没有外部分词依赖时，单字和双字 token 能覆盖一部分短查询。

影响：

- 后续可以在不改变调用方接口的前提下，把内部实现替换为 BM25、TF-IDF、embedding 或 hybrid search。
- 当前分数不是语义相似度，只能作为第一版 baseline。

## 2026-06-18: Day 4 使用规则实现 Intent Router

决定：

- 第一版 router 使用关键词规则，不调用 LLM。
- intent 包含 `analyze_jd`、`match_resume`、`interview_prepare`、`project_explain`、`application_plan` 和 `unknown`。
- 每个 intent 映射到固定的 routed sources。
- 路由结果保留 matched keywords，便于解释和后续 trace。

原因：

- 当前目标是先跑通可解释、可测试的路由闭环。
- 规则版 router 容易写单元测试，也方便用 Router Accuracy 做第一版评测。
- 先把接口固定下来，后续可以替换为模型分类或 LLM router。

影响：

- Agent 后续可以根据 `RouteDecision.routed_sources` 控制 retrieval 的 `source_types`。
- 当前 router 对同义表达覆盖有限，需要后续通过失败样例持续补充关键词或升级分类器。

## 2026-06-18: Day 5 使用 JSONL 记录 Agent Trace

决定：

- 定义 `AgentTrace` 数据结构，记录 query、intent、routed sources、retrieved chunks、latency 和 error type。
- trace 使用 JSONL 格式追加写入本地文件。
- tracing 模块只负责组装、写入和读取 trace，不负责执行 agent 流程。
- 未知 error type 在读取时归一化为 `unknown_error`。

原因：

- JSONL 适合持续追加请求日志，也方便后续 evaluation 逐行读取。
- 将 tracing 与 agent 编排解耦，可以先建立可观测数据结构，再实现回答生成。
- 保留 error type 有利于把 retrieval miss、router error 等失败样例沉淀为 regression tests。

影响：

- 后续 agent 单轮流程可以在 routing 和 retrieval 后调用 `build_agent_trace`。
- evaluation 模块可以直接读取 JSONL trace 统计错误类型、latency 和检索结果。

## 2026-07-26: Day 6 使用抽取式回答生成 citations

决定：

- 第一版 answer 模块不调用 LLM，只基于 top chunks 组织回答。
- 每个被使用的 retrieval result 都生成一条 `Citation`。
- 检索结果为空，或参数限制无效时，返回明确的证据不足说明。
- `Citation` 至少保留 chunk id、source path、source type、title、rank 和 score。

原因：

- 当前目标是先保证回答不脱离检索证据，而不是追求自然语言生成质量。
- 抽取式回答更容易测试，也更适合作为 citation accuracy 的第一版基础。
- 先固定 `AnswerResult` 接口，后续可以替换为 LLM answer composer。

影响：

- 单轮流程已经具备 `routing -> retrieval -> answer + citations` 的核心链路。
- 后续 trace 可以把 `AnswerResult.answer` 和 citations 写入 `AgentTrace`。
- 当前回答更像证据摘要，不是真正的模型推理回答。

## 2026-07-26: Day 7 建立基础 Evaluation 与 Regression Seed

决定：

- 第一版 evaluation 只实现 Recall@k 和 Router Accuracy。
- 评测输入使用人工标注 expected 与系统 predicted 的显式样例，不自动伪造结果。
- regression seed 使用 JSONL 格式，每行记录一个失败样例和失败类型。
- 新增 `docs/evaluation_explained_zh.md` 说明当前指标、限制和下一步计划。

原因：

- 当前项目的核心差异点之一是可评测，必须尽早把指标接口固定下来。
- Recall@k 能检查检索是否召回正确证据，Router Accuracy 能检查意图和 source 路由是否正确。
- 失败样例先以 seed 形式沉淀，后续再逐步转成可执行 regression tests。

影响：

- 第一周已经跑通 `ingestion -> routing -> retrieval -> answer/citations -> tracing -> evaluation` 的最小工程闭环。
- 后续每次修改 retrieval、router 或 answer，都可以用同一批评测样例观察是否退化。
- 当前评测样例很小，只能证明评测链路能运行，不代表真实业务效果。

## 2026-07-26: Week 1 总结

已完成：

- 项目文档、架构说明、编码规则和任务板。
- 中文 raw data 样本和岗位时效性 metadata。
- `ingestion`：本地文档读取、front matter 解析、chunk 切分、metadata 继承。
- `job_importer`：JSON/CSV 岗位导入到 `JobPosting` 和 `Document`。
- `retrieval`：关键词重叠 baseline，支持 source type 过滤和 top-k。
- `routing`：规则版 intent router，输出 routed sources。
- `tracing`：AgentTrace 和 JSONL 写入/读取。
- `agent`：抽取式 answer 和 citations。
- `evaluation`：Recall@k、Router Accuracy 和 regression seed 格式。

当前限制：

- 检索仍是关键词 baseline，没有 BM25、embedding、hybrid search 或 rerank。
- Router 是规则版，对同义表达和多意图 query 支持有限。
- Answer 是抽取式，不调用 LLM，也不做复杂归纳推理。
- Citation 只记录来源，还没有自动判断引用是否真的支持回答。
- Evaluation 样例很少，不能代表真实效果。

下一步：

- 扩充真实中文求职 query 和人工标注样例。
- 将 trace 自动转成 evaluation 输入。
- 把 regression seed 转成可执行回归测试。
- 再考虑 BM25、rerank、LLM answer composer 和 citation accuracy。
