# Task Board

## Week 1 Goal

跑通一个最小可评测 RAG 闭环：从本地多源文本到 chunk，再到检索、引用、trace 和基础评测。第一周不追求效果最好，只追求结构清晰、能运行、能测试、能解释。

## 每个任务的确认模板

后续每次实现或重构模块时，都要能回答：

- 新增或修改了哪个模块。
- 模块输入是什么。
- 模块输出是什么。
- 核心函数有哪些。
- 失败时怎么排查。
- 面试官如果问这个模块，应该怎么讲。

完成任务时必须同步说明：

- 修改了哪些文件。
- 如何运行。
- 如何测试。
- 当前限制。
- 下一步建议。

## Day 1: 项目骨架与文档

状态：done

任务：

- [x] 确定项目目录结构。
- [x] 完成 `docs/project_brief.md`。
- [x] 完成 `docs/architecture.md` 初版。
- [x] 完成 `docs/coding_rules.md`。
- [x] 完成 `docs/task_board.md`。
- [x] 创建空目录，不实现业务代码。

验收标准：

- [x] 文档能说明项目目标、范围、架构和第一周计划。
- [x] 目录结构能支撑后续模块实现。

模块确认：

- 新增模块：无业务模块；本日只建立项目骨架和文档系统。
- 输入：项目目标、技术原则、第一周计划、后续任务提示词模板。
- 输出：`docs/` 下的项目说明、架构说明、编码规则、任务板，以及后续模块目录。
- 核心函数：无；Day 1 不实现业务代码。
- 失败排查：检查文档是否为空；检查目录是否缺失；检查是否误新增业务代码。
- 面试讲法：Day 1 的价值是先把项目边界、模块职责、评测目标和开发节奏固定下来，避免直接写聊天机器人式 demo。这个项目从第一天就围绕可评测、可观测、可复盘来组织。

## Day 2: 数据样本与 Chunk Schema

状态：done

任务：

- [x] 准备最小样本文本：1 份 JD、1 份简历、1 份面经、1 份项目日志、1 份用户画像。
- [x] 定义 `Chunk` 数据结构。
- [x] 实现本地 `.md` 或 `.txt` 文件读取。
- [x] 实现基础 chunk 切分。
- [x] 为 ingestion 写最小单元测试。

验收标准：

- [x] 可以从 `data/raw/` 生成统一 chunk 列表。
- [x] 每个 chunk 包含 source type、source path、text 和 metadata。
- [x] 测试覆盖正常读取和空文件场景。

模块确认：

- 新增模块：`intern_rag.ingestion`。
- 输入：`data/raw/` 下按 source type 分类的 `.md` 或 `.txt` 文件。
- 输出：统一的 `Chunk` 列表，每个 chunk 包含 `id`、`source_type`、`source_path`、`title`、`text` 和 `metadata`。
- 核心函数：`read_text_file`、`split_text`、`infer_source_type`、`build_chunks_from_file`、`load_chunks_from_raw_dir`。
- 失败排查：先确认文件后缀是否为 `.md` 或 `.txt`；再确认文件是否位于合法 source type 目录下；空文件会返回空 chunk 列表；切分异常时检查 `max_chars` 是否大于 0。
- 面试讲法：这个模块把多源原始材料统一成后续 retrieval 可以消费的最小证据单元。第一版刻意不用复杂 parser 或 embedding，只保证 source metadata 完整、chunk id 可复现、空文件和非法输入有明确行为。

## Day 3: 基础检索

状态：done

任务：

- [x] 实现最小关键词检索或轻量 TF-IDF 检索。
- [x] 支持按 source type 过滤。
- [x] 返回 top k retrieval results。
- [x] 为 retrieval 写最小单元测试。

验收标准：

- [x] 给定 query 可以返回排序后的 chunks。
- [x] 检索结果包含 score、rank、chunk id。
- [x] 测试覆盖 source filter 和 top k。

模块确认：

- 新增模块：`intern_rag.retrieval`。
- 输入：用户 query、`list[Chunk]`、`top_k`、可选 `source_types`。
- 输出：按 score 排序的 `list[RetrievalResult]`，每条结果包含 `chunk_id`、`score`、`rank`、`chunk` 和 `reason`。
- 核心函数：`tokenize_text`、`score_chunk`、`retrieve_top_k`。
- 失败排查：先确认 query 是否为空；再确认 chunks 是否有内容；如果按 source type 过滤无结果，检查 `chunk.source_type` 是否匹配；如果中文召回异常，检查 token 是否被 `tokenize_text` 正确生成。
- 面试讲法：第一版 retrieval 不是追求最优效果，而是建立可评测的检索接口。它用关键词重叠率给 chunk 打分，支持按知识源过滤，并返回 rank 和 score，为后续 Recall@k、rerank 和 citation 打基础。

## Day 4: Intent Router

状态：done

任务：

- [x] 定义第一版 intent 列表。
- [x] 实现规则版 router。
- [x] 根据 intent 输出 routed sources。
- [x] 为 routing 写最小单元测试。

验收标准：

- [x] 常见 query 能路由到合理知识源。
- [x] 无法识别时返回 `unknown`。
- [x] 测试覆盖 JD 分析、简历匹配、面试准备、项目讲解和未知输入。

模块确认：

- 新增模块：`intern_rag.routing`。
- 输入：用户 query。
- 输出：`RouteDecision`，包含 `intent`、`routed_sources` 和 `matched_keywords`。
- 核心函数：`route_query`。
- 失败排查：先看 query 是否为空；再看关键词是否覆盖当前表达；若路由不符合预期，检查 `INTENT_KEYWORDS` 和 `INTENT_TO_SOURCES` 是否需要补充。
- 面试讲法：第一版 router 用规则而不是 LLM，是为了让意图识别可解释、可测试。它把用户问题映射到 intent，再输出应该优先检索的知识源，为后续 agent 编排和 Router Accuracy 评测打基础。

## Day 5: Agent Trace

状态：done

任务：

- [x] 定义 `AgentTrace` 结构。
- [x] 实现 JSONL trace writer。
- [x] 在单轮流程中记录 intent、retrieved chunks、latency 和 error type。
- [x] 为 tracing 写最小单元测试。

验收标准：

- [x] 每次请求能写出一条完整 trace。
- [x] trace 可被重新读取。
- [x] 测试覆盖正常写入和错误类型记录。

模块确认：

- 新增模块：`intern_rag.tracing`。
- 输入：query、`RouteDecision`、`list[RetrievalResult]`、latency 信息和 error type。
- 输出：`AgentTrace`，以及可追加保存的 JSONL trace 文件。
- 核心函数：`build_agent_trace`、`write_trace_jsonl`、`read_traces_jsonl`、`retrieval_result_to_trace`。
- 失败排查：先确认 trace 文件路径是否可写；再检查 `error_type` 是否在支持列表中；如果 retrieved chunks 缺字段，检查传入的 `RetrievalResult` 是否完整。
- 面试讲法：trace 模块让系统每一步可观测。它不负责做路由或检索，只把单轮请求中的 intent、路由来源、检索结果、耗时和错误类型结构化记录下来，方便后续复盘失败样例和计算评测指标。

## Day 6: Answer + Citations

状态：done

任务：

- [x] 实现第一版答案组织逻辑。
- [x] 从 top chunks 中生成简短回答。
- [x] 输出 citation 列表。
- [x] 为 citation 结构写最小测试。

验收标准：

- [x] 回答不会脱离检索到的 chunks。
- [x] citation 能指向 chunk id 和 source path。
- [x] 当证据不足时返回明确的不确定性说明。

模块确认：

- 新增模块：`intern_rag.agent`。
- 输入：用户 query、按相关性排序的 `list[RetrievalResult]`、可选 `max_chunks` 和 `snippet_chars`。
- 输出：`AnswerResult`，包含 `answer`、`citations`、`used_chunk_ids` 和 `is_evidence_sufficient`。
- 核心函数：`compose_answer`、`citation_from_result`、`format_evidence_snippet`。
- 失败排查：先确认 retrieval 是否返回结果；再检查 `max_chunks` 和 `snippet_chars` 是否大于 0；若 citation 缺来源，检查 `RetrievalResult.chunk` 是否包含 `source_path`、`source_type` 和 `title`。
- 面试讲法：第一版 answer 不调用 LLM，而是把 top chunks 的原文片段组织成回答，并为每个片段生成 citation。这样可以先保证回答不脱离证据，后续再替换为 LLM 生成或更复杂的 citation 校验。

## Day 7: Evaluation 与 Regression Seed

状态：done

任务：

- [x] 准备最小评测样例。
- [x] 实现 Recall@k 的基础计算。
- [x] 实现 Router Accuracy 的基础计算。
- [x] 建立 `tests/regression/` 的失败样例格式。
- [x] 写第一周总结到 `docs/decision_log.md`。

验收标准：

- [x] 可以运行一个基础 evaluation。
- [x] 至少有 2 到 3 条 regression seed。
- [x] 文档说明当前指标、限制和下一步计划。

模块确认：

- 新增模块：`intern_rag.evaluation`。
- 输入：人工标注的 relevant chunk ids、系统 retrieved chunk ids、expected/predicted intent 和 sources。
- 输出：`EvaluationReport`，包含 `recall_at_k`、`router_accuracy`、样例数量和 k。
- 核心函数：`calculate_recall_at_k`、`calculate_average_recall_at_k`、`calculate_router_accuracy`、`evaluate_cases`、`load_evaluation_cases`。
- 失败排查：先确认评测样例是否包含 expected 和 predicted 字段；再检查 relevant chunk ids 是否和 retrieval 输出 chunk ids 对齐；若 Router Accuracy 异常，检查 sources 是否采用集合口径比较。
- 面试讲法：evaluation 模块把主观“感觉效果不错”变成可计算指标。第一版只做 Recall@k 和 Router Accuracy，用人工标注样例和系统输出对比，为后续失败驱动优化和 regression tests 打基础。
