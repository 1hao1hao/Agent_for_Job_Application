# EvalRAG Project Brief

## 项目名称

**EvalRAG：可观测、可评测的多源知识 RAG Agent Harness**

仓库名和 Python 包名暂时保持不变，避免项目重命名干扰当前开发。对外名称使用 EvalRAG。

## 项目重新定位

求职资料是 EvalRAG 当前的中文演示领域和评测语料，不是项目能力边界。

EvalRAG 解决的问题是：

> 传统 RAG 系统通常只能展示最终回答，难以定位路由、召回、上下文、生成和引用中的具体失败；检索与 Prompt 修改缺少固定评测集，旧问题也容易在迭代中复发。EvalRAG 因此把多源路由、混合检索、证据生成、运行 Trace、离线评测和 Regression 组织成一个可复现的质量闭环。

项目不以“接入了多少框架”为亮点，而以以下能力为主线：

1. **可观测**：一次请求的路由、召回、上下文、模型输出、引用、重试和耗时可以回放。
2. **可评测**：固定数据集和运行配置能够产出 Router、Retrieval、Grounding 与 End-to-End 指标。
3. **可回归**：人工确认的失败样例能够转成可执行测试。
4. **可拒答**：证据不足、引用非法或模型输出异常时不强行回答。
5. **可优化**：通过同一评测集比较 Keyword、Dense 与 Hybrid，并用失败 Trace 解释指标变化。

## 为什么称为 Agent Harness

Harness 指围绕模型和 RAG 核心能力提供运行控制与质量保障的外壳。EvalRAG 的 Harness 负责：

- 统一请求、响应和模块契约。
- 编排 Router、Retriever、Generator 与 Validator。
- 控制证据门控、有限重试和拒答状态。
- 保存 Trace、运行配置和评测工件。
- 对修改前后的系统执行同集评测与回归。

这里的 Agent 指“根据状态决定生成、重试或拒答”的轻量工作流，不代表多智能体协作。P0 不做多 Agent。

## 当前实现状态

截至 2026-08-07，P0 已完成：

- 五类中文语料的 Document/Chunk Schema、岗位时效性 metadata 和 JSON/CSV importer。
- 100 文档、310 Chunk、120 Query 的 `evalrag_v0.2`，包含 80 dev / 40 frozen test。
- Rule/Semantic/Hybrid Router 与 Keyword/Dense/RRF Hybrid Retriever。
- Evidence Gate、预算感知 Context、结构化 LLM Generation、Citation Validator 和
  source/format 单次重试。
- 一次请求一条 JSONL Agent Trace，记录多次 attempt、阶段延迟与 token。
- Router、Retrieval、Citation、Abstention、Semantic Key-Point、Claim-Level
  Grounding 和 Regression 评测。
- 真实 DeepSeek dev/test Run、frozen retrieval 对照和 144 个自动化测试。
- 单来源、多来源、拒答固定 Demo，以及 README、架构和最终实验报告。
- 标准 BM25、FastAPI HTTP 契约、PostgreSQL 状态持久化、Redis Evaluation Worker
  和通过 GitHub Actions 验证的四服务 Docker Compose 部署。

当前真实限制：

- v0.2 是项目自建、审核的半真实 benchmark，不代表线上业务分布。
- 最终 Hybrid 提高当前 test 的 Recall/MRR，但检索 P95 从 Keyword 的 99.55 ms
  增至 802.50 ms。
- required-source Evidence Gate 造成 6 条可回答 Case 的 retrieval miss。
- Claim-Level Grounding 与 Generator 使用同模型家族；frozen 有 3 条 unknown，
  因此严格 E2E 为 unavailable，不能宣称零幻觉。
- Reranker 在 dev 负向消融后关闭；没有实测神经 CrossEncoder 提升。
- BM25 在 dev 消融中未优于既有最终 Hybrid，因此保留为可选检索配置，而不是
  替换冻结配置；可视化、高并发压测和 pgvector 继续暂缓。

## 项目文档入口

1. `docs/overview/project_map_zh.md`：系统主链和核心数据结构。
2. `docs/overview/system_flows_explained_zh.md`：在线、离线和服务化数据流。
3. `docs/overview/architecture.md`：模块职责、失败分支和设计取舍。
4. `docs/evaluation/evaluation_protocol.md`：数据划分、指标公式和实验约束。
5. `docs/evaluation/final_experiment_report.md`：正式结果和失败分析。

## P0 最终形态

P0 是第一个可复现发布版本，目标链路为：

```text
Query
  -> Source Router
  -> Retriever: Keyword / Dense / Hybrid
  -> Evidence Gate
       -> sufficient
       -> broaden sources and retry once
       -> abstain
  -> Context Builder
  -> LLM Structured Generation
  -> Citation Validator
  -> RagResponse
  -> Agent Trace
  -> Offline Evaluation
  -> Regression
```

### P0 功能交付

- 正常单来源回答。
- 多来源融合回答。
- 引用只能指向本轮上下文中的 chunk。
- 非法 JSON 和非法引用有受控错误。
- 证据不足时返回 `insufficient_evidence`。
- source 扩展和模型格式修复都最多重试一次。
- 一次 query 对应一条完整 Trace。
- Keyword、Dense、Hybrid 三种检索策略可配置切换。
- 一个命令运行离线评测并生成报告。
- 一个稳定 CLI 运行入口；P1 在相同 Pipeline 上增加 FastAPI 服务，不复制核心逻辑。

### P0 数据与评测交付

最低发布规模与当前进度：

| 项目 | 最低要求 | 说明 |
|---|---:|---|
| 文档 | 100 份，已达到 | 70 份为透明标记的项目自建合成 benchmark |
| Chunks | 310，已达到 | 使用 420 字符合理切分，自然产生 |
| 评测 Query | 120 条，已达到 | 80 dev / 40 frozen test，覆盖四类 Query |
| Query 类型 | 4 类 | 单来源、多来源、语义改写、不可回答 |
| Retriever | 3 组 | Keyword、Dense、Hybrid RRF |
| Router | 3 组 | Rule、Semantic、Hybrid |
| 固定 Demo | 3 个 | 单来源、多来源、拒答 |
| 闭环案例 | 至少 1 个 | 失败定位、修改、重评、Regression |

以上数据规模由 `scripts/prepare_evaluation_data.py` 实际统计。它们证明评测链路
规模，不代表线上真实业务流量；frozen test 已在配置冻结后一次性运行。

## 核心实验问题

P0 不做无目的的功能堆叠，围绕三个可回答的实验问题推进。

### RQ1：Hybrid Retrieval 是否优于 Keyword Baseline

固定相同语料、Query、top-k 和评测代码，对比：

- Keyword。
- Dense。
- Hybrid RRF。

主要指标：

- Recall@3。
- Recall@5。
- MRR。
- P50/P95 Retrieval Latency。

重点分析：

- `C++`、`LoRA`、`RAG` 等精确术语。
- 同义表达和语义改写。
- 长问题和跨来源问题。

### RQ2：证据门控能否减少无依据回答

对比：

- 无 Evidence Gate 的直接生成。
- Citation Validation + Evidence Sufficiency + Abstention。

主要指标：

- Citation Validity。
- Abstention Accuracy。
- Unsupported Answer Rate。
- End-to-End Success Rate。
- P50/P95 End-to-End Latency。

### RQ3：Regression 是否能防止旧问题复发

至少完整展示一次：

```text
评测失败
  -> 通过 trace 定位 failure type
  -> 修改 Router / Retriever / Prompt
  -> dev 集验证
  -> 冻结 test 集复测
  -> 固化为 Regression Case
```

主要指标：

- Regression Pass Rate。
- 修复前后目标指标变化。
- 是否出现其他 case 退化。

## 指标体系

| 层级 | 指标 | 项目中的含义 |
|---|---|---|
| Routing | Router Accuracy | intent 与 source 选择是否符合人工标签 |
| Retrieval | Recall@3、Recall@5、MRR | 正确证据是否被召回以及排名是否靠前 |
| Grounding | Citation Validity | 引用 id 是否来自本轮上下文 |
| Grounding | Key-Point Coverage | 回答覆盖了多少人工标注关键点 |
| Safety | Abstention Accuracy | 不可回答问题是否正确拒答 |
| Safety | Unsupported Answer Rate | 回答是否包含无法由引用支持的结论 |
| End-to-End | Success Rate | 路由、召回、引用和回答状态是否共同达标 |
| Performance | P50/P95 Latency | 检索与端到端延迟分布 |
| Cost | Input/Output Tokens、Estimated Cost | 模型调用成本 |
| Regression | Regression Pass Rate | 已固化失败样例是否仍通过 |

每个指标的固定口径见 `docs/evaluation/evaluation_protocol.md`。

## P0 目标值与真实值

目标用于判断是否继续优化，不能作为已经取得的实测结论发布。

| 指标 | P0 目标 | 实测值 |
|---|---:|---:|
| Hybrid Recall@5 相对 Keyword | 提升至少 5 个百分点 | +7.22 pp（67.22% -> 74.44%） |
| Citation Validity | 100% | 100%（40 frozen test） |
| Abstention Accuracy | 不低于 85% | 100%（10/10 unanswerable） |
| Fixed Regression Pass Rate | 100% | 100%（1/1；4 open 不计分母） |
| Retrieval P95 | 记录并解释 | Hybrid 802.50 ms；Keyword 99.55 ms |
| End-to-End P95 | 记录并解释 | 4136.71 ms |

如果目标没有达到，报告真实结果并继续分析；不得为了满足目标修改标签、删除失败 case 或预填数字。

## 演示领域

P0 使用中文实习求职知识作为评测 profile：

| Source type | 内容 | 典型问题 |
|---|---|---|
| `jd` | 岗位职责、要求、时效性 | 某岗位要求哪些能力 |
| `resume` | 脱敏技能与经历 | 候选人与岗位是否匹配 |
| `interview` | 公开面经和知识笔记 | 如何准备 RAG 面试问题 |
| `project_logs` | 项目架构、实验与复盘 | 如何解释项目设计 |
| `user_profile` | 求职偏好和现实约束 | 如何安排投递计划 |

这些 source type 沿用现有实现。P0 不为改名而破坏已有接口。

## 发布工件

对外发布的每项结论都必须能够链接到以下工件之一：

```text
data/evaluation/
  evalrag_v0.2.jsonl
  corpus_manifest.jsonl
  corpus_stats.json

reports/
  runs/
  ablations/
  comparisons/
  final/
  regression/

examples/fixed_demos/
  single_source.json
  multi_source.json
  abstention.json

docs/
  README.md
  overview/
    project_map_zh.md
    architecture.md
    architecture_diagram.md
    system_flows_explained_zh.md
  evaluation/
    evaluation_protocol.md
    final_experiment_report.md
  guides/
    chunking_explained_zh.md
```

报告至少记录：

- `run_id`。
- Git commit。
- dataset version。
- model、prompt 和 retriever 配置。
- 样例数量。
- 指标结果。
- 失败 case ids。
- 运行环境和时间。

## P1 工程增强

`P1-D1` 一次性完成：

- 标准 BM25 Retriever 与 Keyword/BM25/Dense/RRF dev 消融。
- FastAPI 在线 Query、Trace 查询和 Evaluation Job 接口。
- PostgreSQL 持久化请求、Trace、Job 和 Run 元数据。
- Redis Queue 与独立 Evaluation Worker，异步执行长时间批量评测。
- Docker Compose 编排 API、Worker、PostgreSQL 与 Redis，并验证重启后状态可恢复。

PostgreSQL 与 Redis 有明确分工：PostgreSQL 是任务状态真相来源，Redis 只做队列；
完整报告保存在持久化 volume。当前 310 Chunk 的 Dense 精确扫描尚未证明需要向量
数据库，因此 P1-D1 不引入 pgvector。大规模 ANN/pgvector 必须由 profiling 和同集
Recall/P95 对照驱动。

P1 后续优先级：

1. P1-D2 把请求生命周期、Stage Event、异常捕获和 Trace sink 收口到 Agent Runtime，
   通过故障注入与重构前后行为一致性证明 Harness 解耦。
2. P1-D3 引入多轮 Context Manager、PostgreSQL/Redis 会话状态和摘要压缩，并比较
   full/window/summary 策略的 Prompt token、质量、延迟和成本。
3. Skill Registry 仅在 Runtime 稳定且出现多个真实执行计划后做对照；Multi-Agent、
   无业务工具的 MCP 和心理风险流程不进入 EvalRAG 主线。

后续再考虑 Trace 可视化，不默认增加前端、Kubernetes 或微服务拆分。

## 非目标

当前不做：

- 多智能体协作。
- MCP。
- Kubernetes 和微服务拆分。
- 自训练 embedding 或 reranker。
- 复杂前端。
- 自动投递。
- 非合规爬虫。
- 为使用框架而迁移到 LangGraph、LangChain 或 LlamaIndex。

## 设计演进

项目的主要演进路径是：

> 我先用原生 Python 建立可解释的 Rule Router 与 Keyword Retrieval Baseline，并为路由、召回、引用和耗时设计 Trace。正式 dev 结果暴露 Router Accuracy 和语义召回瓶颈后，我增加 Semantic/Hybrid Router、Dense Retrieval 和 RRF Hybrid，在同一冻结数据集上比较 Router Accuracy、Recall@k、MRR 和延迟。生成阶段不直接相信模型输出，而是通过 Citation Validator 和 Evidence Gate 决定回答、重试或拒答。最后，我把真实失败 Trace 固化为 Regression Case，确保路由、检索和 Prompt 修改不会让旧问题复发。

其中所有数字都来自仓库内可复现报告。
