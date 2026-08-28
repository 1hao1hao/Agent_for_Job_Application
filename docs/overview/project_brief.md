# EvalRAG Project Brief

## 项目定义

**EvalRAG：面向中文求职知识推理的图增强、自适应 RAG Agent Harness。**

求职资料是当前演示领域，不是系统能力边界。EvalRAG 解决传统 RAG 的三个工程问题：最终回答
难以追溯到具体决策和证据、策略修改缺少同集量化对照、已修失败容易在后续迭代中复发。

## 核心能力

1. **图增强自适应检索**：根据问题的词面、语义、多来源和关系证据需求，在 BM25、Dense、
   RRF Hybrid 与 Graph+Vector 间选择，并支持低置信候选的按需 CrossEncoder。
2. **可靠回答控制**：Evidence Gate 在生成前决定生成、扩源一次或拒答；Generator 使用 JSON
   契约；Citation Validator 拦截不存在、重复或状态冲突的引用。
3. **自适应 Context 与分层记忆**：ContextPolicy 生成 ContextPlan，ContextEngine 在统一 token
   budget 下编排 Profile、History、Summary、Semantic Memory 和 RAG Evidence。
4. **可观测 Runtime**：一次请求对应一个 root Run，路由、检索、重排、门控、Context、模型和
   校验作为 Span，保存配置、attempt、latency、token、错误与 checkpoint/replay 信息。
5. **评测回归闭环**：版本化 Corpus、EvaluationCase 和 RunConfig 产生真实 predictions、逐 Case
   failures、指标报告和 executable regression，并由 CI Evaluation Gate 阻止已知退化。
6. **服务与持久化**：FastAPI 提供 Query、Trace、Session 和 Evaluation Job 接口；PostgreSQL、
   Redis、pgvector、Neo4j 与独立 Worker 支持状态、缓存、异步任务和持久化检索。

## 当前主链

```text
RagRequest
-> AgentRuntime
-> Feedback Hybrid Router
-> QueryFeatures -> EvidenceRequirement
-> Adaptive BM25 / Dense / RRF / Graph+Vector
-> optional CrossEncoder
-> Evidence Gate: generate | broaden once | abstain
-> ContextPolicy -> ContextPlan -> ContextEngine
-> Generator / Model Gateway
-> Citation Validator
-> RagResponse + Run/Span Trace
```

离线链路：

```text
Corpus + EvaluationCase + RunConfig
-> system predictions + traces
-> routing / retrieval / reliability / grounding / context metrics
-> failures + report
-> open/fixed RegressionCase
-> CI Gate / frozen release
```

## 数据与实验状态

- 当前主版本 `evalrag_v0.3`：658 份去重文档、4208 个 Chunk、240 条 Query（160 dev / 80
  frozen test），覆盖五类 source 和 8 类检索/关系场景。
- Graph+Vector 在 80 条 frozen test 上相对 BM25：Recall@5 `46.67% -> 63.33%`，MRR
  `35.19% -> 57.58%`；P95 `15.50 ms -> 1209.40 ms`。
- Adaptive Context 在 60 组/300 turns dev 上相对 Summary+Recent 保持 100% Follow-up
  Success，并将平均 Prompt Token `75.55 -> 60.68`、History Redundancy `42.86% -> 0`。
- Adaptive selector 与 On-demand Reranker 已完成同集对照，但尚未获得质量/延迟 Pareto 最优；
  负结果和逐 Case 失败保留在报告中。
- v0.3 frozen E2E 暴露过度拒答，不能将检索指标提升表述成端到端答案准确率提升。

## 为什么称为 Harness

Harness 是围绕 Router、Retriever 和 LLM 的运行控制与质量保障层。它统一模块契约、管理状态与
有限重试、保存 Trace 和工件、支持 Replay/Regression，并让 API、CLI、Worker 和 Evaluation
复用同一执行逻辑。这里的 Agent 表示根据状态选择检索、生成、重试或拒答的工作流，不表示多
Agent 协作。

## 证据与边界

- v0.3 标签为 corpus-grounded AI-assisted，不冒充线上分布或独立人工金标准。
- Recall@k/MRR/NDCG 只衡量检索；Citation Validity 只验证引用 ID；LLM semantic/grounding
  grader 也不是人工答案准确率。
- Context 实验是确定性 dev benchmark，尚无 untouched multi-turn frozen test。
- Model Gateway 的 primary 真实 smoke 已运行；备用 Provider 的真实 fallback 受缺少对应 key 限制。
- 项目不包含前端、Kubernetes、微服务拆分、多 Agent 或在线自动学习 Router。

## 文档入口

1. [项目地图](project_map_zh.md)：模块、算法、取舍和效果。
2. [当前架构图](architecture_diagram.md)：四条链路的连接关系。
3. [系统链路解释](system_flows_explained_zh.md)：按数据流快速复习。
4. [详细架构](architecture.md)：契约、边界和失败状态。
5. [最终实验报告](../evaluation/final_experiment_report.md)：真实指标、负结果和工件路径。
