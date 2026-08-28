# EvalRAG 架构图

本文只展示当前 P1 架构和关键边界。模块职责、算法取舍与实测结果见
[项目地图](project_map_zh.md) 和 [最终实验报告](../evaluation/final_experiment_report.md)。

## 1. 离线知识构建

```mermaid
flowchart LR
    RAW["公开资料 / 脱敏自有资料"] --> IMP["Importer / Ingestion"]
    IMP --> DOC["统一 Document"]
    DOC --> CHUNK["标题、段落、句子边界感知 Chunking"]
    CHUNK --> QA["SHA-256 / SimHash 去重<br/>provenance 审核"]
    QA --> SNAP["Manifest + Versioned Chunks + Stats"]
    SNAP --> BM25["BM25 统计"]
    SNAP --> DENSE["BGE Dense Index"]
    SNAP --> GRAPH["Job-Skill-Experience Graph"]
    DENSE --> FILE["文件精确索引"]
    DENSE --> PGV["pgvector exact / HNSW"]
    GRAPH --> GFILE["版本化 Graph JSON"]
    GRAPH --> NEO["Neo4j"]
```

- `Manifest` 记录每份材料的来源、revision、许可、采集时间、公开/脱敏和审核状态。
- `Versioned Chunks` 固定索引与评测共同消费的证据快照，避免数据变化后实验无法复现。
- `Stats` 记录数量、长度分布、空文档、重复 hash、模板行和来源占比等质量信息。

## 2. 在线回答链路

```mermaid
flowchart TD
    REQ["RagRequest<br/>query / user_id / session_id / config"] --> RUNTIME["AgentRuntime<br/>root Run + checkpoint"]
    RUNTIME --> ROUTER["Feedback Hybrid Router<br/>Rule + Semantic + approved anchors"]
    ROUTER -->|RouteDecision| ANALYZER["Query Analyzer<br/>QueryFeatures -> EvidenceRequirement"]
    ANALYZER --> SELECT["Adaptive Retriever<br/>BM25 / Dense / RRF / Graph+Vector"]
    SELECT --> CONF["候选置信度"]
    CONF -->|低置信且策略允许| RERANK["CrossEncoder Rerank<br/>最多一次"]
    CONF -->|无需重排| RESULTS["ranked RetrievalResult"]
    RERANK --> RESULTS
    RESULTS --> GATE["Evidence Gate<br/>route / count / score / source coverage"]
    GATE -->|retryable, max once| BROADEN["去掉 source filter<br/>扩源检索一次"]
    BROADEN --> SELECT
    GATE -->|insufficient| ABSTAIN["RagResponse<br/>insufficient_evidence"]

    REQ --> MEMORY["SessionMemoryService"]
    MEMORY --> STORE["History: Redis -> PostgreSQL<br/>Profile/Summary: PostgreSQL<br/>Memory: pgvector"]
    STORE --> SIGNAL["ContextSignalExtractor<br/>token pressure / follow-up / memory score"]
    SIGNAL --> POLICY["ContextPolicy -> ContextPlan"]
    GATE -->|sufficient| ENGINE["ContextEngine<br/>预算、优先级、跨层去重"]
    POLICY --> ENGINE
    ENGINE -->|ManagedContext| GEN["Generator<br/>结构化 Prompt / JSON contract"]
    GEN --> GATEWAY["Model Gateway<br/>timeout / retry / circuit breaker / fallback"]
    GATEWAY -->|非法 JSON, max once| REPAIR["格式修复生成一次"]
    REPAIR --> GATEWAY
    GATEWAY --> PARSED["GenerationResult"]
    PARSED --> VALID["Citation Validator<br/>存在 / 重复 / 状态组合"]
    VALID -->|valid| RESP["RagResponse<br/>answer + citations + trace_id"]
    VALID -->|invalid| ERROR["受控 error"]

    RUNTIME -.-> TRACE["Run / Span AgentTrace"]
    ROUTER -.-> TRACE
    ANALYZER -.-> TRACE
    SELECT -.-> TRACE
    RERANK -.-> TRACE
    GATE -.-> TRACE
    ENGINE -.-> TRACE
    GATEWAY -.-> TRACE
    VALID -.-> TRACE
```

主链数据结构：

```text
RagRequest
-> RouteDecision
-> QueryFeatures -> EvidenceRequirement
-> list[RetrievalResult] + RetrievalDecision
-> EvidenceDecision
-> ContextPlan -> ManagedContext
-> GenerationResult
-> ValidationResult
-> RagResponse + AgentTrace(Run/Spans)
```

Evidence Gate 判断“当前证据是否值得调用模型”；Citation Validator 判断“模型返回的引用
结构是否合法”；Claim-Level Grounding 在离线评测中判断“回答中的事实是否被引用证据支持”。

## 3. 离线评测、失败与回归

```mermaid
flowchart TD
    CORPUS["Corpus v0.3<br/>Manifest + Chunks + Graph"] --> RUNNER["Evaluation Runner"]
    CASE["EvaluationCase<br/>query / expected sources / relevant ids / points"] --> RUNNER
    CONFIG["Versioned RunConfig<br/>dataset / split / strategy / model"] --> RUNNER
    RUNNER --> PRED["系统真实 Predictions + Traces"]
    PRED --> RETM["Retrieval / Routing Metrics<br/>Recall@k / MRR / NDCG / latency"]
    PRED --> RELM["Answer Reliability<br/>Citation / Abstention"]
    PRED --> AUDIT["Semantic Key-Point<br/>Claim-Level Grounding"]
    PRED --> CTX["Context Benchmark<br/>follow-up / tokens / redundancy"]
    RETM --> ART["case_results + failures + summary + report"]
    RELM --> ART
    AUDIT --> ART
    CTX --> ART
    ART --> ROOT["Trace 定位 root cause"]
    ROOT --> REG["RegressionCase<br/>open / fixed"]
    REG --> CI["Executable Regression + CI Evaluation Gate"]
    ART --> FREEZE["配置冻结后运行 frozen test"]
```

- `dev` 用于比较策略和修复失败；`frozen test` 只在配置固定后运行，不用于继续调参。
- `failure` 是系统输出未满足标签或可靠性协议的 Case，不只是脚本异常。
- `open regression` 跟踪尚未修复的问题；`fixed regression` 在 CI 中防止旧问题复发。

## 4. 服务、任务与持久化

```mermaid
flowchart LR
    CLIENT["HTTP Client"] --> API["FastAPI"]
    API -->|POST /v1/query| RUNTIME["AgentRuntime / RagPipeline"]
    RUNTIME --> RESPONSE["RagResponse + trace_id"]
    RUNTIME --> PG["PostgreSQL<br/>request / Run / Trace / Job / Profile"]
    API -->|GET /v1/traces/id| PG

    API -->|POST /v1/evaluation-jobs| PG
    API -->|job_id| REDIS["Redis Queue"]
    REDIS --> WORKER["Evaluation Worker"]
    WORKER --> ERUN["Evaluation Runner"]
    ERUN --> FILES["Persistent Volume<br/>reports / cases / failures"]
    WORKER -->|running / succeeded / failed| PG

    API --> CACHE["Redis Session Cache"]
    CACHE -->|miss / error| PG
    RUNTIME --> PGV["pgvector Memory / Dense Index"]
    RUNTIME --> NEO["Neo4j Graph"]
```

PostgreSQL 是任务状态和运行元数据的真相来源；Redis 只承担短期队列与最近会话缓存；
pgvector/Neo4j 保存持久化检索结构；完整报告和逐 Case 工件保存在文件卷。Docker Compose
编排 API、Worker、PostgreSQL、Redis 和 Neo4j，GitHub Actions 执行服务链与持久化验证。

## 5. 当前配置取舍

| 决策点 | 当前实现与证据边界 |
|---|---|
| Router | Rule/Semantic/Hybrid/Feedback 均可配置；反馈版在 v0.2 dev 提升，但在 v0.3 Source Exact 未超过 Rule，不能宣称跨分布稳定提升。 |
| Query Analyzer | 使用四个核心证据信号生成 `EvidenceRequirement`，再映射 Retriever；当前是可解释规则版，不是学习得到的 Policy。 |
| Retrieval | Graph+Vector 在 v0.3 frozen 相比 BM25 提升 Recall@5/MRR，但 CPU P95 明显增加；Adaptive selector 在 dev 的覆盖略升、MRR 下降。 |
| Reranker | 支持 Never/Always/On-demand CrossEncoder；Always 改善部分 MRR 但延迟高，On-demand 尚未取得质量/延迟 Pareto 最优。 |
| Evidence | 只允许一次扩源重试，区分弱证据、正常不可回答和模型格式错误，避免无限循环。 |
| Context | `ContextPolicy -> ContextPlan -> ContextEngine` 动态选择分层记忆并执行统一预算；当前 60 组/300 turns 结果属于确定性 Context benchmark。 |
| Generation | JSON contract、temperature=0；Model Gateway 提供有界重试、熔断和 Provider fallback，鉴权错误不盲重试。 |
| Grounding | claim-level verdict 为 supported/unsupported/unknown；Judge unavailable 不记为 supported。 |

所有数字及其 dataset、split、配置和限制统一以
[最终实验报告](../evaluation/final_experiment_report.md) 为准。
