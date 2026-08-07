# EvalRAG 架构图

## 在线回答链路

```mermaid
flowchart TD
    REQ["RagRequest\nquery / top_k / retriever"] --> ROUTER["Router\nRule / Semantic / Hybrid"]
    ROUTER -->|RouteDecision| RET["Retriever\nKeyword / Dense / RRF Hybrid"]
    RET -->|"list[RetrievalResult]"| GATE["Evidence Gate\nscore / count / source coverage"]
    GATE -->|retryable, max once| RETRY[扩大 source 范围]
    RETRY --> RET
    GATE -->|insufficient| REFUSE["RagResponse\ninsufficient_evidence"]
    GATE -->|sufficient| CONTEXT["Context Builder\n完整 Chunk + character budget"]
    CONTEXT -->|BuiltContext| GEN["Generator\nJSON contract + cited ids"]
    GEN -->|format error, max once| REPAIR[格式修复生成]
    REPAIR --> GEN
    GEN -->|GenerationResult| VALID["Citation Validator\n存在 / 重复 / sufficient 组合"]
    VALID -->|valid| RESP["RagResponse\nanswer + citations"]
    VALID -->|invalid| ERROR["RagResponse\ncontrolled error"]

    ROUTER -. stage result .-> TRACE["AgentTrace JSONL\none request / multiple attempts"]
    RET -. ranks + reason .-> TRACE
    GATE -. decision .-> TRACE
    CONTEXT -. used/skipped ids .-> TRACE
    GEN -. output + tokens .-> TRACE
    VALID -. issues .-> TRACE
```

主线传递的数据：

```text
RagRequest
-> RouteDecision
-> list[RetrievalResult]
-> EvidenceDecision
-> BuiltContext
-> GenerationResult
-> ValidationResult
-> RagResponse + AgentTrace
```

Evidence Gate 判断“当前是否值得调用模型”，Citation Validator 判断“模型返回的引用
结构是否合法”，Claim-Level Grounding 则在离线评测中判断“事实是否真的被引用证据
支持”。三者解决的问题不同。

## 离线评测链路

```mermaid
flowchart TD
    RAW["五类中文文档"] --> ING["Ingestion\nDocument -> Chunk"]
    ING --> CORPUS["Manifest + versioned chunks + stats"]
    LABEL["EvaluationCase\nexpected intent/sources/ids/points"] --> RUNNER[Evaluation Runner]
    CORPUS --> RUNNER
    CONFIG["RunConfig\ndataset / split / strategy / version"] --> RUNNER
    RUNNER --> PRED["System predictions\ncase_results + traces"]
    PRED --> METRIC["Router / Recall / MRR\nCitation / Abstention / latency"]
    PRED --> AUDIT["Semantic Key-Point\nClaim-Level Grounding"]
    METRIC --> REPORT["summary + failures + report"]
    AUDIT --> REPORT
    REPORT --> FAILURE["confirmed failure"]
    FAILURE --> REG["RegressionCase\nfixed or open"]
    REG --> TEST[automated regression]
```

## 关键配置取舍

| 决策 | 当前选择 | 原因 |
|---|---|---|
| Router | Rule + Semantic Hybrid | dev Accuracy 最高，但记录 CPU 延迟代价 |
| Retriever | RRF Hybrid | frozen Recall@3/MRR 优于 Keyword 与 Dense 单路 |
| Reranker | 关闭 | dev Recall、MRR 和 P95 同时退化 |
| Evidence | 最多一次 source 扩展 | 避免弱证据直接生成和无限重试 |
| Generation | JSON contract，temperature=0 | 便于解析、引用校验和复现 |
| Grounding | claim-level，unknown 不算 supported | Judge 失败不能伪装成答案可靠 |

对应数字和工件见 [最终实验报告](final_experiment_report.md)。
