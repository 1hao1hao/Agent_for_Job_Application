# EvalRAG 面试讲解指南

## 三分钟主线

**第 1 分钟：问题与系统。**

我做的不是一个只展示回答的聊天 Demo，而是一个可观测、可评测、可回归的 RAG
Agent Harness。当前用中文求职资料作为多源场景，在线链路包含 Hybrid Router、
Keyword/Dense/RRF Hybrid Retrieval、Evidence Gate、Context Builder、结构化 LLM
Generation 和 Citation Validator；每次请求只落一条 Trace，但 Trace 中保留多次
attempt，能还原路由、召回、重试、拒答和耗时。

**第 2 分钟：实验与取舍。**

我构建了 100 文档、310 Chunk、120 Query 的 v0.2 benchmark，只在 80 条 dev 上
调配置，在 40 条 frozen test 上一次性比较。Hybrid 相比 Keyword 将 Recall@3 从
55.56% 提升到 68.33%，Recall@5 从 67.22% 提升到 74.44%，MRR 从 60.83% 提升
到 66.78%，但 P95 从 99.55 ms 增加到 802.50 ms。Reranker 在 dev 上反而同时
降低 Recall/MRR 并增加延迟，所以最终关闭，没有只保留好看的实验。

**第 3 分钟：可靠性与失败闭环。**

真实模型 frozen run 中，Citation Validity 和 Abstention Accuracy 为 100%，总延迟
P95 为 4136.71 ms。随后我对保存的 predictions 做逐要点和逐 claim 审核，发现
Grounding 有 3 条 unknown，因此没有把 0/20 known 写成“零幻觉”。一个真实 Router
失败通过 Trace 定位到过宽规则词，修复后 Rule Accuracy 从 88.75% 到 91.25%，并
转成 fixed regression；这形成了 Trace 定位、单变量修复、全量 dev 对照和回归保护
的闭环。

## 六个核心追问方向

### 1. 数据

**可能追问：语料为什么要有 manifest、versioned chunks 和 frozen test？**

回答边界：manifest 说明文档来源、日期、公开和脱敏状态；versioned chunks 固定一次
实验消费的证据；frozen test 防止看完结果继续调参。要主动说明 v0.2 是项目自建半
真实 benchmark，不冒充线上真实流量。

### 2. 检索

**可能追问：为什么用 RRF，而不是把 Keyword 与 Dense 分数直接相加？**

回答边界：两路原始分数量纲不可比，RRF 只融合 rank；它能兼顾精确术语和语义候选，
但需要解释 P95 成本以及 dev 上 MRR 不总是提升。

### 3. 可靠性

**可能追问：Evidence Gate、Citation Validator 和 Grounding 有什么区别？**

回答边界：Gate 在生成前判断证据数量、分数和来源覆盖；Validator 在生成后检查引用
ID 和输出组合是否合法；Grounding 在离线评测中逐 claim 判断事实是否被 cited
Context 支持。三者不能互相替代。

### 4. 评测

**可能追问：Recall@5、Citation Validity 和 E2E Success 分别能证明什么？**

回答边界：Recall@5 是相关 Chunk 覆盖；Citation Validity 只验证 ID；E2E 才组合
回答/拒答、引用、要点和 Grounding。当前 E2E 因 Judge unknown 为 unavailable，
不能用局部指标代替。

### 5. 延迟与成本

**可能追问：为什么 Hybrid 和 Semantic Router 延迟高，怎样优化？**

回答边界：当前单进程 CPU 编码 query，Hybrid 还执行两路检索；文档向量已经离线
构建。后续可做 query embedding 缓存、批处理、量化或更轻模型，但需要用 profiling
和同集指标验证，不能只说上向量数据库。

### 6. 失败案例

**可能追问：讲一个由真实失败推动的修改。**

回答顺序固定为：现象 -> Trace 定位阶段 -> root cause -> 单变量假设 -> 修改前后
完整 dev -> 是否有退化 -> Regression Case。可使用“公司”过宽规则词案例，证据在
`reports/failure_analysis/p0-d4-company-keyword/`。

## 数字与证据速查

| 需要记住的数字 | 证据 |
|---|---|
| 100 文档 / 310 Chunk / 120 Query | `data/evaluation/corpus_stats.json`、`evalrag_v0.2_validation.json` |
| Hybrid frozen Recall@3 68.33%、Recall@5 74.44%、MRR 66.78% | `reports/comparisons/p0-d5-v02-frozen-test-20260804/` |
| Hybrid retrieval P95 802.50 ms | 同上 |
| Live frozen P95 4136.71 ms、39,147 tokens、$0.006317 | `reports/final/p0-d5-live-llm-v0.2/` |
| Semantic Coverage 74.44%，Grounding 3 unknown | `reports/final/p0-d6-semantic-grounding-v0.2/` |
| Rule Router 88.75% -> 91.25%，fixed 1/1 | `reports/ablations/p0-d4-v02-dev-20260803/`、`reports/regression/p0-d5-v0.2/` |

## 面试前阅读顺序

1. `docs/project_map_zh.md`：先恢复两条主链和数据结构。
2. `src/intern_rag/agent/pipeline.py` 的 `RagPipeline.run()`：看懂正常回答、source retry、拒答和格式修复。
3. `src/intern_rag/retrieval/hybrid.py`：看懂两路结果如何用 RRF 去重融合。
4. `src/intern_rag/evaluation/semantic_audit.py`：看懂保存 predictions 如何被重新审核。
5. 本文的失败案例与数字速查：练习三分钟讲稿，不需要通读所有辅助脚本。
