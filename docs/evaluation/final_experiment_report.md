# EvalRAG P0-P1 最终实验报告

## 1. 实验范围与数据版本

### 1.1 当前主版本：evalrag_v0.3

- Corpus：669 份输入经 SHA-256/SimHash 去重后保留 658 份文档、4208 个 Chunk。
- 来源：339 份 JD、120 份面试资料、159 份项目文档、20 份简历、20 份用户画像。
- 来源方式：558 份公开数据集/开源仓库材料，100 份本地半真实材料；公开材料保存 URL、
  revision、许可与采集时间。
- Benchmark：240 条 Query，160 dev / 80 frozen test；每类 30 条。
- 当前类别：单来源、跨来源、语义改写、hard negative（当前标为不可回答）、不可回答、时效冲突、
  2-hop 关系和 3-hop 关系。
- Frozen 规则：只在 dev 选择配置；test 上每个声明策略运行一次，查看结果后不继续调参。
- Corpus 统计：[corpus_stats_v0.3.json](../../data/evaluation/corpus_stats_v0.3.json)；标签校验：
  [evalrag_v0.3_validation.json](../../data/evaluation/evalrag_v0.3_validation.json)。

v0.3 Benchmark 标签由系统依据真实 Chunk ID 和 Graph Edge 构造并进行一致性校验，审核方式为
`corpus_grounded_ai_assisted`，不是完整人工标注。当前知识构建、Graph+Vector、Adaptive
Retrieval 和 P1 frozen 结果均以 v0.3 为主。

### 1.2 evalrag_v0.2 是历史 P0 基线

v0.2 包含 100 份本地半真实文档、310 个 Chunk 和 120 条 Query；单来源、多来源、语义改写、
不可回答各 30 条，80 dev / 40 frozen test。它只用于保存 Keyword/Dense/RRF、Router、
Evidence Gate 和早期 LLM Pipeline 的历史可复现实验，**不代表当前知识库规模和 Query 设计**。

对应工件仍保留在 [corpus_stats.json](../../data/evaluation/corpus_stats.json) 和
[evalrag_v0.2_validation.json](../../data/evaluation/evalrag_v0.2_validation.json)，不能因为升级到
v0.3 就覆盖或删除。

### 1.3 为什么 data/raw 仍能看到 v02 文件

`data/raw/` 是早期本地输入层，不是当前完整 Corpus。目录中仍有约 70 份 `v02_*.md`，因为版本
升级采用追加和生成新快照，而不是修改旧原始材料。v0.3 构建时会读取这 100 份本地材料，再合并
558 份公开材料；公开材料直接规范化到：

```text
data/processed/documents/evalrag_v0.3.jsonl
data/processed/chunks/evalrag_v0.3.jsonl
data/evaluation/corpus_manifest_v0.3.jsonl
```

因此打开 `data/raw/jd/v02_*.md` 看到不相关文本，不代表 v0.3 只有这些文本；但它也暴露了真实
问题：v0.3 仍把旧半真实材料以及教师、电气等宽领域岗位纳入统一抽样，部分 Query 甚至直接询问
`v02_07_jd` 这类文件标题。规模提高了，面向 AI/RAG 求职推理的场景纯度仍然不足。

### 1.4 下一版 Benchmark 的改进口径

已冻结的 v0.3 不原地修改。下一版必须创建独立 `evalrag_v0.4`，Query 不再以“根据某文件标题
概括内容”为主要模板，而围绕 Query Analyzer 的证据需求分层，每类 30 条、保持 160 dev / 80
frozen test：

| Query 类型 | 主要验证能力 | 示例方向 |
|---|---|---|
| `exact_fact` | BM25 精确术语和岗位字段召回 | 某 AI 岗位明确要求哪些框架或实习时长 |
| `semantic_explanation` | Dense 同义表达和概念解释 | 不使用原文措辞说明如何降低 RAG 无依据回答 |
| `multi_source_synthesis` | Hybrid 跨来源证据融合 | 结合 JD、简历和项目日志分析技能缺口 |
| `relation_reasoning` | Graph+Vector 实体链接和有界多跳 | 哪个项目通过哪些技能证明符合目标岗位 |
| `comprehensive_analysis` | Adaptive 在 Hybrid/Graph 间选路 | 比较两个岗位并结合个人经历给出取舍依据 |
| `freshness_conflict` | 岗位版本、状态与来源优先级 | 同一岗位多个版本冲突时应采用哪条证据 |
| `answerable_hard_negative` | 在同主题干扰项中找正确证据 | 多个 RAG 项目中只有一个满足特定技术条件 |
| `unanswerable` | Evidence Gate 与可靠拒答 | Corpus 没有内部薪资或未公开招聘结论 |

v0.4 还需限制目标领域，优先选择 AI Agent、RAG、搜索推荐、后端/平台工程相关 JD 与项目材料；
`relevant_chunk_ids`、关系路径和 expected points 必须回到具体证据，不能再用标题或 `unknown`
充当核心答案标签。在该版本真正构建并完成同集实验前，本报告不会把它写成已实现结果。

## 2. 检索策略

### Dev 消融

| Retriever | Recall@3 | Recall@5 | MRR | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| Keyword | 73.33% | 80.56% | 85.97% | 9.23 ms | 14.09 ms |
| Dense | 72.78% | 80.83% | 67.42% | 198.93 ms | 1391.53 ms |
| RRF Hybrid | **80.83%** | **85.56%** | 83.19% | 202.90 ms | 1095.15 ms |

Hybrid 增加了多来源候选覆盖，但 dev MRR 低于 Keyword，说明语义候选进入前排也会
引入排序噪声。差异 Case 与分类指标保存在
[P0-D3 Ablation](../../reports/ablations/p0-d3-v02-dev-20260801/report.md)。

### Frozen test

| Retriever | Recall@3 | Recall@5 | MRR | P95 |
|---|---:|---:|---:|---:|
| Keyword | 55.56% | 67.22% | 60.83% | 99.55 ms |
| Dense | 56.67% | 74.44% | 59.28% | 494.05 ms |
| RRF Hybrid | **68.33%** | **74.44%** | **66.78%** | 802.50 ms |

最终选择 Hybrid：相比 Keyword，Recall@3 +12.77 pp、Recall@5 +7.22 pp、MRR
+5.95 pp；代价是 P95 +702.96 ms。原始工件见
[Frozen Comparison](../../reports/comparisons/p0-d5-v02-frozen-test-20260804/report.md)。

Reranker 只在 dev 对比：token-overlap candidate 将 Hybrid Recall@3 从 80.83% 降至
73.33%、Recall@5 从 85.56% 降至 83.89%、MRR 从 83.19% 降至 79.44%，P95
从 700.21 ms 增至 902.75 ms，因此冻结配置为关闭。该轻量 scorer 再次按词面重叠
覆盖率排序，削弱了 Hybrid 已融合的语义信号，并把共享“岗位、简历、混合检索”等
词语的同主题干扰 Chunk 推到前排。例如 `v02_multi_002` 的相关面试 Chunk 从第 4 名
跌出 top 5，使 Recall@5 从 1.0 降至 0.667。这个结果只能说明当前 token-overlap
candidate 不适合该数据集，不能推导出 CrossEncoder 等 Reranker 普遍无效。负向结果见
[Reranker Ablation](../../reports/ablations/p0-d5-reranker-dev-20260804/report.md)。

P1 补跑了真实 `BAAI/bge-reranker-base` CrossEncoder（固定 revision
`2cfc18c9415c912f9d8155881c133215df768a70`）。在完全相同的 Hybrid top-20 和 80 条
dev 上，Recall@3/Recall@5/MRR 分别从 80.83%/85.56%/83.19% 降至
63.06%/76.67%/63.17%，CPU P95 从 700.71 ms 增至 7511.44 ms。失败分析显示，
当前 benchmark 包含较多同主题近似段落，通用 CrossEncoder 倾向把语义相关段落排在
人工指定的精确证据前。因此默认配置仍关闭 Reranker；该结论来自真实神经模型负向
实验，不再使用“只有 adapter”表述。工件见
[CrossEncoder Ablation](../../reports/ablations/p1-cross-encoder-v02-dev-20260811/report.md)。

### Context Builder 紧预算消融

原 `rank_prefix` 严格按 rank 填充，可能在多来源问题中先装入多个同来源 Chunk。
新增 `source_balanced`：先为 Router 必需的每个来源选择最高排名证据，再按 rank 填充
剩余预算；单条 Chunk 过长时跳过而不截断。实验复用已保存的 Hybrid dev 预测，不重新
运行 Router/Retriever，只比较 Context 选证据策略。

| Budget | Strategy | Relevant Recall | Source Coverage | Full Source Rate |
|---:|---|---:|---:|---:|
| 800 | Rank Prefix | 66.98% | 56.92% | 24.53% |
| 800 | Source Balanced | **68.55%** | **61.01%** | **30.19%** |
| 1200 | Rank Prefix | **80.19%** | 61.32% | 30.19% |
| 1200 | Source Balanced | 79.25% | **78.93%** | **54.72%** |
| 4000 | Rank Prefix | 94.34% | 80.82% | 60.38% |
| 4000 | Source Balanced | 94.34% | 80.82% | 60.38% |

紧预算下 Source Balanced 明显提高来源覆盖，1200 字符时以 0.94 pp relevant recall
换取 17.61 pp 平均来源覆盖和 24.53 pp 完整来源覆盖；默认 4000 字符下 top-5 全部
放得下，两种策略没有差异。P1 服务默认启用 Source Balanced，完整预算扫描见
[Context Ablation](../../reports/ablations/p1-context-builder-v02-dev-20260811/report.md)。

### Adaptive Context Engine 与分层记忆消融

`evalrag_context_v0.1/dev` 包含 60 组、每组 5 轮，共 300 turns，覆盖指代、省略、历史约束、
跨会话记忆、冲突、主题切换、多来源和不可回答。四种策略运行相同 Case 和 220 token 预算：

| Strategy | Follow-up | Semantic KPC | Prompt tokens | History Redundancy | Memory Recall | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Recent window | 36.67% | 28.30% | 67.18 | 42.86% | 0.00% | 1800.79 |
| Summary + recent | 100.00% | 100.00% | 75.55 | 42.86% | 0.00% | 3903.68 |
| Semantic memory | 100.00% | 100.00% | 65.38 | 0.00% | 84.91% | 3795.67 |
| Adaptive policy | **100.00%** | **100.00%** | **60.68** | **0.00%** | 56.60% | 4108.42 |

Adaptive 先以 History Token Pressure、指代/省略和 BGE 语义连续性、Memory
`similarity * importance` 生成 ContextPlan，再由同一 ContextEngine 执行预算和跨层去重。
相对质量同为 100% 的 Summary+Recent，平均 Prompt Token 降低 19.68%，History
Redundancy 降低 100%。按需 Memory Recall 56.60% 低于 Semantic Memory 的 84.91%，
说明减少 Memory 调用会牺牲部分“相关记忆进入 Context”的覆盖，但本组场景可由 Summary
保留必要事实。Adaptive CPU P95 比 Summary+Recent 高 5.24%，来自每轮语义信号计算，
因此该方案优化的是 Prompt 成本与重复度，并非延迟上的全面 Pareto 提升。

Semantic KPC 与策略信号使用固定 `BAAI/bge-small-zh-v1.5` revision
`7999e1d3359715c523056ef9478215996d62a620`。该实验由 Context Engine 产生确定性摘录，
不调用 LLM，数据是 scenario-authored AI-assisted，因此 100% 只说明构造场景中的必要事实被保留，
不能写作真实多轮问答准确率。逐 turn signals/plan、裁剪原因和 Case 结果见
[Adaptive Context report](../../reports/ablations/p1-adaptive-context-v02-dev-20260823/report.md)。

### Corpus v0.3 与 Graph + Vector

P1-D4 使用固定 revision 的公开数据集/开源仓库和脱敏自有资料构建 `evalrag_v0.3`。
669 份输入经 exact hash 与 SimHash 去重后保留 658 份，生成 4208 个 Chunk；公开/私有
分别为 558/100 份，五类 source 均有覆盖。公开材料保存 URL、revision、采集时间、许可
和审核方式；240 条评测标签均标记为 corpus-grounded AI-assisted，不能表述为人工审核。

160 条 dev 使用完全相同的 v0.3 标签和 top-k：

| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 32.08% | 37.08% | 30.04% | 30.47% | 11.65 | 14.88 |
| Dense | 40.42% | 43.75% | 40.24% | 38.84% | 308.23 | 811.23 |
| Adaptive Vector | 40.83% | 44.58% | 40.26% | 39.14% | 526.71 | 5267.36 |
| Graph-only | 29.58% | 35.42% | 37.29% | 32.74% | 8.50 | 13.51 |
| Graph + Vector | **42.50%** | **47.50%** | **41.25%** | **40.49%** | 513.09 | 1496.60 |

Graph + Vector 总体质量最佳；Graph-only 在 2/3 跳关系题上优于纯 Dense，但在时效和普通
语义题上明显不足，说明图适合补充显式关系而非替代文本召回。Adaptive Vector 的 P95
受少量真实 CrossEncoder 调用影响达到 5.27 秒，Graph + Vector 仍有约 1.50 秒尾延迟，
因此当前结果是质量/延迟取舍，不是全面 Pareto 提升。80 条 frozen test 的一次性结果见第 8 节。

## 8. P1-D7 最终冻结发布

### 8.1 Reranker 收口（dev-only）

在相同 `evalrag_v0.3/dev` 160 条 Case 上，比较 BGE base top-20、BGE base top-10
和多语 MiniLM top-10。MiniLM 固定 revision `1427fd65...8825`，约 1.18 亿参数：

| Variant | Recall@5 | MRR | NDCG@5 | 调用率 | P95 |
|---|---:|---:|---:|---:|---:|
| BGE base k20 | 44.58% | 40.26% | 39.14% | 13.13% | 5566.71 ms |
| BGE base k10 | 43.75% | 39.58% | 38.34% | 21.25% | 3157.58 ms |
| MiniLM k10 | 43.75% | 40.21% | 38.69% | 21.25% | 2100.94 ms |

最终按预先声明的取舍选择低置信时按需 MiniLM k10：P95 明显降低，MRR 基本持平，
Recall@5 退化 0.83 pp。不同 candidate_k 会改变 Adaptive confidence 和触发 Case，
因此调用率不是常数。工件见 `reports/ablations/p1-d7-reranker-closure-v03-dev-20260816/`。

### 8.2 Frozen general retrieval 与 graph challenge

最终清单 `configs/final/p1_v0.3.json` 保存 dataset/config SHA-256，运行脚本拒绝覆盖
已有发布目录。80 条 v0.3 frozen test 的结果如下：

| Strategy | Recall@3 | Recall@5 | MRR | NDCG@5 | P95 |
|---|---:|---:|---:|---:|---:|
| BM25 | 39.17% | 46.67% | 35.19% | 36.49% | 15.50 ms |
| P1 Graph + Vector | **49.17%** | **63.33%** | **57.58%** | **55.08%** | 1209.40 ms |

10 条 graph challenge frozen（8 条可答）的 Recall@5/MRR/NDCG@5 为
91.67%/58.33%/63.48%，路径有效率和选路准确率均为 100%。失败 Case 未删除，
且 frozen 后不再使用该版本调参。完整工件位于
`reports/releases/p1-d7-v03-frozen-20260816/`。

### 8.3 Model Gateway 与故障矩阵

Gateway 统一 DeepSeek primary 与 OpenAI-compatible backup，硬限制每 Provider 最多
2 次、并发 4、60 秒 timeout，并只对 timeout/连接/429/5xx 指数退避；401/403
不重试。连续失败触发 circuit breaker，恢复窗口只允许一次 half-open probe。
每次 attempt 记录 provider、reason、latency、tokens 和估算成本，不记录 Prompt 或密钥。

6 类确定性 Fake 故障场景成功率 83.33%、fallback rate 50%、P95 3.53 ms；其中一类
双 Provider 不可用按预期受控失败。这只验证控制流，不代表真实线上 SLO。真实 DeepSeek
primary smoke 成功（59 tokens，约 7.81 秒）；未提供 OpenAI 凭证，所以真实 fallback
未执行。工件分别位于 `reports/fault_injection/p1-d7-model-gateway-v01/` 与
`reports/smoke/p1-d7-model-gateway/`。

### 8.4 P1 frozen E2E 负结果

80 条 v0.3/test 的最终 Pipeline Run 返回 8 answered、69 insufficient、3 error；
Citation Validity 96.25%、Abstention Accuracy 100%，但可答案例的“合法引用且词面
要点覆盖至少 50%”成功率只有 8.33%。26 次 Gateway 调用共 39,620 tokens，P95
3739.64 ms。该 Run 未做 Claim-Level Grounding，因此不计算 UAR 或完整 grounded E2E。

Trace 显示主要问题不是 Graph Retriever 未运行，而是 v0.2 Router/Evidence Gate 仍要求旧
intent/source coverage：47 条停在 `unanswerable_route`，7 条扩源后仍为
`required_sources_missing`，另有 3 条 `citation_invalid`。它对 v0.3 的关系型、hard-negative
和语义改写问题产生大量 unexpected abstention。这说明局部 Recall 提升不能替代端到端评测。冻结 prediction 保留在
`reports/runs/p1-d7-v03-frozen-20260816-live-e2e/`，不会为获得更好数字重新生成。

### 8.5 未完成的外部环境验证

`evalrag_context_v0.1` 只有 dev，没有 untouched test，因此 Context/Memory 只保留 dev
消融，不伪造 multi-turn frozen。校园服务器没有 Docker Engine，pgvector HNSW 与 Neo4j
真实重启恢复留给 GitHub Actions 的持久化工作流；本地只完成 adapter、配置与 Fake repository
测试。完整测试为 216 run、212 passed、4 skipped，CI Gate 与 fixed regression pass rate
均为 100%。

- 质量报告：`reports/data_quality/evalrag_v0.3/collection_report.json`
- Benchmark 校验：`data/evaluation/evalrag_v0.3_validation.json`
- 消融报告：`reports/ablations/p1-d4-v03-dev-20260816-final/report.md`

## 3. Router 与可靠性

Rule、Semantic、Hybrid Router 在相同 80 条 dev 上的 Accuracy 分别为 91.25%、
87.50%、96.25%；Hybrid P95 为 2600.27 ms，表明质量提升伴随显著 CPU 语义编码
开销。完整分类结果见 [Router Ablation](../../reports/ablations/p0-d4-v02-dev-20260803/report.md)。

真实 LLM frozen Trace 按人工标签拆分后的阶段干预统计如下。“方向正确”只表示最终
回答/拒答状态符合 `answerable` 标签，不代表回答内容已通过全部质量指标。

| 标签子集 | 阶段结果 | Case 数 | 最终状态 | 方向是否正确 | 说明 |
|---|---|---:|---|---|---|
| 可回答（30） | Gate sufficient -> Generator -> Validator 通过 | 23 | `answered` | 是 | 正常返回带合法 citation 的回答，内容质量继续由 Coverage 与 Grounding 判断 |
| 可回答（30） | Gate required sources missing，扩源一次后仍不足 | 6 | `insufficient_evidence` | 否 | 相关证据虽有召回，但没有覆盖 Router 要求的全部来源，属于 unexpected abstention |
| 可回答（30） | Gate sufficient，Generator 输出 `sufficient=false` 且携带 citations | 1 | `error` | 否 | Validator 正确拦截结构矛盾，但该可回答 Case 最终未回答 |
| 不可回答（10） | Router unknown -> Gate unanswerable route | 9 | `insufficient_evidence` | 是 | 未调用 Generator，直接可靠拒答 |
| 不可回答（10） | Gate sufficient -> Generator 判断 Context 不含所问信息 | 1 | `insufficient_evidence` | 是 | 模型返回 `sufficient=false` 且 citations 为空，Validator 校验通过 |

这是一份基于已保存 Trace 的**阶段干预审计**，不是无 Gate/有 Gate 的随机对照实验，
因此不能声称 Gate 单独带来多少百分点提升。10 条不可回答 Case 最终全部拒答，其中
9 条由 Gate 直接处理、1 条由 Generator 声明证据不足；Validator 还拦截了 1 条结构
矛盾输出。同时，required-source coverage 过严造成 6 条 retrieval miss。Case 与
Trace 位于 [Live Frozen Run](../../reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/)。

## 4. 端到端、延迟与成本

| 指标 | Frozen test 结果 | 边界 |
|---|---:|---|
| Citation Validity | 100.00% | 只证明返回 ID 位于本轮 Context |
| Semantic Key-Point Coverage | 74.44% | 模型评分；lexical 为 68.89% |
| Abstention Accuracy | 100.00% | 10 条不可回答 Case 全部拒答 |
| Claim-level UAR | 0/20 known | 另有 3 unknown，不能写作零幻觉 |
| E2E Success | unavailable | Grounding unknown 按协议使总体不可用 |
| Total latency P50 / P95 | 3226.93 / 4136.71 ms | 单进程、CPU embedding |
| LLM calls / tokens | 25 / 39,147 | 33,170 input + 5,977 output |
| Estimated cost | $0.006317 | 2026-08-04 Provider 价格快照 |

真实生成工件见 [P0-D5 Live LLM](../../reports/final/p0-d5-live-llm-v0.2/report.md)；离线
Semantic/Grounding 审核复用保存 predictions，没有重新运行 Generator，见
[P0-D6 Audit](../../reports/final/p0-d6-semantic-grounding-v0.2/report.md)。

## 5. 失败驱动修复

真实失败 Query 是“请告诉我未公开公司薪资名单。”；原始 benchmark 措辞较生硬，
自然含义是“请提供一份未公开的公司内部薪资名单”。其 EvaluationCase 标注为
`answerable=false`、期望 `unknown`，因为 Corpus 中没有未公开薪资数据。Rule Router
却仅因命中宽泛关键词“公司”，输出 `analyze_jd -> [jd]`；这里错误的不是“查公司岗位
应该去 JD”，而是把任何包含“公司”的问题都当成可回答的 JD 分析。

排查时先从 Case Result/Trace 看到 `matched_keywords=["公司"]` 和错误的
`RouteDecision`，因此把根因定位在 Router，而不是 Retriever 或 Generator。修复只删除
`analyze_jd` 规则中的“公司”，保持数据、Retriever 和其他配置不变；重跑完整 80 条
dev 后，Rule Router Accuracy 从 88.75% 提升到 91.25%，错误数从 9 降到 7，未发现
其他 Case 退化。最后把该 Query 固化为 `reg-router-company-list-v1`：以后 Router 改动
都必须断言它仍返回 `unknown + []`。fixed regression 1/1 通过；另有 4 条尚未修复的
open case 明确不进入通过率分母。

- Root cause：[failure analysis](../../reports/failure_analysis/p0-d4-company-keyword/analysis.json)
- Before/after：[P0-D4 report](../../reports/ablations/p0-d4-v02-dev-20260803/report.md)
- Regression：[summary](../../reports/regression/p0-d5-v0.2/summary.json)

## 6. P1-D6 Runtime、Router Feedback 与 CI Gate

P1-D6 在同一 `evalrag_v0.2/dev` 80 Case 上比较四个 Router 版本：

| Router | Accuracy | Unknown Precision | Unknown Recall | P95 ms |
|---|---:|---:|---:|---:|
| Rule v0.2 | 91.25% | 100% | 100% | 0.010 |
| Semantic v0.2 | 87.50% | 88.89% | 80% | 394.259 |
| Hybrid v0.2 | 96.25% | 100% | 100% | 389.733 |
| Feedback Hybrid v0.3 | **100%** | 100% | 100% | 200.071 |

首个候选直接把三条完整失败 Query 加入语义 prototype，因不同意图共享相同主题正文而污染相邻
Query，Accuracy 降至 88.75%，被 shadow gate 拒绝。最终版本从确认反馈提取冒号前的短意图锚点，
其余 Query 委托原 Hybrid，修复 3 条且完整 dev 无新增退化。该方法适合明确意图前缀，不等于
可泛化的在线自学习；反馈数据、父版本、报告和 active/rollback 状态均有记录。

CI Gate 运行 4 条 fixed regression，Pass Rate 为 100%。Router 是本轮变化项；Recall@5、NDCG@5、
Grounding Support、E2E 和 P95 的未修改子系统使用版本化 dev/reference 值，不包装成新提升，且未
运行 frozen test。Runtime 另覆盖 Fake Replay、checkpoint/resume、sink 故障隔离和三入口复用。

- Router shadow：`reports/ablations/p1-d6-router-feedback-v01-dev-20260816/`
- CI Gate：`reports/ci/p1-d6-evaluation-gate-v01/`
- Runtime/fault matrix：`reports/runtime/p1-d6-runtime-v01/summary.json`
- Sanitized traces：`traces/sanitized_examples/p1-d6-{answered,retry,error}.json`

## 9. v0.3 统一消融矩阵（dev-only）

P1-D9 在同一 `evalrag_v0.3/dev` 上补齐 Router、Retriever 和 Reranker 策略对照；
Context Engine 复用代码未变的 60 组/300 turns 正式工件。v0.3 没有
`expected_intent`，因此 Router 只报告来源路由指标，不把它冒充为 Intent Accuracy。

### 9.1 Router 来源路由

| Router | Source Exact | Answerable Exact | Source Precision | Source Recall | Unanswerable Acc | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Rule | 43.75% | 26.67% | 49.06% | 47.50% | 95.00% | 0.009 |
| Semantic | 25.00% | 15.00% | 27.19% | 27.81% | 55.00% | 500.659 |
| Hybrid | 43.75% | 26.67% | 49.06% | 47.50% | 95.00% | 396.603 |
| Feedback Hybrid | 43.75% | 26.67% | 49.06% | 47.50% | 95.00% | 496.285 |

Feedback Hybrid 在 v0.2 dev 上的改善没有迁移到 v0.3：3 条旧反馈锚点没有覆盖
新的关系型、跨来源 Query 分布。这是路由数据与新 Corpus 不匹配的真实失败，
不是应该隐藏的均值。

### 9.2 全检索策略

120 条可答 dev Case 使用相同 relevant IDs、`top_k=5` 且不加 source filter：

| Retriever | Recall@3 | Recall@5 | MRR | NDCG@5 | P95 ms |
|---|---:|---:|---:|---:|---:|
| Keyword | 26.25% | 27.50% | 29.31% | 26.79% | 92.496 |
| BM25 | 32.08% | 37.08% | 30.04% | 30.47% | 14.591 |
| Dense | 40.42% | 43.75% | 40.24% | 38.84% | 1422.115 |
| Keyword + Dense RRF | 36.67% | 41.67% | 38.39% | 36.67% | 1427.806 |
| BM25 + Dense RRF | 41.67% | 50.00% | 44.03% | 43.07% | 1115.262 |
| Graph-only | 29.58% | 35.42% | 37.29% | 32.74% | 14.055 |
| **Graph + Vector RRF** | **55.42%** | **58.33%** | **52.10%** | **51.03%** | 1186.649 |
| Adaptive Graph | 43.75% | 48.33% | 42.21% | 41.03% | 1904.738 |

固定 Graph + Vector RRF 是本轮最佳质量策略；当前 Query Analyzer 对部分关系型
Query 漏触发 Graph，使 Adaptive Graph 的 Recall@5/MRR 低 10.00/9.89 pp。因此
自适应不能只看“是否减少重策略调用”，还必须评估 selector 造成的质量损失。

#### Evidence-Need Query Analyzer 迭代（2026-08-25）

后续将旧 `is_cross_document` 规则改为：

```text
QueryFeatures（精确 / 语义 / 多源 / 关系）
  -> EvidenceRequirement
  -> BM25 / Dense / Hybrid / Graph+Vector
```

Graph 只由实体关系推理需求触发；精确事实、语义解释和多源综合分别选择 BM25、Dense 和
Hybrid。首轮因漏掉“知识关系、2/3 跳联系”和“不使用原文、概括”等表达，Recall@5/MRR
退化到 43.75%/38.29%；根据逐 Case Trace 做单变量修复后，160 条 dev 的 Recall@5 为
48.75%、MRR 为 39.01%、P95 为 1322.25 ms。相比旧 Adaptive 的 48.33%/42.21%，
召回覆盖略升但首条相关证据排序退化，因此仍没有超过固定 Graph+Vector。

失败工件和修复后报告均保留在
[`p1-query-evidence-adaptive-v03-dev-20260825-fixed`](../../reports/ablations/p1-query-evidence-adaptive-v03-dev-20260825-fixed/report.md)。
该实验是 dev-only，没有重跑或修改 frozen test。

### 9.3 Reranker 控制变量

三组固定为 BM25 + Dense + RRF、同一 MiniLM revision、`candidate_k=10`，只改变
`never / always / on_demand` 策略。评测共 160 Case，质量指标的分母为 120 条可答 Case。

| Policy | Recall@3 | Recall@5 | MRR | NDCG@5 | Invocation | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Never | 43.75% | 50.83% | 44.31% | 43.67% | 0.00% | 1252.334 |
| Always | **47.08%** | **50.83%** | **49.22%** | **46.95%** | 100.00% | 2799.094 |
| On demand | 44.58% | 50.00% | 45.18% | 43.89% | 18.12% | 2174.664 |

Always Rerank 将 MRR 提高 4.91 pp，但 P95 增加 1546.76 ms；On-demand 只调用
29/160 Case，但 MRR 只提高 0.87 pp、Recall@5 下降 0.83 pp，P95 仍增加
922.33 ms。本轮没有 Pareto 最优策略，不再笼统声称“Reranker 无效”或
“按需 Rerank 同时提升质量和延迟”。

### 9.4 Context Engine

P1-D9 的固定 mode 表已由本报告前文的 Adaptive Context v0.2 对照替代。新实验仍使用
相同 `evalrag_context_v0.1/dev` 60 组/300 turns 和 220 token 预算，但统一启用升级后的
跨层去重并新增 Adaptive Policy，因此旧版 76.62/66.22 token 与微秒级 baseline 延迟
不能和新版表混用。当前可引用结论为：Adaptive 保持 100% Follow-up Success，相对
Summary+Recent 将 Prompt Token 降低 19.68%、History Redundancy 降低 100%，但 CPU
P95 高 5.24%，且该结果不是自由生成答案准确率。

完整配置、逐 Case 结果、failures 和差异 Case 见
[`p1-d9-v03-dev-ablation-20260817`](../../reports/ablations/p1-d9-v03-dev-ablation-20260817/report.md)。
本轮是 dev-only 实验，没有重跑 frozen test，也没有修改冻结标签。

## 7. 结论与限制

EvalRAG 已形成 `Pipeline -> Trace -> Evaluation -> Failure -> Regression` 闭环，
并以 frozen test 记录最终配置的质量、延迟和成本。结果支持“Hybrid 提高当前 benchmark
的检索覆盖”“Validator 拦截结构矛盾输出”等局部结论，不支持“线上准确率”“零幻觉”
或“Reranker 在所有策略下都能无代价提升效果”。最新 v0.3 dev 证据表明，固定
Graph + Vector 是当前 quality-first candidate，但 Adaptive selector、Router 迁移和按需
Reranker 仍有可明确定位的改进空间。

当前主要限制是 v0.3 虽然扩大到 658 份文档，但仍混有早期 `v02_*` 半真实材料和与目标
AI/RAG 岗位无关的宽领域 JD；部分 Query 依赖文件标题或 `unknown` 要点，标签为
corpus-grounded AI-assisted 而非完整人工审核。Adaptive selector 的 Recall@5 略升但 MRR
退化，Evidence Gate 对新关系型 Query 仍有过度拒答，Grounding Judge 与 Generator 属于
同模型家族且存在 unknown。FastAPI、PostgreSQL、Redis Worker 和 Docker Compose 已实现；
校园服务器没有 Docker Engine，真实多容器持久化验证主要依赖 GitHub Actions。可视化前端
未实现；Full E2E 压测已在单 Uvicorn worker、真实 PostgreSQL/Redis 和 DeepSeek 下执行，
但只代表共享校园服务器环境，不作为生产 SLA。

## 10. Adaptive Retrieval v2 与 Evidence Gate 收敛（2026-09-03）

本轮不修改 `evalrag_v0.3` 的 Case、relevant Chunk 或 Graph Edge 标签。先在 160 条 dev
（120 可答）上运行固定 BM25、Dense、BM25+Dense RRF、Graph+Vector，以及可复现的
Adaptive v1 和 v2。语义组 Dense 与 Hybrid 的 Recall@5 同为 50%，但 MRR 为
30.83%/40.00%、P95 为 1520.39/1115.83 ms，因此锁定语义需求使用 Hybrid。

| Dev strategy | Recall@5 | MRR | NDCG@5 | P95 ms |
|---|---:|---:|---:|---:|
| Adaptive v1 replay | 54.58% | 47.76% | 47.01% | 1518.74 |
| Adaptive v2 | 55.42% | 49.88% | 48.78% | 1847.35 |
| Always Graph+Vector | 58.33% | 52.10% | 51.03% | 1696.57 |

Adaptive v2 的策略分布为 BM25 39、Hybrid 41、Graph+Vector 40、`none` 40，Graph 调用率
25%。它相对 v1 改善 Recall@5/MRR，但没有超过 always Graph，也没有取得全面质量/延迟
Pareto 优势。逐 EvidenceRequirement 分组、10 条代表差异 Case 与真实 prediction 位于
`reports/ablations/p1-adaptive-v2-v03-dev-20260903-r1/`。

Gate calibration 分别保存四种 Retriever 的正负 top-1 score 分布和完整 threshold curve。
满足 FAR <= 5% 的最佳点仍分别产生 BM25 33.33%、Dense 51.11%、Hybrid 100%、Graph
100% 的 FRR，均超过 25% 预算，因此锁定配置关闭 raw score gate，转而检查最少结果数、
多来源覆盖与 Graph path。dev 上 Gate FAR 32.74%、FRR 0%、不可回答拒答准确率 100%、
deterministic E2E success 54.38%；这是“原始分数不可分”的负结果，不宣称门控已解决。

CI Gate v2 真实运行 v1/v2 dev prediction：Recall@5/MRR 使用 `1/120` 动态容差，P95
允许最多 1.25 倍工程预算，fixed Regression 必须 100%，NDCG/策略分布/Graph 调用率只报告。
本次门禁通过。配置、数据、图与核心源码 SHA-256 锁定后，在已有历史 Run 的 80 条 test 上
执行本次 release test：

| Release strategy | Recall@5 | MRR | NDCG@5 | P95 ms |
|---|---:|---:|---:|---:|
| BM25 | 46.67% | 35.19% | 36.49% | 15.26 |
| Graph+Vector | **69.17%** | **62.92%** | **61.38%** | 1889.44 |
| Adaptive v2 | **69.17%** | 53.11% | 54.69% | 2031.67 |

锁定 Gate 在 release test 的 Abstention Accuracy 为 100%，但 FAR 为 25%、deterministic
E2E success 为 60%。该 split 曾用于历史实验，因此这里明确称为“锁定配置 release test”，
不包装成全新盲测。正式工件位于
`reports/releases/p1-adaptive-v2-v03-release-test-20260903-r1/`。

## 11. Raw Score 仅触发重试的 Gate 实验（2026-09-04）

在 `evalrag_v0.3/dev` 160 条 Case 上，使用相同 Adaptive v2 Retriever 对比结构
Gate 与 `low_confidence_retry_threshold` 候选。候选阈值来自既有 calibration
operating point，每种 Retriever 独立使用；低分只触发一次扩源，重试后不作为
hard reject 条件。

| Gate | FAR | FRR | 可回答接受率 | 重试率 | 重试成功率 | 证据恢复率 | E2E Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Structural v2 | 43.27% | 0.00% | 84.17% | 11.88% | 0.00% | 0.00% | 60.00% |
| Retry v2.1 candidate | 32.74% | 0.00% | 70.00% | 65.00% | 30.77% | 0.00% | 54.38% |

候选规则触发了 `104/160` 条重试，却没有把任何结构证据缺失 Case 恢复为
Positive，失败数由 64 增至 73。FAR 下降来自更多拒绝而非证据恢复，且 E2E
退化，因此不设为默认配置；正式 release 继续使用
`configs/evidence/gate_calibrated_v0.3.json`。本实验未读取或重跑 test。

## 12. Bounded Agent 架构收口验证（2026-09-05）

本轮只将检索、扩源、生成和拒答显式收束为最多 4 步的 Controller，并移除 Analyzer 对
benchmark 原句的不可回答捷径；未修改数据标签、Retriever 算法或 frozen test。全量测试
`261 run / 257 passed / 4 skipped / 0 failed`，fixed regression `1/1` 通过。

v0.3/dev CI Gate 的 Recall@5、MRR、NDCG@5 分别为 55.42%、49.88%、48.78%，质量项没有
变化；但本机 candidate/reference P95 为 1770.79/1400.81 ms，增长 26.4%，略超 25% 工程预算，
因此本次 Gate 总体为 failed。该负结果保留在 `reports/ci/evaluation-gate-v2/`，没有重跑挑选
更好延迟，也没有读取或重新运行 frozen test。

## 13. Full E2E 容量验证（2026-09-06）

本轮使用 12 条固定 Query 覆盖 exact fact、semantic、multi-source 和 relation reasoning，
其中约 30% 请求携带 Session。正式链路包含 FastAPI、Adaptive Retrieval、Evidence Gate、
Context Engine、Generator/Validator、请求级 Trace、真实 PostgreSQL 与 Redis；真实模型档使用
DeepSeek。PostgreSQL、Redis、Neo4j 和 pgvector 均通过真实连接预检，但锁定 Retriever 的逐请求
Dense/Graph 召回仍使用版本化本地 exact index/graph artifact，因此不能声称本轮使用了
pgvector/Neo4j 执行在线检索。

| Mode / concurrency | Requests | RPS | P95 | Failure |
|---|---:|---:|---:|---:|
| Deterministic / 1 | 10 | 0.36 | 4.73 s | 0.00% |
| Deterministic / 5 | 2 | 0.08 | 25.68 s | 0.00% |
| Real DeepSeek / 1 | 35 | 0.20 | 7.44 s | 8.57% |
| Real DeepSeek / 2 | 33 | 0.19 | 17.01 s | 6.06% |
| Real DeepSeek / 5 | 26 | 0.14 | 51.90 s | 3.85% |
| Real DeepSeek / 10 | 25 | 0.13 | 90.32 s | 28.00% |

真实模型阶段完成 72 次 Generation 调用，Provider 返回 input/output/total tokens 为
106,220/14,101/120,321；配置未固化官方单价，因此成本不作推算。并发 10 出现 6 次 90 秒
请求超时且没有 Provider 429，Trace 中 Retrieval P95 79.90 秒、Generation P95 15.70 秒，
线程峰值 916，说明当前主要瓶颈是共享 CPU 上的检索模型推理线程过度订阅和排队。真实模型
软拐点为并发 2；完整逐档结果和资源采样见
[`p1-full-e2e-20260906`](../../reports/loadtest/p1-full-e2e-20260906/report.md)。

基于上述线程峰值，保持 Retriever、Gate 和数据不变，仅设置
`OMP/MKL/OPENBLAS_NUM_THREADS=1` 进行 deterministic 对照：并发 5 从 0.08 RPS、P95
25.68 秒改善为 7.02 RPS、P95 0.94 秒；并发 10 达到 7.07 RPS、P95 1.87 秒，线程峰值
由原始档的 976 降至 18。优化后并发 20 为软拐点，并发 50 仍无失败但吞吐回落至 5.91 RPS。
真实 DeepSeek 档未在该优化后重跑，因此不把 deterministic 改善幅度外推为真实 Provider 容量。
本次固定 Query 的 Reranker 调用数为 0，CrossEncoder 的 Full E2E 容量路径为 **NOT TESTED**；
现有 CrossEncoder 结论仍只来自 dev 消融和自动化测试。
