# EvalRAG P0-P1 最终实验报告

## 1. 实验范围

- Dataset：`evalrag_v0.2`，项目自建并审核的中文求职场景 benchmark。
- Corpus：100 文档、310 Chunk；JD、简历、面经、项目日志、用户画像各 20 份。
- Query：120 条，单来源、多来源、语义改写、不可回答各 30 条；80 dev / 40 frozen test。
- Frozen 规则：只在 dev 选择配置；test 上每个声明策略运行一次，查看结果后不调参。
- Corpus 统计：[corpus_stats.json](../../data/evaluation/corpus_stats.json)；标签校验：[evalrag_v0.2_validation.json](../../data/evaluation/evalrag_v0.2_validation.json)。

Chunk 长度为 121--376 字符，均值 253.15，P50 261，P95 277；空文档和重复
content hash 均为 0。数据为项目自建半真实材料，不等同于线上业务数据。

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

### Context Engine 与分层记忆消融

`evalrag_context_v0.1/dev` 包含 60 组、每组 5 轮，共 300 turns，覆盖指代、省略、历史约束、
跨会话记忆、冲突、主题切换、多来源和不可回答。四种策略运行相同 Case 和 220 token 预算：

| Strategy | Follow-up | Semantic KPC | Grounding | Prompt tokens | Repeat reads | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| No memory | 36.67% | 28.30% | 36.67% | 38.18 | 0 | 0.101 |
| Recent window | 36.67% | 28.30% | 36.67% | 67.18 | 3 | 0.134 |
| Summary + recent | 100.00% | 100.00% | 100.00% | 76.62 | 3 | 0.145 |
| Semantic memory | 100.00% | 100.00% | 100.00% | 66.22 | 0 | 296.933 |

Semantic KPC 使用固定 `BAAI/bge-small-zh-v1.5` revision
`7999e1d3359715c523056ef9478215996d62a620`。该实验由 Context Engine 产生确定性摘录，
不调用 LLM，数据是 scenario-authored AI-assisted，因此 100% 只说明构造场景中的必要事实被保留，
不能写作真实多轮问答准确率。Semantic Memory 从加入同用户干扰项的候选集合中执行真实 BGE
top-k，质量保持的代价是 CPU P95 明显增加。逐 Case 裁剪和 15 个差异 Case 见
[P1-D5 report](../../reports/ablations/p1-d5-context-memory-v01-dev-20260816-bge/report.md)。

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

| Strategy | Follow-up | Key-point | Prompt Tokens | Token Reduction vs Raw | Repeat Reads | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| No memory | 36.67% | 28.30% | 38.18 | 31.50% | 0 | 0.101 |
| Recent window | 36.67% | 28.30% | 67.18 | 0.00% | 3 | 0.134 |
| Summary + recent | 100.00% | 100.00% | 76.62 | 0.00% | 3 | 0.145 |
| Semantic memory | 100.00% | 100.00% | 66.22 | 4.18% | 0 | 296.933 |

Context 结论不变：Semantic Memory 在该确定性场景集上保持全部 Follow-up 要点，
同时减少 Prompt Token 和重复读取，代价是 BGE 检索带来约 297 ms P95。它不是
真实 LLM 多轮答案准确率。

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

当前主要限制是 benchmark 规模和分布仍偏项目化，Hybrid CPU 延迟较高，Evidence
Gate 对多来源覆盖要求过严，Grounding Judge 与 Generator 属于同模型家族且存在
unknown。FastAPI、Docker、可视化和高并发压测未实现，也不计入 P0 成果。
