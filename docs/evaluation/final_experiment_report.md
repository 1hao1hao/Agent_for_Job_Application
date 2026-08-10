# EvalRAG P0 最终实验报告

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
从 700.21 ms 增至 902.75 ms，因此冻结配置为关闭。负向结果见
[Reranker Ablation](../../reports/ablations/p0-d5-reranker-dev-20260804/report.md)。

## 3. Router 与可靠性

Rule、Semantic、Hybrid Router 在相同 80 条 dev 上的 Accuracy 分别为 91.25%、
87.50%、96.25%；Hybrid P95 为 2600.27 ms，表明质量提升伴随显著 CPU 语义编码
开销。完整分类结果见 [Router Ablation](../../reports/ablations/p0-d4-v02-dev-20260803/report.md)。

真实 LLM frozen Trace 的阶段干预统计如下：

| 阶段结果 | Case 数 | 含义 |
|---|---:|---|
| Evidence Gate 判 sufficient | 25 | 进入 Generator，其中 23 answered、1 模型主动拒答、1 被 Validator 拦截 |
| Evidence Gate 按 unanswerable route 拒答 | 9 | 未调用 Generator |
| Evidence Gate 按 required sources 缺失拒答 | 6 | 可回答 Case 的 retrieval miss，属于意外拒答 |
| Citation Validator 拦截 | 1 | 模型 `sufficient=false` 却返回 citations，最终未作为 answered 返回 |
| 最终 answered | 23 | 返回合法 citation 的回答 |

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

`公司有哪些岗位？` 曾因过宽关键词“公司”被 Rule Router 错误路由为 JD 分析。Trace
把根因定位到规则命中，而不是 Retriever；删除这一主要变量后，Rule Router dev
Accuracy 从 88.75% 提升到 91.25%，完整 dev 未隐藏退化。该 Case 被固化为
`reg-router-company-list-v1`，fixed regression 1/1 通过；另有 4 条 open case 明确
不进入通过率分母。

- Root cause：[failure analysis](../../reports/failure_analysis/p0-d4-company-keyword/analysis.json)
- Before/after：[P0-D4 report](../../reports/ablations/p0-d4-v02-dev-20260803/report.md)
- Regression：[summary](../../reports/regression/p0-d5-v0.2/summary.json)

## 6. 结论与限制

EvalRAG P0 已形成 `Pipeline -> Trace -> Evaluation -> Failure -> Regression` 闭环，
并以 frozen test 记录最终配置的质量、延迟和成本。结果支持“Hybrid 提高当前 benchmark
的检索覆盖”“Validator 拦截结构矛盾输出”等局部结论，不支持“线上准确率”“零幻觉”
或“Reranker 已提升效果”。

当前主要限制是 benchmark 规模和分布仍偏项目化，Hybrid CPU 延迟较高，Evidence
Gate 对多来源覆盖要求过严，Grounding Judge 与 Generator 属于同模型家族且存在
unknown。FastAPI、Docker、可视化和高并发压测未实现，也不计入 P0 成果。
