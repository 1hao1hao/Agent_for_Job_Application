# 最后一次证据质量修复与冻结

**PROJECT_FROZEN**。实现 Commit：`7a6cef3`。冻结清单见
[project_frozen_v0.3.json](../../../configs/retrieval/project_frozen_v0.3.json)。

## 最终架构和默认状态

保留 `EvidencePlanner → Orchestrator → Assembler → Verifier` 实验实现，但冻结
**Evidence-Gap** 为当前候选；线上默认仍为
`configs/retrieval/adaptive_evidence_rerank_v0.3.json`。本轮工程优化结束，不继续架构重构。

判断理由：新方案最终证据保留改善，但原始候选 Recall@5、MRR 仍低于 Evidence-Gap，
未满足“两个阶段均不劣”的既定标准。不能只选有利阶段，更不能据此宣称回答准确率提高。

## 实际修复

- 重排只处理头部但保留尾部；完整路径识别前不做 Top-K 截断。
- 图路径选出覆盖各边的原文见证，不强制装入所有替代出处；每个 Chunk 保存所有路径成员关系。
- 完整组与单条证据共同按增量槽覆盖、相关 rank 和预算装箱；共享 Chunk 只计费一次，输出保留相关性顺序。
- 不完整路径不再连带删除其他完整路径或普通事实证据；完整组检查仍然必需。
- Pipeline 注入实际格式器、Token 估计器、Prompt 预留和字符预算；Context 避免重复前置裁剪，并支持共享路径成员。
- Gate、授权范围、评测标签未放宽；新增 3 个针对性测试，没有新依赖或数据库迁移。

代码：[`evidence_plan.py`](../../../src/intern_rag/retrieval/evidence_plan.py)、
[`evidence_oriented.py`](../../../src/intern_rag/retrieval/evidence_oriented.py)、
[`graph.py`](../../../src/intern_rag/retrieval/graph.py)、
[`context_engine.py`](../../../src/intern_rag/agent/context_engine.py)、
[`pipeline.py`](../../../src/intern_rag/agent/pipeline.py)。
旧 Adaptive/Gap 仅增加完整候选诊断，不更改其默认检索输出。

## 一次集中 dev 对照

固定 `evalrag_v0.3/dev`，160 Case / 120 可答 / 40 关系 Case；同一 Router、模型缓存、
两线程 CPU、Evidence 1200 估计 Tokens、Prompt 1800、证据 4000 字符。
参考策略使用相同预算 Assembler 审计适配器处理完整候选，**不是改动其生产输出**。
完整预测来自实际 Retriever；gold 仅在预测完成后用于评分。

| 策略 | 阶段 | Recall@5 | MRR | 标注边覆盖完整率 | Top-5 零召回/120 | 候选命中后全部丢失/120 |
|---|---|---:|---:|---:|---:|---:|
| 旧 Adaptive | 原始候选 | 51.25% | 52.56% | 0.0% | 51 | 0 |
| 旧 Adaptive | Assembler | 50.42% | 52.08% | 0.0% | 51 | 17 |
| 旧 Adaptive | Context | 50.42% | 52.08% | 0.0% | 51 | 17 |
| Evidence-Gap | 原始候选 | 72.92% | 68.59% | 57.5% | 25 | 0 |
| Evidence-Gap | Assembler | 63.75% | 66.25% | 10.0% | 35 | 18 |
| Evidence-Gap | Context | 63.75% | 66.25% | 10.0% | 35 | 18 |
| Oriented 修复后 | 原始候选 | 69.58% | 67.88% | 67.5% | 32 | 0 |
| Oriented 修复后 | Assembler | 78.33% | 70.56% | 65.0% | 19 | 9 |
| Oriented 修复后 | Context | 78.33% | 71.11% | 65.0% | 19 | 9 |

Recall@5 为可答 Case 的 gold Chunk top-5 召回率均值；MRR 在该阶段完整有序输出中
计算第一个 gold 的倒数排名，不截成 MRR@5。路径指标是关系 Case 的标注边覆盖，
不等于自然语言推理正确率。零召回只看 top-5；“全部丢失”比较完整候选与完整阶段集合，
因此二者不能直接相减。装箱过滤噪声可使后段 gold 进入 top-5，不表示重新检索了新证据。

| 策略 | 本地确定性检索+组装+Context P95 | 平均逻辑检索调用 |
|---|---:|---:|
| 旧 Adaptive | 1100.08 ms | 1.0000 |
| Evidence-Gap | 1087.64 ms | 2.2750 |
| Oriented 修复后 | 1099.59 ms | 2.3625 |

逻辑调用按策略 Trace 的操作计数，不等于网络 API 次数或底层 encoder 次数；
P95 不包含真实 LLM，不是线上 SLA，也不与历史不同线程数的延迟作因果比较。

## 丢失与退化核查

上一轮 41 条“候选曾命中、最终零召回”案例，本轮 Context 找回 **36/41**。
仍未恢复：`v03_cross_source_002`、`004`、`008`、`010`、`022`。
这 5 条原始候选分别有 80/79/76/80/51 条，装箱后均只剩 3 条；其丢弃记录均为
`effective_budget`，Context 没有再丢弃。损失定位在 Assembler 的预算选择，不应归因于
LLM 或宣称 Retriever 从未召回。具体 gold 的去向可从逐阶段 Chunk IDs 复查。

相对 Evidence-Gap 的逐 Case Recall@5：

| 阶段 | 改善 | 退化 | 相同 |
|---|---:|---:|---:|
| 原始候选 | 12 | 19 | 89 |
| Assembler | 28 | 12 | 80 |
| Context | 28 | 12 | 80 |

任一阶段退化的 Case 并集为 **21 条**，完整 Case IDs 和前后指标保留在
[summary.json](summary.json) 的 `regressions`，41 条逐项核查在 `audit_previous_41`。
新方案仍有 19 条 Context top-5 零召回、9 条候选命中后全丢失；未隐去这些失败。
旧 Oriented 56.25% / 12.5% 是旧打包输出指标，不能直接与本表原始候选指标比较。

## 验证和未解决项

全量 **296 passed / 6 skipped**，定向 **59 passed**，修改文件 Ruff 通过，
`git diff --check` 通过。命令和范围见 [verification.json](verification.json)。

DeepSeek 无敏感内容连接 Smoke 成功：1 次生成、53 input / 5 output / 58 total Tokens；
价格未配置，成本 **NOT AVAILABLE**。安全审核未批准本地证据外发，
少量配对 E2E **NOT TESTED / E2E NOT VERIFIED**；不得把连接成功视为 E2E 成功。
Frozen test **NOT TESTED**（任务明确不运行）；真实数据库服务验证本轮 **NOT TESTED**。

剩余技术债：原始候选排名仍退化；关系组与长跨源证据仍竞争预算；Token 使用确定性
近似估计，不是供应商精确 tokenizer；结构就绪不等于事实支持。以上均保留为限制，
不在本轮继续调参或添加新架构。

复现命令：

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python scripts/run_evidence_freeze.py
```

每条 Case 断点保存，源码/配置/语料 SHA-256 见 [manifest.json](manifest.json)；
修改实现后不可复用该 Run 断点。实验执行时父提交为 `85bc549`，随后相同源码提交为 `7a6cef3`。
原始工件为 `adaptive_case_results.jsonl`、`evidence_gap_case_results.jsonl`、
`evidence_oriented_case_results.jsonl`、`summary.json`；未重跑或挑选更好的 Run。
