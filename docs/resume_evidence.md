# EvalRAG Resume Evidence

## 用途

该文档连接“简历表述”和“仓库证据”。所有数字必须能回到固定 dataset、run 和报告。

## 当前可证明事实

| 表述 | 代码/数据证据 | 状态 |
|---|---|---|
| 支持五类中文多源资料 | `data/raw/`、`ingestion` | verified |
| 统一 Document/Chunk Schema 与 metadata 继承 | `src/intern_rag/ingestion/`、tests | verified |
| 规则 Source Router | `src/intern_rag/routing/`、tests | verified |
| Keyword Retrieval 支持 source filter 和 top-k | `src/intern_rag/retrieval/`、tests | verified |
| Answer + Citations 与证据不足结果 | `src/intern_rag/agent/answer.py`、tests | verified |
| AgentTrace JSONL | `src/intern_rag/tracing/`、tests | verified |
| Recall@k 与 Router Accuracy 代码 | `src/intern_rag/evaluation/`、tests | verified |
| Rag 请求响应契约与预算感知 Context Builder | `src/intern_rag/agent/schemas.py`、`context.py`、tests | verified |
| 142 个自动化测试覆盖核心链路、真实 PostgreSQL/Redis 集成、DeepSeek adapter、Semantic/Grounding grader 与固定 Demo 导出 | `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'` | verified at 2026-08-07 |
| 30 份五类语料和 60 条四类 Query 已审核 | `data/evaluation/`、审核报告 | verified |
| Keyword dev 正式 baseline 与标准 Run Artifacts | `reports/runs/keyword-dev-formal-v0.1-20260730/` | verified |
| 100 文档、310 Chunk、120 Query 的 v0.2 检索 benchmark | `data/evaluation/`、`data/processed/chunks/` | verified; project-authored synthetic benchmark |
| Keyword、Dense、Hybrid 统一接口和 dev 消融 | `src/intern_rag/retrieval/`、`reports/ablations/p0-d3-v02-dev-20260801/` | verified |
| Rule、Semantic、Hybrid Router 同集 dev 对照 | `reports/ablations/p0-d4-v02-dev-20260803/` | verified |
| Reranker 小接口、CrossEncoder adapter 与负向 dev 消融 | `src/intern_rag/retrieval/rerank.py`、`reports/ablations/p0-d5-reranker-dev-20260804/` | verified; neural weights not executed |
| 40 条 frozen test 检索对照 | `reports/comparisons/p0-d5-v02-frozen-test-20260804/` | verified; no post-test tuning |
| Citation、Key-Point、Abstention 与 E2E Metrics | `src/intern_rag/evaluation/metrics.py`、P0-D5 final report | verified for deterministic extractive baseline |
| fixed/open executable regression | `tests/regression/`、`reports/regression/p0-d5-v0.2/` | verified; fixed 1/1, open 4 |
| DeepSeek 真实 Pipeline dev/frozen Run | `reports/final/p0-d5-live-llm-v0.2/` | verified; 80 dev / 40 test |
| Semantic Key-Point 与 Claim-Level Grounding 离线审核 | `src/intern_rag/evaluation/semantic_audit.py`、`reports/final/p0-d6-semantic-grounding-v0.2/` | verified；frozen 3 条 unknown |
| 单来源、多来源和拒答固定 Demo | `scripts/export_fixed_demos.py`、`examples/fixed_demos/` | verified; replay saved frozen artifacts |
| 标准 BM25、离线 index 与统一 Retriever 接口 | `src/intern_rag/retrieval/bm25.py`、`tests/unit/test_bm25_retrieval.py` | verified |
| Keyword/BM25/Dense/BM25+Dense RRF dev 消融 | `reports/ablations/p1-d1-bm25-dev-20260807/` | verified; BM25 not selected as final config |
| FastAPI Query/Trace/Evaluation Job 接口 | `src/intern_rag/serving/`、API tests | verified locally |
| PostgreSQL 状态持久化与 Redis Evaluation Worker | `src/intern_rag/persistence/`、`src/intern_rag/worker/`、stack integration test | verified locally with real services |
| Docker Compose 四服务编排 | `compose.yaml`、`.github/workflows/p1-service.yml` | config verified; runtime CI pending |

这些事实不能被扩写为：

- 已有真实业务准确率。
- 已证明预训练神经 embedding 优于 Keyword。
- 神经 CrossEncoder Reranker 带来指标提升。
- contract-level Unsupported Answer Rate 等于真实 LLM 幻觉率。
- BM25 或 BM25+Dense 已优于当前 frozen Hybrid 最终配置。
- Docker Compose 已完成运行验证（在 CI 成功前）。

## 最终简历描述

### 项目描述

> 面向多源异构知识问答开发可观测、可评测的 RAG Agent Harness，围绕 Source 路由、混合检索、证据生成、运行 Trace、离线评测和 Regression 进行系统设计，重点解决传统 RAG 链路难定位、策略迭代缺少量化依据和失败问题重复出现。

### 核心工作

**端到端 RAG Harness：**

> 基于原生 Python 实现 Router、Keyword/Dense/Hybrid Retrieval、Context Builder、结构化 LLM Generation、Citation Validation 和 Evidence Sufficiency，处理 100 份文档与 310 个知识片段，支持单来源、多来源回答和证据不足拒答。

证据：

- Code：`src/intern_rag/agent/`、`src/intern_rag/retrieval/`
- Report：`reports/final/p0-d5-v0.2/`
- Trace：`reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/traces.jsonl`

**检索策略优化：**

> 在 40 条冻结测试 Query 上对比 Keyword、Dense 和 RRF Hybrid；Hybrid 相比 Keyword 将 Recall@5 从 67.22% 提升至 74.44%，MRR 从 60.83% 提升至 66.78%。针对前排排序增加 Reranker dev 消融，发现质量和延迟均退化后关闭该配置，并保留差异 Case 与负向结果。

证据：

- Dataset：`data/evaluation/evalrag_v0.2.jsonl`
- Baseline Run：`reports/runs/p0-d5-v02-frozen-test-20260804-keyword/`
- Final Run：`reports/runs/p0-d5-v02-frozen-test-20260804-hybrid/`
- Comparison：`reports/comparisons/p0-d5-v02-frozen-test-20260804/`

**可观测与可靠性：**

> 设计请求级 Trace，记录路由、召回、上下文、模型输出、引用、重试、分阶段耗时和 token；在 40 条 frozen test 的真实模型 Pipeline 上取得 100% Citation Validity 和 100% Abstention Accuracy，端到端 P95 为 4136.71 ms；进一步通过 Claim-Level Grounding 发现 3 条审核 unknown，避免沿用缺少逐 claim 证据的 E2E 数字。

证据：

- Trace schema：`src/intern_rag/tracing/trace.py`
- Reliability Report：`reports/final/p0-d5-live-llm-v0.2/report.md`
- Traces：`reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/traces.jsonl`

**评测与回归：**

> 搭建覆盖 Router Accuracy、Recall@k、MRR、Citation Validity、Key-Point Coverage、Abstention Accuracy 和 End-to-End Success Rate 的离线评测；建立 fixed/open Regression Case 机制，已修复路由失败回归 1/1 通过，4 条 open case 不计入通过率。

证据：

- Metrics Code：`src/intern_rag/evaluation/metrics.py`
- Evaluation Report：`reports/final/p0-d5-v0.2/`
- Regression Tests：`tests/regression/test_regression_suite.py`

## 指标登记表

| 指标 | Dataset / Split | Baseline Run | Final Run | 真实值 | 报告路径 |
|---|---|---|---|---:|---|
| Router Accuracy | `evalrag_v0.1/dev` | `keyword-dev-formal-v0.1-20260730` | v0.1 baseline only | 22.50% | `reports/runs/keyword-dev-formal-v0.1-20260730/summary.json` |
| Recall@3 | `evalrag_v0.1/dev` | `keyword-dev-formal-v0.1-20260730` | v0.1 baseline only | 39.72% | `reports/runs/keyword-dev-formal-v0.1-20260730/summary.json` |
| Recall@5 | `evalrag_v0.1/dev` | `keyword-dev-formal-v0.1-20260730` | v0.1 baseline only | 53.61% | `reports/runs/keyword-dev-formal-v0.1-20260730/summary.json` |
| MRR | `evalrag_v0.1/dev` | `keyword-dev-formal-v0.1-20260730` | v0.1 baseline only | 45.89% | `reports/runs/keyword-dev-formal-v0.1-20260730/summary.json` |
| Keyword Recall@5 | `evalrag_v0.2/dev` | `p0-d3-v02-dev-20260801-keyword` | 当前 dev 对照 | 80.56% | `reports/runs/p0-d3-v02-dev-20260801-keyword/summary.json` |
| Hybrid Recall@3 | `evalrag_v0.2/dev` | Keyword 73.33% | BGE+RRF Hybrid | 80.83% | `reports/ablations/p0-d3-v02-dev-20260801/summary.json` |
| Hybrid Recall@5 | `evalrag_v0.2/dev` | Keyword 80.56% | BGE+RRF Hybrid | 85.56% | `reports/runs/p0-d3-v02-dev-20260801-hybrid/summary.json` |
| Hybrid MRR | `evalrag_v0.2/dev` | Keyword 85.97% | BGE+RRF Hybrid | 83.19% | `reports/ablations/p0-d3-v02-dev-20260801/summary.json` |
| Hybrid Retrieval P95 | `evalrag_v0.2/dev` | Keyword 14.09 ms | BGE+RRF Hybrid | 1095.15 ms | `reports/ablations/p0-d3-v02-dev-20260801/summary.json` |
| Keyword Recall@5 | `evalrag_v0.2/test` | Keyword | frozen comparison | 67.22% | `reports/comparisons/p0-d5-v02-frozen-test-20260804/summary.json` |
| Hybrid Recall@3 | `evalrag_v0.2/test` | Keyword 55.56% | BGE+RRF Hybrid | 68.33% | `reports/comparisons/p0-d5-v02-frozen-test-20260804/summary.json` |
| Hybrid Recall@5 | `evalrag_v0.2/test` | Keyword 67.22% | BGE+RRF Hybrid | 74.44% | `reports/comparisons/p0-d5-v02-frozen-test-20260804/summary.json` |
| Hybrid MRR | `evalrag_v0.2/test` | Keyword 60.83% | BGE+RRF Hybrid | 66.78% | `reports/comparisons/p0-d5-v02-frozen-test-20260804/summary.json` |
| Citation Validity | `evalrag_v0.2/test` | extractive E2E 100% | DeepSeek live E2E | 100.00% | `reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/summary.json` |
| Key-Point Coverage | `evalrag_v0.2/test` | extractive E2E 80% | DeepSeek live E2E | 68.89% | `reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/summary.json` |
| Abstention Accuracy | `evalrag_v0.2/test` | extractive E2E 90% | DeepSeek live E2E | 100.00% | `reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/summary.json` |
| Semantic Key-Point Coverage | `evalrag_v0.2/test` | lexical 68.89% | DeepSeek semantic grader | 74.44%（+5.56 pp） | `reports/final/p0-d6-semantic-grounding-v0.2/report.md` |
| Unsupported Answer Rate | `evalrag_v0.2/test` | provisional Codex review 0% | claim-level grader | 0/20 known；另有 3 unknown，不写作零幻觉 | `reports/final/p0-d6-semantic-grounding-v0.2/report.md` |
| End-to-End Success Rate | `evalrag_v0.2/test` | extractive E2E 67.50% | P0-D6 claim-level audit | unavailable（3 条 Grounding unknown） | `reports/final/p0-d6-semantic-grounding-v0.2/report.md` |
| Retrieval P95 | `evalrag_v0.1/dev` | `keyword-dev-formal-v0.1-20260730` | v0.1 baseline only | 0.645 ms | `reports/runs/keyword-dev-formal-v0.1-20260730/summary.json` |
| End-to-End P95 | `evalrag_v0.2/test` | extractive 1498.39 ms | DeepSeek live E2E | 4136.71 ms | `reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/summary.json` |
| Live LLM Tokens | `evalrag_v0.2/test` | unavailable | DeepSeek live E2E | 39,147 | `reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/summary.json` |
| Estimated Cost | `evalrag_v0.2/test` | unavailable | DeepSeek live E2E | $0.006317 | `reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/summary.json` |
| Fixed Regression Pass Rate | `regression v0.2` | fixed cases | fixed cases | 100% (1/1) | `reports/regression/p0-d5-v0.2/summary.json` |
| BM25 Recall@5 | `evalrag_v0.2/dev` | Keyword 80.56% | BM25 | 78.89%（-1.67 pp） | `reports/ablations/p1-d1-bm25-dev-20260807/summary.json` |
| BM25 MRR | `evalrag_v0.2/dev` | Keyword 85.97% | BM25 | 68.33%（-17.64 pp） | `reports/ablations/p1-d1-bm25-dev-20260807/summary.json` |
| BM25+Dense Recall@3 | `evalrag_v0.2/dev` | Keyword 73.33% | BM25+Dense RRF | 73.89%（+0.56 pp） | `reports/ablations/p1-d1-bm25-dev-20260807/summary.json` |

## 失败案例模板

```text
Case ID:
Original Run / Trace:
现象:
Failure Type:
Root Cause:
修改假设:
代码改动:
修改前指标:
修改后指标:
额外延迟/成本:
Regression Case:
是否有其他退化:
```

至少完成一个案例后才能在简历中使用“评测驱动优化”或“Regression 闭环”。

## 投递前检查

- [x] 所有简历数字均有 dataset、run 和 report，不含目标值占位符。
- [x] 每个数字有 dataset、run 和 report。
- [x] 面试材料包含 Recall@5、MRR 和 Citation Validity 的口径。
- [x] 面试材料说明 dev/test 分割与 frozen test 边界。
- [x] 固定 Demo 包含 answered trace 和 abstain trace。
- [x] 面试材料包含一个失败闭环及其延迟边界。
- [x] README 与简历数字一致。
- [ ] GitHub 默认分支包含对应代码和报告。
