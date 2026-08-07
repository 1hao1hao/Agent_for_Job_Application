# P0-D5 最终报告

## 范围

- 数据集：`evalrag_v0.2`
- Frozen test：40 条，其中可回答 30 条、不可回答 10 条
- 执行规则：每种声明策略只运行一次，查看 test 后不调参
- 最终检索：`hybrid`；Reranker 在 dev 负向消融后关闭

## Frozen Retrieval

| 策略 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|
| keyword | 55.56% | 67.22% | 60.83% |
| dense | 56.67% | 74.44% | 59.28% |
| hybrid | 68.33% | 74.44% | 66.78% |
| hybrid_rerank_candidate | 60.00% | 73.33% | 61.67% |

Hybrid 的 frozen-test MRR 最高（66.78%），Recall@5 与 Dense 并列最高（74.44%），因此被选为最终策略。

## Reranker 决策

仅在 dev 上运行的 token-overlap reranker 将 Recall@3 从 80.83% 降至 73.33%，Recall@5 从 85.56% 降至 83.89%，MRR 从 83.19% 降至 79.44%，因此最终关闭。仓库保留真实 CrossEncoder adapter 和 Fake scorer 测试，但本报告不声称获得了神经 Reranker 实测结果。

## 端到端 Extractive Baseline

| 指标 | 结果 |
|---|---:|
| Citation Validity | 100.00% |
| Key-Point Coverage | 80.00% |
| Abstention Accuracy | 90.00% |
| Unsupported Answer Rate | 0.00% |
| End-to-End Success Rate | 67.50% |
| 端到端延迟 P50 | 303.78 ms |
| 端到端延迟 P95 | 1498.39 ms |

Generator 是 deterministic extractive 实现，因此 token 与 API 估算成本不可用。Citation Validity 只检查引用 ID 是否存在于当前 Context；Unsupported Answer Rate 是 extractive contract 级结果，不是独立语义核验。

## Regression

- Fixed：1/1 通过（100.00%）
- Open：3 条，不进入通过率分母

## 证据路径

- Reranker dev 消融：`reports/ablations/p0-d5-reranker-dev-20260804/`
- Frozen retrieval 对照：`reports/comparisons/p0-d5-v02-frozen-test-20260804/`
- 端到端 case results 与 traces：`reports/runs/p0-d5-v02-frozen-test-20260804-extractive-e2e/`
- Executable regression：`reports/regression/p0-d5-v0.2/`

## 审计说明

Frozen predictions 保存后发现指标实现将“可回答但拒答”误记为 unknown，而不是失败。系统没有重新运行 predictions，只从原始 `case_results.jsonl` 重算指标；修复前摘要保留为 `summary_before_metric_fix.json`。
