# EvalRAG v0.2 Frozen Test Retrieval Comparison

- Split: `test`; 40 cases；每种策略只运行一次。
- 配置在查看 test 前已冻结；查看结果后不继续调参。
- dev 决策已禁用 token-overlap Reranker，candidate 仅作为固定负对照。

| Strategy | Router Accuracy | Recall@3 | Recall@5 | MRR | Retrieval P95 ms |
|---|---:|---:|---:|---:|---:|
| keyword | 0.9500 | 0.5556 | 0.6722 | 0.6083 | 99.545 |
| dense | 0.9500 | 0.5667 | 0.7444 | 0.5928 | 494.053 |
| hybrid | 0.9500 | 0.6833 | 0.7444 | 0.6678 | 802.503 |
| hybrid_rerank_candidate | 0.9500 | 0.6000 | 0.7333 | 0.6167 | 605.864 |

## Boundary

这些指标只衡量 Router/Retrieval，不等于答案准确率。所有失败 case 均保留在各 run 的 failures.jsonl。
