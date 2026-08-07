# P0-D6 Semantic Metrics 与 Claim-Level Grounding Audit

## 实验边界

- Dataset：`evalrag_v0.2`，80 dev / 40 frozen test。
- Grader：`deepseek-v4-flash`，Key-Point `p0-d6-key-point-v1` 与 Grounding
  `p0-d6-grounding-v1`。
- 只读取 P0-D5 已保存 predictions 与 Trace，没有重新运行 Router、Retriever、Generator。
- Judge 与 Generator 使用同一模型家族，不是独立人工审核，存在自评偏差。

## Key-Point Coverage 对照

| Split | Lexical | Semantic | Delta | Improved | Regressed | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| dev | 48.89% | 54.44% | +5.56 pp | 9 | 1 | 0 |
| frozen test | 68.89% | 74.44% | +5.56 pp | 5 | 0 | 0 |

dev 与 frozen 各保存 10 条差异/一致 Case。语义评分能识别“岗位时效性/下架版本”、
“工具调用/外部操作”等同义表达；dev 的 `v02_multi_005` 出现一次退化，未删除。

## Claim-Level Grounding

| Split | Answered | Known | Unknown | Unsupported Answer Rate | E2E |
|---|---:|---:|---:|---:|---:|
| dev | 34 | 32 | 2 | 3.12% | unavailable |
| frozen test | 23 | 20 | 3 | 0.00% | unavailable |

dev 找到 1 条 unsupported factual claim：回答断言未引用的 resume chunks 主题与问题
不同。frozen 的 UAR 0% 只表示 20 条已知 verdict 中没有 unsupported；另有 3 条
unknown，因此不能写成“全部 23 条回答零幻觉”，E2E 也按协议保持 unavailable。

## Judge 延迟、Token 与成本

| Split | Calls | Unavailable | P50 | P95 | Tokens | Estimated Cost |
|---|---:|---:|---:|---:|---:|---:|
| dev | 68 | 2 | 3897.90 ms | 9799.49 ms | 79902 | $0.010840 |
| frozen test | 46 | 3 | 4059.23 ms | 8803.11 ms | 54168 | $0.010669 |

## Source Prediction Hashes

```text
dev case results: c4fc5233d221bec2d85eaced11253fc27dbb1298977882555bb82a48eba89f12
dev traces:       24ec23fa2d920ac31916f8dd8252ed8042073bf098ceec81c65ac438d85f2362
test case results:bf020ca916c0dba1ab0ae85a82d9a16fa84f2e2421e893d6d16c89bda96b9061
test traces:      8a3af2a1481810a17d07fe37a87cd56dbac406e373ff0a0bf4779836a37c6615
```

## 工件

- Dev：`reports/runs/p0-d6-v02-dev-20260804-deepseek-v4-flash-v2/`
- Frozen：`reports/runs/p0-d6-v02-test-20260804-deepseek-v4-flash/`
- 每个目录包含 `point_verdicts.jsonl`、`claim_verdicts.jsonl`、
  `grader_calls.jsonl`、`case_results.jsonl`、`failures.jsonl` 和报告。

## 结论边界

Semantic Coverage 是模型评分，不等于答案准确率。Citation Validity 仍只验证 ID；
Claim-Level Grounding 才核对事实与引用文本。unknown 不进入 UAR 已审核分母，并使
完整 E2E unavailable，避免把 Judge 失败伪装成 supported。
