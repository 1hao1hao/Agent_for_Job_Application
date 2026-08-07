# DeepSeek 真实 LLM 端到端报告

## 固定配置

- Model：`deepseek-v4-flash`
- Mode：非思考模式
- Prompt：`p0-deepseek-json-v1`
- Dataset：`evalrag_v0.2`，80 dev / 40 frozen test
- Pipeline：Hybrid Router -> Hybrid Retriever -> Evidence Gate -> Context -> DeepSeek JSON Generator -> Citation Validator -> Trace

## 结果

| Split | Cases | Citation Validity | Key-Point Coverage | Abstention Accuracy | Unsupported Answer Rate | E2E Success | P95 Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | 80 | 100.00% | 48.89% | 100.00% | 0.00% | 62.50% | 5774.24 ms |
| frozen test | 40 | 100.00% | 68.89% | 100.00% | 0.00% | 67.50% | 4136.71 ms |

## Token 与成本

| Split | LLM Calls | Input | Output | Total | Estimated Cost |
|---|---:|---:|---:|---:|---:|
| dev | 35 | 46293 | 8697 | 54990 | $0.008916 |
| frozen test | 25 | 33170 | 5977 | 39147 | $0.006317 |

价格按 2026-08-04 官方美元单价快照估算；实际账单以 Provider 为准。

## 真实失败

- dev：{'key_point_incomplete': 3, 'unexpected_abstention': 26, 'router_wrong': 1}
- frozen test：{'unexpected_abstention': 7, 'router_wrong': 1, 'retrieval_miss': 5}
- frozen `citation_invalid`：['v02_multi_026']
- 两个 split 均未发生 JSON format retry；Validator 各拦截 1 条 `sufficient=false` 但仍携带 citations 的不一致输出。
- 当前首要失败是 Evidence Gate 的 required-source coverage 过严导致 unexpected abstention。

## 审核边界

Answered case 由 Codex 逐条对照引用 Context 检查事实支持性，不是独立人工审核，也不是 LLM-as-a-Judge。Citation Validity 只证明引用 ID 合法；Key-Point Coverage 使用规范化子串匹配。

## 工件

- Dev：`reports/runs/p0-d5-v02-dev-20260804-deepseek-v4-flash/`
- Frozen test：`reports/runs/p0-d5-v02-frozen-test-20260804-deepseek-v4-flash/`
- 每个目录包含 config、case results、failures、traces、latency、support review 和 summary。
